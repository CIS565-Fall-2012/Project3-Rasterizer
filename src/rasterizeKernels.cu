// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_vbo_eyeCoord;
float* device_cbo;
int* device_ibo;
triangle* primitives;

//#define GEOMETRYSHADER
#define BackFaceCulling
//#define AntiAliasing
#define Clipping

bool firstTime=true;
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    depthbuffer[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
	  index=x + ((resolution.y-y) * resolution.x);
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: Implement a vertex shaderdevice_vbo, vbosize,ModelViewCudaMatrix,device_vbo_eyeCoord,ProjectionCudaMatrix
__global__ void vertexShadeKernel(float* vbo, int vbosize,cudaMat4 modelView, float* vbo_eyeSpace,cudaMat4 Projection ){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  // vertice in ObjSpace
	  glm::vec4 vertice4=glm::vec4(vbo[3*index],vbo[3*index+1],vbo[3*index+2],1.0f);  
	  // vertice in EyeSpace
	  vertice4=multiplyMV4(modelView,vertice4);

	  vbo_eyeSpace[3*index]=vertice4.x;
	  vbo_eyeSpace[3*index+1]=vertice4.y;
	  vbo_eyeSpace[3*index+2]=vertice4.z;

	  // vertice in ClippingSpace
	  vertice4=multiplyMV4(Projection,vertice4);
	  // vertice in NDC
	  if((abs(vertice4.w)>1e-3))
	   vertice4*=1.0f/vertice4.w;
	  vbo[3*index]=vertice4.x;
	  vbo[3*index+1]=vertice4.y;
	  vbo[3*index+2]=vertice4.z;
	  
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float* vbo_eyeSpace,bool* backfaceflags){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  triangle tr;
	  int VerticeIndice0=ibo[index*3];
	  int VerticeIndice1=ibo[index*3+1];
	  int VerticeIndice2=ibo[index*3+2];

	  tr.p0=glm::vec3(vbo[3*VerticeIndice0],vbo[3*VerticeIndice0+1],vbo[3*VerticeIndice0+2]);
	  tr.p1=glm::vec3(vbo[3*VerticeIndice1],vbo[3*VerticeIndice1+1],vbo[3*VerticeIndice1+2]);
	  tr.p2=glm::vec3(vbo[3*VerticeIndice2],vbo[3*VerticeIndice2+1],vbo[3*VerticeIndice2+2]);

	  tr.pe0=glm::vec3(vbo_eyeSpace[3*VerticeIndice0],vbo_eyeSpace[3*VerticeIndice0+1],vbo_eyeSpace[3*VerticeIndice0+2]);
	  tr.pe1=glm::vec3(vbo_eyeSpace[3*VerticeIndice1],vbo_eyeSpace[3*VerticeIndice1+1],vbo_eyeSpace[3*VerticeIndice1+2]);
	  tr.pe2=glm::vec3(vbo_eyeSpace[3*VerticeIndice2],vbo_eyeSpace[3*VerticeIndice2+1],vbo_eyeSpace[3*VerticeIndice2+2]);


	  tr.c0=glm::vec3(cbo[0],cbo[1],cbo[2]);
	  tr.c1=glm::vec3(cbo[3],cbo[4],cbo[5]);
	  tr.c2=glm::vec3(cbo[6],cbo[7],cbo[8]);
	  primitives[index]=tr;
	  
#ifdef BackFaceCulling
	  glm::vec3 normal=getNormalInEyeSpace(tr);
	  glm::vec3 eyeDir=glm::normalize(glm::vec3(0,0,0)-((tr.pe1+tr.pe2+tr.pe0)/3.0f));
	 if(glm::dot(eyeDir,normal)<0)
		backfaceflags[index]=true;
	 else
		backfaceflags[index]=false;
#endif
  }
}

__global__ void addMoreGeometry(triangle* primitives, int primitivesCount, triangle* newPrimitives, int newPrimitivesCount,bool* newbackfaceflags){
	 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(index<primitivesCount){
		  triangle tr=primitives[index];
		  glm::vec3 normal=getNormal(tr);
		  glm::vec3 normalE=glm::vec3(0,0,0);
		  float offset=0.001f;

		  glm::vec3 center01=(tr.p0+tr.p1)/2.0f+offset*normal;
		  glm::vec3 centerE01=(tr.pe0+tr.pe1)/2.0f+offset*normalE;
		 glm::vec3 Color01=tr.c2;
		 
		  glm::vec3 center02=(tr.p0+tr.p2)/2.0f+offset*normal;
		  glm::vec3 centerE02=(tr.pe0+tr.pe2)/2.0f+offset*normalE;
		   glm::vec3 Color02=tr.c1;

		  glm::vec3 center12=(tr.p1+tr.p2)/2.0f+offset*normal;
		  glm::vec3 centerE12=(tr.pe1+tr.pe2)/2.0f+offset*normalE;
		  glm::vec3 Color12=tr.c0;
		
		  triangle tr1;
		  tr1.c0=tr.c0;
		  tr1.c1=Color01;
		  tr1.c2=Color02;
		  tr1.p0=tr.p0;
		  tr1.p1=center01;
		  tr1.p2=center02;
		  tr1.pe0=tr.pe0;
		  tr1.pe1=centerE01;
		  tr1.pe2=centerE02;
		  newPrimitives[4*index]=tr1;


		  triangle tr2;
		  tr2.c0=Color01;
		  tr2.c1=tr.c1;
		  tr2.c2=Color12;
		  tr2.p0=center01;
		  tr2.p1=tr.p1;
		  tr2.p2=center12;
		  tr2.pe0=centerE01;
		  tr2.pe1=tr.pe1;
		  tr2.pe2=centerE12;
		  newPrimitives[4*index+1]=tr2;

		  triangle tr3;
		  tr3.c0=Color01;
		  tr3.c1=Color12;
		  tr3.c2=Color02;
		  tr3.p0=center01;
		  tr3.p1=center12;
		  tr3.p2=center02;
		  tr3.pe0=centerE01;
		  tr3.pe1=centerE12;
		  tr3.pe2=centerE02;
		  newPrimitives[4*index+2]=tr3;

		  triangle tr4;
		  tr4.c0=Color02;
		  tr4.c1=Color12;
		  tr4.c2=tr.c2;
		  tr4.p0=center02;
		  tr4.p1=center12;
		  tr4.p2=tr.p2;
		  tr4.pe0=centerE02;
		  tr4.pe1=centerE12;
		  tr4.pe2=tr.pe2;
		  newPrimitives[4*index+3]=tr4;

#ifdef BackFaceCulling
	  glm::vec3 eyeDir1=glm::normalize(glm::vec3(0,0,0)-((tr1.pe1+tr1.pe2+tr1.pe0)/3.0f));
	  glm::vec3 normal1=getNormalInEyeSpace(tr1);
	  if(glm::dot(eyeDir1,normal1)<0)
		  newbackfaceflags[4*index]=true;
	  else 
		  newbackfaceflags[4*index]=false;

	  glm::vec3 eyeDir2=glm::normalize(glm::vec3(0,0,0)-((tr2.pe1+tr2.pe2+tr2.pe0)/3.0f));
	  glm::vec3 normal2=getNormalInEyeSpace(tr2);
	  if(glm::dot(eyeDir2,normal2)<0)
		  newbackfaceflags[4*index+1]=true;
	  else 
		  newbackfaceflags[4*index+1]=false;

	  glm::vec3 eyeDir3=glm::normalize(glm::vec3(0,0,0)-((tr3.pe1+tr3.pe2+tr3.pe0)/3.0f));
	  glm::vec3 normal3=getNormalInEyeSpace(tr3);
	  if(glm::dot(eyeDir3,normal3)<0)
		  newbackfaceflags[4*index+2]=true;
	  else 
		  newbackfaceflags[4*index+2]=false;

	  glm::vec3 eyeDir4=glm::normalize(glm::vec3(0,0,0)-((tr4.pe1+tr4.pe2+tr4.pe0)/3.0f));
	  glm::vec3 normal4=getNormalInEyeSpace(tr4);
	  if(glm::dot(eyeDir4,normal4)<0)
		  newbackfaceflags[4*index+3]=true;
	  else 
		  newbackfaceflags[4*index+3]=false;
#endif

	 }
}

//TODO: Implement a rasterization method, such as scanline.
__device__ unsigned int lock = 0u;
__global__ void rasterizationKernel(triangle* primitivesList, int Count, fragment* depthbuffer, glm::vec2 resolution,bool* backfaceflags,bool* previousFlags){
  
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<Count){
	 #ifdef BackFaceCulling
	  if(backfaceflags[index])return;
	 #endif

	  triangle tr=primitivesList[index];
	  glm::vec3 minBoundary,maxBoundary;
	  getAABBForTriangle(tr,minBoundary,maxBoundary);

	  #ifdef Clipping
	  
	  if(minBoundary.x>1.0f)return;
	  if(maxBoundary.x<-1.0f)return;
	  if(minBoundary.y>1.0f)return;
	  if(maxBoundary.y<-1.0f)return;
      #endif

	  double dx=2.0/resolution.x;
	  double dy=2.0/resolution.y;
	  int start_x,end_x;
	  int start_y,end_y;

      if(minBoundary.x<-1.0f)start_x=0;
	  else start_x=(int)((minBoundary.x+1.0f)*resolution.x/2.0)-2;
	  if(minBoundary.y<-1.0f)start_y=0;
	  else start_y=(int)((minBoundary.y+1.0f)*resolution.y/2.0)-2;

	  if(maxBoundary.x>1.0f)end_x=resolution.x-1;
	  else end_x=(int)((maxBoundary.x+1.0f)*resolution.x/2.0)+2;
	  if(maxBoundary.y>1.0f)end_y=resolution.y-1;
	  else end_y=(int)((maxBoundary.y+1.0f)*resolution.y/2.0)+2;
	 
	 for(int j=start_y;j<=end_y;++j)
	 {
		 for(int i=start_x;i<=end_x;++i)

		 {
           	float x_value=-1.0f+(float)(i+0.5f)*dx;
			float y_value=-1.0f+(float)(j+0.5f)*dy;

#ifdef AntiAliasing
				int count=0;
				glm::vec3 barycoord=glm::vec3(0,0,0);
				glm::vec3 someposition=glm::vec3(0,0,0); 
				int somedepth=0;
				glm::vec3 averageColor=glm::vec3(0,0,0);
				for(int p=0;p<3;p++)
					for(int q=0;q<3;q++)
					{
						x_value+=(p-1)*dx/3.0f;
						y_value+=(q-1)*dy/3.0f;
						barycoord=calculateBarycentricCoordinate(tr,glm::vec2(x_value,y_value));
						if(isBarycentricCoordInBounds(barycoord))
						{
							++count;
							averageColor+=getColorAtCoordinate(barycoord,tr);
							someposition=getPosInEyeSpaceAtCoordinate(barycoord,tr);
							somedepth=calculateDepth(barycoord,tr);
							
						}
						

					}

				if(count==0)
				{
					continue;
				}

				fragment fr;
				fr.normal=getNormalInEyeSpace(tr);
				fr.color=averageColor*1.0f/9.0f;
				fr.position=someposition;
				fr.depth=somedepth;

				int DepthBufferIndex= i + (j * resolution.x);
				bool hold = true;
				while (hold)
				{
					if (atomicExch(&lock, 1u) == 0u) {
						fragment oldValue=depthbuffer[DepthBufferIndex];

						if(abs(oldValue.depth-fr.depth)<DEPTHPRECISION){
							bool partialFilled=previousFlags[DepthBufferIndex];
							if(partialFilled){
								fr.color+=oldValue.color;
								depthbuffer[DepthBufferIndex]=fr;
								if(count!=9) 
									previousFlags[DepthBufferIndex]=true;
								else 
									previousFlags[DepthBufferIndex]=false;
							}
							else{
							}
	
						}
						else if(fr.depth<oldValue.depth){
							if(count!=9) {
								previousFlags[DepthBufferIndex]=true;
							}else{ 
								previousFlags[DepthBufferIndex]=false;
							}
							depthbuffer[DepthBufferIndex]=fr;
						}
		
						 hold = false;
						atomicExch(&lock,0u);
					}
				} 

#else
			glm::vec3 barycoord=calculateBarycentricCoordinate(tr,glm::vec2(x_value,y_value));
				if(!isBarycentricCoordInBounds(barycoord)){
					continue;
				}else{
					fragment fr;
					fr.normal=getNormalInEyeSpace(tr);
					fr.color=getColorAtCoordinate(barycoord,tr);
					fr.position=getPosInEyeSpaceAtCoordinate(barycoord,tr);
					fr.depth=calculateDepth(barycoord,tr);
					int DepthBufferIndex= i + (j * resolution.x);
					bool hold = true;
					while (hold) 
					{
						if (atomicExch(&lock, 1u)==0u) 
						{
							fragment oldValue=depthbuffer[DepthBufferIndex];
								if(fr.depth<oldValue.depth){
									depthbuffer[DepthBufferIndex]=fr;
								}
							hold = false;
							atomicExch(&lock,0u);
						}
					} 
			  }

#endif	
		  } //for x
		}//for y
	}//index if
}
__global__ void initialdepthflags(glm::vec2 resolution, bool* flagarray,bool value){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      flagarray[index] = value;
    }

}
//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 light,glm::vec3 lightColor){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  //simple Phong Model
	  fragment fr=depthbuffer[index];
	  if(fr.depth==INT_MAX)return;
	  glm::vec3 IntersectionPosition=fr.position;
	  glm::vec3 lightDirection=glm::normalize(light-IntersectionPosition);
	  glm::vec3 normal=fr.normal;
	  glm::vec3 eyePos=glm::vec3(0.0f,0.0f,0.0f);
	  glm::vec3 viewdirection=glm::normalize(eyePos-IntersectionPosition);

	  float diffuse = clamp(glm::dot(normal, lightDirection), 0.0f,1.0f);
	  
	  glm::vec3 incidentDir=-lightDirection;
	  glm::vec3 reflectionedlightdirection=glm::normalize(incidentDir-2.0f*normal*glm::dot(incidentDir,normal));
	  float specular =clamp(pow(max(glm::dot(reflectionedlightdirection, viewdirection),0.0f), 20.0f), 0.0f,1.0f);

	  glm::vec3 newColor=glm::vec3(0,0,0);
	  newColor=0.1f*fr.color;
	  newColor+=specular*lightColor;
	  newColor+=diffuse* glm::vec3(lightColor.x*fr.color.x,lightColor.y*fr.color.y,lightColor.z*fr.color.z);
	  depthbuffer[index].color=newColor;

  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer,float frame){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
#ifdef AntiAliasing
	glm::vec3 previousColor=framebuffer[index];
    framebuffer[index] = previousColor*(frame)/(frame+1.0f)+depthbuffer[index].color/(frame+1.0f);
#else
    framebuffer[index] = depthbuffer[index].color;
#endif
  }
}


glm::mat4 calculateProjectMatrix(glm::vec2 resolution){
	
	float aspect=(float)resolution.x/(float)resolution.y;
	assert(zNear>0.0f);
	assert(zFar>0.0f);
	float range=1.0f;
	float left = -range * aspect;
	float right = range * aspect;
	float bottom = -range;
	float top = range;

	glm::mat4 result(0.0f);
	result[0][0] = (2.0f * zNear) / (right - left);
	result[2][0] = (right+left)/(right-left);
	result[1][1] = (2.0f * zNear) / (top - bottom);
	result[2][1] = (top+bottom)/(top-bottom);
	result[2][2] = - (zFar + zNear) / (zFar - zNear);
	result[2][3] = - 1.0f;
	result[3][2] = - (2* zFar * zNear) / (zFar - zNear);
	 
	return result;
}	

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
#ifdef AntiAliasing
	if(firstTime){
	 clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
	 firstTime=false;
	}
#else
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
#endif
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,0);
  frag.depth=INT_MAX;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_vbo_eyeCoord = NULL;
  cudaMalloc((void**)&device_vbo_eyeCoord, vbosize*sizeof(float));

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

   //get Trans Matrix
   glm::mat4 modelMatrix=glm::mat4(1.0f);
   glm::mat4 cameraMatrix=utilityCore::buildTransformationMatrix(glm::vec3(0,-0.25,-1),glm::vec3(0,0,0),glm::vec3(1,1,1));
   glm::mat4 projectionMatrix=calculateProjectMatrix(resolution);
   cudaMat4 ModelViewCudaMatrix=utilityCore::glmMat4ToCudaMat4(cameraMatrix*modelMatrix);
   cudaMat4 ProjectionCudaMatrix=utilityCore::glmMat4ToCudaMat4(projectionMatrix);
  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize,ModelViewCudaMatrix,device_vbo_eyeCoord,ProjectionCudaMatrix);
  cudaDeviceSynchronize();

  //------------------------------
  //primitive assembly
  //------------------------------
  bool *backfaceFlag=NULL;
  #ifdef BackFaceCulling
  cudaMalloc((void**)&backfaceFlag, (ibosize/3)*sizeof(bool));
  #endif

  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives,device_vbo_eyeCoord,backfaceFlag);
  cudaDeviceSynchronize();

  bool* partialfill=NULL;
#ifdef AntiAliasing
    cudaMalloc((void**)&partialfill, (int)resolution.x*(int)resolution.y*sizeof(bool));
	bool value=false;
	initialdepthflags<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, partialfill,value);
#endif
  
#ifdef GEOMETRYSHADER

  bool *newbackfaceFlag=NULL;
  #ifdef BackFaceCulling
  cudaMalloc((void**)&newbackfaceFlag, (ibosize/3*4)*sizeof(bool));
  #endif
   triangle* newprimitives = NULL;
   cudaMalloc((void**)&newprimitives, (ibosize/3*4)*sizeof(triangle));
   addMoreGeometry<<<primitiveBlocks,tileSize>>>(primitives,ibosize/3,newprimitives,ibosize/3*4,newbackfaceFlag);

   int newprimitiveBlocks=ceil(((float)ibosize/3*4)/((float)tileSize));
   rasterizationKernel<<<newprimitiveBlocks, tileSize>>>(newprimitives, ibosize/3*4, depthbuffer, resolution,newbackfaceFlag,partialfill);

  cudaDeviceSynchronize();
#else
 
  //------------------------------
  //rasterization
  //-----------------------------

  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution,backfaceFlag,partialfill);
  cudaDeviceSynchronize();
#endif
 
  glm::vec4 lightPos4=cameraMatrix*glm::vec4(-4.0f,4.0f,4.0f,1.0f);
  glm::vec3 lightPos=glm::vec3(lightPos4.x,lightPos4.y,lightPos4.z);
  glm::vec3 lColor=glm::vec3(1.0f,1.0f,1.0f);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,lightPos,lColor);
  cudaDeviceSynchronize();

  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer,frame);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

#ifdef BackFaceCulling
  cudaFree(backfaceFlag);
 #endif

#ifdef GEOMETRYSHADER
	#ifdef BackFaceCulling
	 cudaFree(newbackfaceFlag);
	#endif
	cudaFree(newprimitives);
#endif

#ifdef AntiAliasing
	cudaFree(partialfill);
#endif

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_vbo_eyeCoord );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

