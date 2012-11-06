// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include <cutil.h>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"


glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;


using namespace std;

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
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize,glm::mat4 mvp){


   int index = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(index<vbosize/3){	
		glm::vec4 in_Position = (mvp  *  glm::vec4(vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2], 1));
		in_Position /= in_Position.w;
		vbo[index * 3]	=	in_Position.x;
		vbo[index * 3 + 1]	=	in_Position.y;
		vbo[index * 3 + 2]	=	in_Position.z;	
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float * nbo, int nbosize, triangle* primitives, glm::vec3 cameraPos, glm::mat4 model, glm::mat4 projection,glm::mat4 view, glm::vec4 viewport, bool enableBackFaceCull){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){

	    int iIndex = 3 * index;
		//vertices
		primitives[index].p0 = glm::vec3(vbo[3 * iIndex], vbo[3 * iIndex + 1], vbo[3 * iIndex + 2]);
		primitives[index].p1 = glm::vec3(vbo[3 * (iIndex + 1)], vbo[3 * (iIndex + 1) + 1], vbo[3 * (iIndex + 1) + 2]);
		primitives[index].p2 = glm::vec3(vbo[3 * (iIndex + 2)], vbo[3 * (iIndex + 2) + 1], vbo[3 * (iIndex + 2) + 2]);

		//color
		primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
		primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
		primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);

		if(enableBackFaceCull){
			//back face culling
			glm::vec3 pos = glm::unProject( (primitives[index].p0 +primitives[index].p1+ primitives[index].p2)/3.0f, view * model,projection, viewport);

			glm::vec3 campos = glm::unProject( cameraPos, view * model,projection, viewport);
			glm::vec3 viewVector = glm::vec3(pos.x,pos.y,pos.z) - campos;
			glm::vec3 normal = glm::vec3(nbo[3 * iIndex], nbo[3 * iIndex + 1], nbo[3 * iIndex + 2]);

			if(glm::dot( viewVector,glm::vec3(nbo[3 * iIndex], nbo[3 * iIndex + 1], nbo[3 * iIndex + 2])) < 0.6)
		   {
			// draw the polygon here, it's visible
				primitives[index].backFace = 0;
			}else
				primitives[index].backFace = 1;

		}else{
			primitives[index].backFace = 0;

		}

  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::mat3 sm, bool enableScissorTest, glm::vec4 scissorWindow){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){

	  if(primitives[index].backFace == 0){
	    triangle tempTri = primitives[index];
	    tempTri.p0 = glm::vec3(tempTri.p0.x + 1, tempTri.p0.y + 1, tempTri.p0.z);
		tempTri.p1 = glm::vec3(tempTri.p1.x + 1, tempTri.p1.y + 1, tempTri.p1.z);
		tempTri.p2 = glm::vec3(tempTri.p2.x + 1, tempTri.p2.y + 1, tempTri.p2.z);
 
		//transform into pixel coordinates
		tempTri.p0 = sm * tempTri.p0;
		tempTri.p1 = sm * tempTri.p1;
		tempTri.p2 = sm * tempTri.p2;

		// find the bounding box
		glm::vec3 minPoint = glm::vec3(0,0,0);
		glm::vec3 maxPoint = glm::vec3(0,0,0);
		getAABBForTriangle(tempTri, minPoint, maxPoint);

		if(enableScissorTest){
			minPoint.x = max(scissorWindow[0], minPoint.x);
			maxPoint.x = min(scissorWindow[2], maxPoint.x);
			minPoint.y = max(scissorWindow[1], minPoint.y);
			maxPoint.y = max(scissorWindow[3], maxPoint.y);
		}
		
		glm::vec3 ndc;
		glm::vec3 baryCoord;
		fragment frag;
		//trace each pixel along x-axis
		for (int x = minPoint.x ; x <= maxPoint.x ; x++) {	
		    //trace for each pixel along the y-axis given x-axis
			for (int y =  minPoint.y; y <=  maxPoint.y; y++) {	
					baryCoord = calculateBarycentricCoordinate(tempTri, glm::vec2(x,y));
					if (isBarycentricCoordInBounds(baryCoord)) { // inside triangle
						int dBufIndex = y * resolution.x + x;
						//normalized device coordinates
						ndc.x = (float)(2 * x/ resolution.x) - 1.0f ;
						ndc.y =  1.0f -  (float)(2 * y/ resolution.y) ;
						frag.color = baryCoord.x*primitives[index].c0 + baryCoord.y*primitives[index].c1 + baryCoord.z*primitives[index].c2;
						frag.normal = glm::normalize(glm::cross(primitives[index].p1 - primitives[index].p0, primitives[index].p2 - primitives[index].p0));
						frag.position = glm::vec3(ndc.x, ndc.y, - ( baryCoord.x*primitives[index].p0.z + baryCoord.y*primitives[index].p1.z + baryCoord.z*primitives[index].p2.z));
						bool syncLoop = true;
						while( syncLoop )
						{
							if( atomicExch( &(depthbuffer[dBufIndex].syncLock), 1 ) == 0 )
							{
								//depth
								if( frag.position.z > depthbuffer[ dBufIndex].position.z )
								{
									depthbuffer[dBufIndex] = frag;
									
								}
								syncLoop = false;
								depthbuffer[ dBufIndex].syncLock = 0;
							}
						}						
					}
			} // end x loop
		}//end y loop	
	  }
	}//end (index<primitivecount)
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 cameraPos, glm::vec3 lightPos,  glm::vec3 diffuseLightColor,  glm::vec3 specularColor, float specularExponent, glm::vec3 ambientColor, bool enableLIGHT2 ){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	    glm::vec3 lightDir = glm::normalize(lightPos - depthbuffer[index].position);
		float diffuse = glm::max(glm::dot(depthbuffer[index].normal, lightDir), 0.0f);
		glm::vec3 colors = glm::vec3(0);
		glm::vec3 camDir = glm::normalize(cameraPos - depthbuffer[index].position);
		float cosAngle = glm::max(glm::dot(camDir, lightDir - 2.0f * glm::dot( lightDir, depthbuffer[index].normal ) * depthbuffer[index].normal ),0.0f);
			if (cosAngle > 0)
			{
				float spec = powf( cosAngle, specularExponent) * glm::dot(specularColor,depthbuffer[index].color) ;
				if(spec > 0){
					colors += depthbuffer[index].color*spec * diffuseLightColor;// * depthbuffer[index].color;
				}
			}
			colors+=(depthbuffer[index].color*diffuse) * diffuseLightColor + 0.15f * ambientColor * depthbuffer[index].color;
			if(enableLIGHT2){
					glm::vec3 lightPos2 = glm::vec3(2,2.5,1.0f);
					glm::vec3 lightDir2 = glm::normalize(lightPos2 - depthbuffer[index].position);
					float diffuse2 = glm::max(glm::dot(depthbuffer[index].normal, lightDir2), 0.0f);
					float cosAngle2 = glm::max(glm::dot(camDir, lightDir2 - 2.0f * glm::dot( lightDir2, depthbuffer[index].normal ) * depthbuffer[index].normal ),0.0f);
						if (cosAngle2 > 0)
						{
							float spec2 = powf( cosAngle2, 10.0f) * glm::dot(specularColor,depthbuffer[index].color) ;
							if(spec2 > 0){
								colors += depthbuffer[index].color*spec2 * diffuseLightColor;// * depthbuffer[index].color;
							}
						}
						colors+=(depthbuffer[index].color*diffuse) * diffuseLightColor;

			}
			depthbuffer[index].color = colors;
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize){

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
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  frag.syncLock = 0;
  frag.distance = -10000;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

    
  //------------------------------
  //Seting up the Projection Matrix
  //------------------------------
  glm::mat4 model      = glm::mat4(1.0f);
  //rotating position for the model
  model = glm::rotate(model,frame, glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 modelViewProj  = projection * view * model;

  //matrix used to transform the corrdinates into pixel coordinates
  glm::mat3 sm = glm::mat3(resolution.x / 2, 0, 0,	0, resolution.y / 2, 0,	0, 0, 1);
	
	
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

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  float * device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, modelViewProj);
   
  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  glm::vec4 viewport = glm::vec4(0.0, 0.0, resolution.x, resolution.y);
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize,device_nbo, nbosize, primitives, cameraPos, model, projection,view, viewport, enableBACKFACECULL);

  cudaDeviceSynchronize();

  //display the number of primitives currently displayed after back face cull
  if(enableSTENCILTEST){
	  int count = 0;
	  for(int i =0;i<ibosize/3;i++){
		  if(primitives[i].backFace == 0)
			  count++;

	  }
	  cout<<"count of primitives displayed after back face cull "<<count<<endl;
  }
  //------------------------------
  //rasterization
  //------------------------------

  
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, sm, enableSCISSORTEST, scissorTestWindow);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, cameraPos, lightPos, diffuseLightColor, specularColor, specularExponent, ambientColor, enableLIGHT2);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();
//  delete device_vbo1;
   cudaFree( device_nbo );

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

void enableBackFaceCull(){

	enableBACKFACECULL = !enableBACKFACECULL;
}
void enableScissorTest(){
	enableSCISSORTEST = !enableSCISSORTEST;
}

void enableStencilBuffer(){
	enableSTENCILTEST = !enableSTENCILTEST;
}
void resetTransformations(float x, float y){

	/*cameraPos.x = x;
	cameraPos.y = y;
    view = glm::lookAt(cameraPos,
				glm::vec3(0.0, 0.5, 0),//look at
				glm::vec3(0, -1, 0));//Head*/

	
}
void enableLight2(){
	enableLIGHT2 = !enableLIGHT2;
}