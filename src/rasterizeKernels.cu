// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
#include <thrust/random.h>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "glm\gtc\/matrix_transform.hpp"

glm::vec3* framebuffer;
fragment* depthbuffer;
int* device_stencil;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives;

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
	  y=resolution.y-y;
	  x=resolution.x-x;
	  index=x+(y*resolution.x);


      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize , cudaMat4 project){//, float *nbo){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){

	  //printf("vbo before tranform %f   %f    %f ",vbo[3*index+0],vbo[3*index+1],vbo[3*index+2]);

	 // for (int i=0;i<vbosize;i++)
	  //{
	//	  printf("vbo before tranform %f  ",vbo[i]);
	 //}

	  
	  glm::vec4 newvbo= glm::vec4(vbo[3*index],vbo[3*index+1],vbo[3*index+2],1); //for point its 1
	 // glm::vec4 newnbo= glm::vec4(nbo[3*index],vbo[3*index+1],vbo[3*index+2],1);

	  glm::vec3 projectedvbo=multiplyMV(project,newvbo);
	 // glm::vec3 projectednbo= multiplyMV(project,newnbo);

	  vbo[3*index]=projectedvbo.x;
	  vbo[3*index+1]=projectedvbo.y;
	  vbo[3*index+2]=projectedvbo.z;

	  

	  //printf("  vbo after tranform %f   %f    %f ",vbo[3*index],vbo[3*index+1],vbo[3*index+2]);

  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float* nbo, int nbosize){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){

	  primitives[index].p0.x=vbo[9*index+0];	primitives[index].p0.y=vbo[9*index+1];	primitives[index].p0.z=vbo[9*index+2];
	  primitives[index].p1.x=vbo[9*index+3];	primitives[index].p1.y=vbo[9*index+4];	primitives[index].p1.z=vbo[9*index+5];
	  primitives[index].p2.x=vbo[9*index+6];	primitives[index].p2.y=vbo[9*index+7];	primitives[index].p2.z=vbo[9*index+8];
		

	  primitives[index].c0.x=cbo[0];	primitives[index].c0.y=cbo[1];	primitives[index].c0.z=cbo[2];
	  primitives[index].c1.x=cbo[3];	primitives[index].c1.y=cbo[4];	primitives[index].c1.z=cbo[5];
	  primitives[index].c2.x=cbo[6];	primitives[index].c2.y=cbo[7];	primitives[index].c2.z=cbo[8];

	  
	  primitives[index].n0.x=nbo[9*index+0];	primitives[index].n0.y=nbo[9*index+1];	primitives[index].n0.z=nbo[9*index+2];
  	  primitives[index].n1.x=nbo[9*index+3];	primitives[index].n1.y=nbo[9*index+4];	primitives[index].n1.z=nbo[9*index+5];
	  primitives[index].n2.x=nbo[9*index+6];	primitives[index].n2.y=nbo[9*index+7];	primitives[index].n2.z=nbo[9*index+8];
  
	 //printf("nbo = %f  \n", nbo[0]);
  }
}
//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  if (index == 2181)
	  {
		  printf("Before %i -- P0: (%f,%f), P1(%f, %f), P2: (%f,%f)\n", index, 
			  primitives[index].p0.x, primitives[index].p0.y,
			  primitives[index].p1.x, primitives[index].p1.y,
			  primitives[index].p2.x, primitives[index].p2.y);
	  }

	  primitives[index].p0.x=(primitives[index].p0.x+1) *resolution.x/2;
	  primitives[index].p0.y=(primitives[index].p0.y+1) *resolution.y/2;

	  primitives[index].p1.x=(primitives[index].p1.x+1) *resolution.x/2;
	  primitives[index].p1.y=(primitives[index].p1.y+1) *resolution.y/2;

	  primitives[index].p2.x=(primitives[index].p2.x+1) *resolution.x/2;
	  primitives[index].p2.y=(primitives[index].p2.y+1) *resolution.y/2;
	  
	  if (index == 2181)
	  {
		  printf("After %i -- P0: (%f,%f), P1(%f, %f), P2: (%f,%f)\n", index, 
			  primitives[index].p0.x, primitives[index].p0.y,
			  primitives[index].p1.x, primitives[index].p1.y,
			  primitives[index].p2.x, primitives[index].p2.y);
	  }

	  int bottom=min(min(primitives[index].p0.y, primitives[index].p1.y),primitives[index].p2.y)-1;
      int top = max(max(primitives[index].p0.y, primitives[index].p1.y),primitives[index].p2.y) + 1;

	 // float top= std::max(primitives[index].p0.y, primitives[index].p1.y, primitives[index].p2.y);
	 // float bottom= std::min(primitives[index].p0.y, primitives[index].p1.y, primitives[index].p2.y);

	  if (index == 2181)
	  {
		  printf(" Bottom Top: (%i,%i)\n", bottom, top);
	  }
	  
	  float slopep0p1,slopep2p0, slopep1p2;
	  //slopep0p1=1;slopep2p0=1; slopep1p2=1;
	  if (top >=resolution.y)
	  {
			top=resolution.y-1;
	  }
	  else if (bottom <= 0 )
	  {
		  bottom=0;
	  }

	  int currentpoint=0;
	  currentpoint=top;
	  int xmin, xmax;
	  float xvaluetest1,xvaluetest2,xvaluetest3;

	  bool tri=false;
	  
	  while(currentpoint!=bottom)
	  {		
		xmax=-1000000;
	    xmin=10000000;
		//////// CASE2

		if( (primitives[index].p1.x - primitives[index].p0.x)!=0)
		{
			if (primitives[index].p1.y+0.0001 <= primitives[index].p0.y && primitives[index].p1.y-0.0001 >= primitives[index].p0.y)
			{
				/*if (index == 1797) printf("Here\n");*/
				xmin=min(primitives[index].p1.x,primitives[index].p0.x);
				xmax=max(primitives[index].p1.x,primitives[index].p0.x);
			}
			else
			{
				if (index == 2181) printf("in else\n");
				slopep0p1= (primitives[index].p1.y - primitives[index].p0.y) / (primitives[index].p1.x - primitives[index].p0.x);
				
				xvaluetest1=(currentpoint-primitives[index].p1.y)/slopep0p1 + primitives[index].p1.x;
				/*if (xvaluetest1>=0 && xvaluetest1<resolution.x)
				{*/

				if (index == 2181) printf("slopep0p1=%f xvaluetest1= %f   current point= %i \n",slopep0p1,xvaluetest1,currentpoint);

				if ((xvaluetest1 <= primitives[index].p0.x && xvaluetest1 >= primitives[index].p1.x)
				||(xvaluetest1 >= primitives[index].p0.x && xvaluetest1 <= primitives[index].p1.x))
				{

					/*if (index == 1797) printf( "checking xvaluetest1 \n");*/
				  if(xvaluetest1<xmin)
				  {
					  xmin=xvaluetest1;
				  }
				  else if (xvaluetest1>xmax)
				  {
					  xmax=xvaluetest1;
				  }
				}

				if (index == 2181)
					printf("Xmin,Xmax After P0P1: (%i, %i)\n", xmin, xmax);
			//}
			}
		}
		else  
		{
				xmin=min((int)primitives[index].p1.x,xmin); 
				xmax=max((int)primitives[index].p1.x,xmax);
		}

		/////////// CASE2

		if ((primitives[index].p2.x - primitives[index].p1.x)!=0)
		{
			if (primitives[index].p2.y == primitives[index].p1.y)
			{
				xmin=min((int)min(primitives[index].p2.x, primitives[index].p1.x),xmin);
				xmax=max((int)max(primitives[index].p2.x, primitives[index].p1.x),xmax);
			}
			else
			{
				if (index == 2181)
					printf("Xmin,Xmax Before P1P2: (%i, %i)\n", xmin, xmax);
				slopep1p2= (primitives[index].p2.y - primitives[index].p1.y) / (primitives[index].p2.x - primitives[index].p1.x);
				xvaluetest2=(currentpoint-primitives[index].p2.y)/slopep1p2 + primitives[index].p2.x;

				if (index == 2181) printf("slopep2p1=%f xvaluetest2= %f ,currentpoint= %i   \n",slopep1p2,xvaluetest2, currentpoint);

				if ((xvaluetest2 <= primitives[index].p1.x && xvaluetest2 >= primitives[index].p2.x) || 
					(xvaluetest2 >= primitives[index].p1.x && xvaluetest2 <= primitives[index].p2.x))
				{
					if(xvaluetest2>=0 && xvaluetest2<=xmin)
					xmin=xvaluetest2;
					if(xvaluetest2<resolution.x && xvaluetest2>xmax)
					{
						xmax=xvaluetest2;
					}
				}
				
				if (index == 2181)
					printf("Xmin,Xmax After P1P2: (%i, %i)\n", xmin, xmax);
			}
		}
		else  
		{
				xmin=min((int)primitives[index].p1.x,xmin);
				xmax=max((int)primitives[index].p1.x,xmax);	 
		}


		////////////////////// CASE3

		if (primitives[index].p0.x - primitives[index].p2.x!=0)
		{
			if (primitives[index].p0.y == primitives[index].p2.y)
			{
				xmin=min((int)min(primitives[index].p0.x, primitives[index].p2.x),xmin);
				xmax=max((int)max(primitives[index].p0.x, primitives[index].p2.x),xmax);
			}
			else
			{
				if (index == 2181)
					printf("Xmin,Xmax Before P0P2: (%i, %i) - CurrentPoint- %i\n", xmin, xmax, currentpoint);

				slopep2p0= (primitives[index].p0.y - primitives[index].p2.y) / (primitives[index].p0.x - primitives[index].p2.x);
				xvaluetest3=(currentpoint-primitives[index].p0.y)/slopep2p0 + primitives[index].p0.x ;
				
				if (index == 2181) printf("slopep2p0=%f xvaluetest3= %d   \n",slopep2p0,xvaluetest3);

				if (xvaluetest3>=0 && xvaluetest3<resolution.x)
				{
					
					if ((xvaluetest3 <= primitives[index].p2.x && xvaluetest3 >= primitives[index].p0.x) || 
					(xvaluetest3 >= primitives[index].p2.x && xvaluetest3 <= primitives[index].p0.x))
					{
						if(xvaluetest3>=0 && xvaluetest3<xmin)
						xmin=xvaluetest3;
						if(xvaluetest3<resolution.x && xvaluetest3>xmax)
						xmax=xvaluetest3;
					}
				
				}
				if (index == 2181)
					printf("Xmin,Xmax After P1P2: (%i, %i)\n", xmin, xmax);
			}
					
		}
		else  
		{
				xmin=min((int)primitives[index].p2.x,xmin);
				xmax=max((int)primitives[index].p2.x,xmax);  
		}
	  
		glm::vec3 barry = calculateBarycentricCoordinate(primitives[index], glm::vec2(xmin, currentpoint));
		  // using (y-y1)/m + x1=xB
		  // here y is currentpoint
		
		while(xmin<=xmax)
		  {
				  int pixel_index= xmin+currentpoint*resolution.x;

				  fragment newfrag;
				  newfrag.color=barry.x*primitives[index].c0 +  barry.y*primitives[index].c1  + barry.z*primitives[index].c2;
				  newfrag.normal= glm::normalize(barry.x*primitives[index].n0 + barry.y*primitives[index].n1 + barry.z*primitives[index].n2);
				  newfrag.lock=1;
				  newfrag.position.x= xmin;
				  newfrag.position.y= currentpoint;

				  //atomic comapre and swap
				  bool loop=true;
				  while(loop)
				  {
						  if( xmin < resolution.x && xmin>=0 && currentpoint<resolution.y && currentpoint>0  )
						  {
							  /*if(xmin==0)
							  {
								  printf("index  %d \n", index);
							  }
							  tri=true;*/
							  if( depthbuffer[index].position.z < newfrag.position.z)
							  {
									//if (atomicExch(&(depthbuffer[pixel_index].lock), 1) == 0)
									{
										depthbuffer[pixel_index]= newfrag;
										loop=false;
									//	atomicExch(&(depthbuffer[pixel_index].lock),0); 
									}
							  }
							// printf("some %f",depthbuffer[pixel_index].normal.y);
							  else
							  {
								 loop=false;
							  }

						  }
				  }
				  xmin++;
		  }
		  currentpoint--;
	  }

	  /*if (tri==false)
	  {
		  printf("index %d \n", index);
	  }*/
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightpos, glm::vec3 lightcol, int* device_stencil)
  {
	  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;
		int index = x + (y * resolution.x);

		//setting up the stencil
	
	  if(x<=resolution.x && y<=resolution.y)
	  {
		  if ( device_stencil[index]==1)
		  {

		  glm::vec3 normal= glm::normalize(depthbuffer[index].normal);
		  glm::vec3 L=lightpos-depthbuffer[index].position;
		  float diffuse=glm::clamp((glm::dot(normal,glm::normalize(L)),0.0),0.0,1.0);
		  
		  glm::vec3 final_col= diffuse*lightcol * depthbuffer[index].color;

		  //depthbuffer[index].color = final_col;
		  }
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, int nbosize, float* nbo, glm::vec3 lightpos, glm::vec3 lightcol){

	//set uf the stencil buffer
  device_stencil =NULL;
  cudaMalloc((void**)&device_stencil, (int) resolution.x*(int)resolution.y*sizeof(int));

  int totalpixels= resolution.x*resolution.y;

  int* stencil=new int[totalpixels];

	
	cudaMalloc((void**)&device_stencil, (int) resolution.x*(int)resolution.y*sizeof(int));
	cudaMemcpy( device_stencil, stencil, totalpixels*sizeof(bool), cudaMemcpyHostToDevice);


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
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_nbo =NULL;
  cudaMalloc ((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy(device_nbo, nbo, nbosize*sizeof(float),cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------

  
  
  //setting up camera first

  glm::vec3 CameraPosition = glm::vec3(0.0f, 0.5f, 7.0f);
  int width = resolution.x;
  int height = resolution.y;
		
  glm::mat4 projection = glm::perspective(60.0f, static_cast<float>(width) / static_cast<float>(height), 0.1f, 50.0f);
	 	
  glm::mat4 camera = glm::lookAt(CameraPosition, glm::vec3(0.0, 0.5, 0), glm::vec3(0, 1, 0));
	 	
  projection = projection * camera;
	 	
  cudaMat4 cudaProjection = utilityCore::glmMat4ToCudaMat4(projection);
	 	
  glm::mat4 invProjection = glm::inverse(projection);
	 	
  cudaMat4 cudaInvProjection = utilityCore::glmMat4ToCudaMat4(invProjection);
	 	
  
  /*
  glm::mat4 projection=glm::perspective(60.0f, static_cast<float>(resolution.x)/ static_cast<float>(resolution.y),0.1f, 30.0f); 
  glm::vec3 cameraposition= glm::vec3(0,2,10);
  glm::mat4 camera= glm::lookAt(cameraposition,glm::vec3(0,0,0),glm::vec3(0,5,0));
  
  //projection=projection*camera;

  cudaMat4 project= utilityCore::glmMat4ToCudaMat4(projection);
  */
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, cudaProjection);
  

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives,device_nbo,nbosize);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lightpos, lightcol, device_stencil);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( device_nbo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree(device_stencil);
}

