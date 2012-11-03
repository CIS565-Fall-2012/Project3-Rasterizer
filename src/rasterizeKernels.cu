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
float* device_cbo;
int* device_ibo;
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
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 mM, glm::mat4 vM, glm::mat4 pM){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3)
  {
	  int idxX = index*3;
	  int idxY = idxX + 1;
	  int idxZ = idxX + 2;

	  glm::vec4 v( vbo[idxX], vbo[idxY], vbo[idxZ], 1.0f );
	  //v = v*mM*vM*pM;
	  v = pM*vM*mM*v;
	  
	  vbo[idxX] = v.x/v.w;
	  vbo[idxY] = v.y/v.w;
	  vbo[idxZ] = v.z/v.w;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int idx0 = index*3;
	  int idx1 = idx0 + 1;
	  int idx2 = idx0 + 2;

	  int idxX = ibo[idx0]*3;
	  int idxY = idxX + 1;
	  int idxZ = idxY + 2;
	  
	  primitives[ index ].p0 = glm::vec3( vbo[idxX], vbo[idxY], vbo[idxZ] );
	  primitives[ index ].c0 = glm::vec3( cbo[idxX%cbosize], cbo[idxY%cbosize], cbo[idxZ%cbosize] );

	  idxX = ibo[idx1]*3;
	  idxY = idxX + 1;
	  idxZ = idxY + 2;

	  primitives[ index ].p1 = glm::vec3( vbo[idxX], vbo[idxY], vbo[idxZ] );
	  primitives[ index ].c1 = glm::vec3( cbo[idxX%cbosize], cbo[idxY%cbosize], cbo[idxZ%cbosize] );

	  idxX = ibo[idx2]*3;
	  idxY = idxX + 1;
	  idxZ = idxY + 2;

	  primitives[ index ].p2 = glm::vec3( vbo[idxX], vbo[idxY], vbo[idxZ] );
	  primitives[ index ].c2 = glm::vec3( cbo[idxX%cbosize], cbo[idxY%cbosize], cbo[idxZ%cbosize] );
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  glm::vec2 p0,p1,p2;
	  

	  p0 = primitives[index].p0.swizzle(glm::X,glm::Y);
	  p1 = primitives[index].p1.swizzle(glm::X,glm::Y);
	  p2 = primitives[index].p2.swizzle(glm::X,glm::Y);
	  
	  
//	  depthbuffer[(int)(floor(minP.x)+resolution.x*floor(minP.y))].color = primitives[index].c0;
//	  depthbuffer[(int)(floor(maxP.x)+resolution.x*floor(maxP.y))].color = primitives[index].c2;

	  glm::vec2 rA = p1-p0;
	  glm::vec2 rB = p2-p0;

	  if( glm::cross( glm::vec3( rA, 0.0f ), glm::vec3( rB, 0.0f ) ).z <= 0 )
		  return;


	  glm::vec2 minP = glm::max( glm::vec2( 0 ), glm::min( p0, glm::min( p1, p2 ) ) );
	  glm::vec2 maxP = glm::min( resolution - glm::vec2( 1 ), glm::max( p0, glm::max( p1, p2 ) ) );

	  glm::vec3 tmp = glm::cross( glm::vec3( 0, 0, 1 ), glm::vec3( rA, 0.0f ) );
	  glm::vec2 vA(tmp.x, tmp.y);

	  tmp = glm::cross( glm::vec3( 0, 0, 1 ), glm::vec3( rB, 0.0f ) );
	  glm::vec2 vB(tmp.x, tmp.y);



	  float lengthA = glm::length( rA );
	  float lengthB = glm::length( rB );

	  for( int y = minP.y; y <= maxP.y; y++ )
	  {
		  for( int x = minP.x; x <= maxP.x; x++ )
		  {
			  glm::vec2 p = glm::vec2( x, y ) - p0;
			  float lambdaA = glm::dot( p, vA ) / glm::dot( rB, vA );
			  float lambdaB = glm::dot( p, vB ) / glm::dot( rA, vB );
			  if( lambdaA >= 0 && lambdaB >= 0 && lambdaA + lambdaB <= 1 )
			  {
				  float depth = primitives[index].p0.z*lambdaA + 
								primitives[index].p1.z*lambdaB + 
								primitives[index].p2.z*(1-lambdaA-lambdaB);
				  if( depth > depthbuffer[(int)(x+resolution.x*y)].position.z )
				  {
					  depthbuffer[(int)(x+resolution.x*y)].position.z = depth;
					  depthbuffer[(int)(x+resolution.x*y)].color = primitives[index].c0*lambdaA + 
																   primitives[index].c1*lambdaB + 
																   primitives[index].c2*(1-lambdaA-lambdaB);
				  }
			  }
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize){


	glm::mat4 modelMatrix = glm::rotate( glm::mat4(1.0f), frame/3, glm::vec3( 0, 1, 0 ) );
	glm::mat4 projectionMatrix = glm::scale(
		glm::translate(
		glm::perspective( 45.0f, resolution.x/resolution.y, 0.1f, 100.0f ),
		glm::vec3(resolution, 0.0f)),glm::vec3(-resolution,1.0f));
	glm::mat4 viewMatrix = glm::lookAt( glm::vec3( 0, 0, 1 + frame/100 ), glm::vec3( 0, 0, 0 ), glm::vec3( 0, 1, 0 ) );

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
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, modelMatrix, viewMatrix, projectionMatrix);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  checkCUDAError("Rasterizer");
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);
  checkCUDAError("Frag Shader");

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
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

