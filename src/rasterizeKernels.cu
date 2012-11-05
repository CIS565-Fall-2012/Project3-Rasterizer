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
vertex* device_vboFull;
float* device_vbo;
float* device_cbo;
int* device_ibo;
int* lock;
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
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag, int* lock){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
	  lock[index] = 0;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, vertex* vboFull, glm::mat4 mM, glm::mat4 vM, glm::mat4 pM, glm::mat4 sM){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3)
  {
	  int idxX = index*3;
	  int idxY = idxX + 1;
	  int idxZ = idxX + 2;

	  glm::vec4 v( vbo[idxX], vbo[idxY], vbo[idxZ], 1.0f );
	  //v = v*mM*vM*pM;
	  v = mM*v;

	  vboFull[index].lightdir = ( glm::vec4( 10, 10, 10, 1 ) - v ).swizzle(glm::X, glm::Y, glm::Z);

	  v = sM*pM*vM*v;
	  
	  vboFull[index].position = v.swizzle(glm::X, glm::Y, glm::Z);
	  vboFull[index].position /= v.w;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(vertex* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, int frame){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int idx0 = index*3;
	  int idx1 = idx0 + 1;
	  int idx2 = idx0 + 2;

	  int idxX = ibo[idx0]*3;
	  int idxY = idxX + 1;
	  int idxZ = idxX + 2;
	  
	  primitives[ index ].v0.position = vbo[ibo[idx0]].position;
	  primitives[ index ].v0.color = glm::vec3( cbo[(idxX+frame)%cbosize], cbo[(idxY+frame)%cbosize], cbo[(idxZ+frame)%cbosize] );

	  idxX = ibo[idx1]*3;
	  idxY = idxX + 1;
	  idxZ = idxX + 2;

	  primitives[ index ].v1.position = vbo[ibo[idx1]].position;
	  primitives[ index ].v1.color = glm::vec3( cbo[(idxX+frame)%cbosize], cbo[(idxY+frame)%cbosize], cbo[(idxZ+frame)%cbosize] );

	  idxX = ibo[idx2]*3;
	  idxY = idxX + 1;
	  idxZ = idxX + 2;

	  primitives[ index ].v2.position = vbo[ibo[idx2]].position;
	  primitives[ index ].v2.color = glm::vec3( cbo[(idxX+frame)%cbosize], cbo[(idxY+frame)%cbosize], cbo[(idxZ+frame)%cbosize] );
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, int * lock, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  glm::vec3 p0,p1,p2;
	  

	  p0 = primitives[index].v0.position;
	  p1 = primitives[index].v1.position;
	  p2 = primitives[index].v2.position;

	  glm::vec3 rA = p1-p0;
	  glm::vec3 rB = p2-p0;

	  glm::vec3 norm = glm::normalize( glm::cross( glm::normalize( rA ), glm::normalize( rB ) ) );

	  if( norm.z <= 0 ) return; //backface culling -should try to move this to the primitive stage if time allows

	  glm::vec3 minP;
	  glm::vec3 maxP;

	  triangle tri = primitives[index];

	  getAABBForTriangle( tri, minP, maxP );

	  int dIndex;
	  int dY;

	  for( int y = minP.y; y <= maxP.y; y++ )
	  {
		  dY = y*resolution.x;
		  for( int x = minP.x; x <= maxP.x; x++ )
		  {
			  glm::vec3 barycoord = calculateBarycentricCoordinate( tri, glm::vec2( x, y ) );
			  dIndex = dY + x;
			  if( isBarycentricCoordInBounds( barycoord ) )
			  {
				  float depth = getZAtCoordinate( barycoord, tri );
				  bool inLoop = true;
				  while( inLoop )
				  {
					  if( atomicExch( &(lock[dIndex]), 1 ) == 0 )
					  {
						  if( depth > depthbuffer[dIndex].position.z )
						  {
							  depthbuffer[dIndex].position.x = x;
							  depthbuffer[dIndex].position.y = y;
							  depthbuffer[dIndex].position.z = depth;
							  depthbuffer[dIndex].color = tri.v0.color*barycoord.x + 
														  tri.v1.color*barycoord.y + 
														  tri.v2.color*barycoord.z;
							  depthbuffer[dIndex].normal = norm;
							  depthbuffer[dIndex].lightdir = tri.v0.lightdir*barycoord.x + 
															 tri.v1.lightdir*barycoord.y + 
															 tri.v2.lightdir*barycoord.z;
						  }
						  inLoop = false;
						  atomicExch( &(lock[dIndex]), 0 );
					  }
				  }
			  }
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution ){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  depthbuffer[index].color *= ( float ) max( 0.0f, -glm::dot( glm::normalize( depthbuffer[index].normal ), glm::normalize( depthbuffer[index].lightdir ) ) ) + glm::vec3( 0.1f );
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


	glm::mat4 modelMatrix = glm::rotate( glm::mat4(1.0f), frame, glm::vec3( 0, 1, 0 ) );
	glm::mat4 projectionMatrix = glm::perspective( 45.0f, resolution.x/resolution.y, 0.1f, 100.0f );
	glm::mat4 screenMatrix = glm::mat4(1.0f);
	screenMatrix = glm::scale( screenMatrix, glm::vec3( resolution, 1.0f ) );
	screenMatrix = glm::translate( screenMatrix, glm::vec3( 0.5, 0.5, 0 ) );
	screenMatrix = glm::scale( screenMatrix, -glm::vec3( 0.5, 0.5, -1 ) );
	glm::mat4 viewMatrix = glm::lookAt( glm::vec3( 0, 2, 2 ), glm::vec3( 0, 0, 0 ), glm::vec3( 0, 1, 0 ) );

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
  lock = NULL;
  cudaMalloc((void**)&lock, (int)resolution.x*(int)resolution.y*sizeof(int));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag,lock);

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

  device_vboFull = NULL;
  cudaMalloc((void**)&device_vboFull, vbosize/3*sizeof(float));

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_vboFull, modelMatrix, viewMatrix, projectionMatrix, screenMatrix);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vboFull, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, 0*(int)frame);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, lock, resolution);

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
  cudaFree( device_vboFull );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

