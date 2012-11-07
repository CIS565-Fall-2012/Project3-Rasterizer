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
float* device_nbo;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, glm::mat4 MVP){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  
	  glm::vec4 curVertex = glm::vec4(vbo[index*3], vbo[index*3 + 1], vbo[index*3 + 2], 1.0f);
	  curVertex = MVP * curVertex;

	  vbo[index*3] = curVertex.x;
	  vbo[index*3 + 1] = curVertex.y;
	  vbo[index*3 + 2] = curVertex.z;

	  glm::vec4 curNormal = glm::vec4(nbo[index*3], nbo[index*3 + 1], nbo[index*3 + 2], 1.0f);
	  curNormal = MVP * curNormal;

	  nbo[index*3] = curNormal.x;
	  nbo[index*3 + 1] = curNormal.y;
	  nbo[index*3 + 2] = curNormal.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){

	  int i = 9*index;
	  primitives[index].p0 = glm::vec3(vbo[i], vbo[i + 1], vbo[i + 2]);
	  primitives[index].p1 = glm::vec3(vbo[i + 3], vbo[i + 4], vbo[i + 5]);
	  primitives[index].p2 = glm::vec3(vbo[i + 6], vbo[i + 7], vbo[i + 8]);

	  primitives[index].n0 = glm::vec3(nbo[i], nbo[i + 1], nbo[i + 2]);
	  primitives[index].n1 = glm::vec3(nbo[i + 3], nbo[i + 4], nbo[i + 5]);
	  primitives[index].n2 = glm::vec3(nbo[i + 6], nbo[i + 7], nbo[i + 8]);

	  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);

	  primitives[index].culled = false;
  }
}


__global__ void backfaceCullingKernel(triangle* primitives, int primitivesCount, glm::vec3 camPos, glm::mat4 model, glm::mat4 view, glm::mat4 projection, glm::vec4 viewport){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		
		glm::vec3 worldSpaceP0 = glm::unProject(primitives[index].p0, view * model,projection, viewport);
		glm::vec3 worldSpaceP1 = glm::unProject(primitives[index].p1, view * model,projection, viewport);
		glm::vec3 worldSpaceP2 = glm::unProject(primitives[index].p2, view * model,projection, viewport);
		glm::vec3 primitiveNormal = glm::cross(worldSpaceP2 - worldSpaceP0, worldSpaceP1 - worldSpaceP0);
		
		if(glm::dot(camPos - worldSpaceP0, primitiveNormal) > 0.0f)
		{
			primitives[index].culled = true;
			//printf("->");
		}
	}

}


//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount)// && primitives[index].culled == false)
  {
	  //translate vertices by +1, then scale by res/2
	  triangle curTriangle = primitives[index];
	  curTriangle.p0 = glm::vec3(curTriangle.p0.x + 1, curTriangle.p0.y + 1, curTriangle.p0.z);
	  curTriangle.p1 = glm::vec3(curTriangle.p1.x + 1, curTriangle.p1.y + 1, curTriangle.p1.z);
	  curTriangle.p2 = glm::vec3(curTriangle.p2.x + 1, curTriangle.p2.y + 1, curTriangle.p2.z);
	  
	  glm::mat3 scalingMatrix = glm::mat3(resolution.x / 2, 0, 0,
										  0, resolution.y / 2, 0,
										  0, 0, 1);

	  curTriangle.p0 = scalingMatrix * curTriangle.p0;
	  curTriangle.p1 = scalingMatrix * curTriangle.p1;
	  curTriangle.p2 = scalingMatrix * curTriangle.p2;

	  //scan each pixel and use barycentric co ordinates to determine interpolated color value
	  glm::vec3 min, max, barycentric;
	  bool isInside;
	  getAABBForTriangle(curTriangle, min, max);
	  
	  for(int i = floor(min.x); i < ceil(max.x); i++)
	  {
		  for(int j = floor(min.y); j < ceil(max.y); j++)
		  {
			  barycentric = calculateBarycentricCoordinate(curTriangle, glm::vec2(i,j));
			  isInside = isBarycentricCoordInBounds(barycentric);
			  if(isInside)
			  {
				  int ind = resolution.x * resolution.y - resolution.x + i - j*resolution.x;
				  if(i < resolution.x && j < resolution.y)
				  {
					  glm::vec3 tempPosition = glm::vec3(i,j,getZAtCoordinate(barycentric, curTriangle));
					  if(tempPosition.z > depthbuffer[(int)(resolution.x * resolution.y) - ind].position.z)
					  {
						  depthbuffer[ind].position = tempPosition;
					  //depthbuffer[ind].color = glm::vec3(1,1,1);
						  depthbuffer[ind].color = getColorAtCoordinate(barycentric, curTriangle);
						  depthbuffer[ind].normal = getNormalAtCoordinate(barycentric, curTriangle);
					  }
				  }
			  }
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightPosition, glm::vec3 lightColor){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	  glm::vec3 lightDirection = glm::normalize(lightPosition - depthbuffer[index].position);
	  depthbuffer[index].normal = glm::normalize(depthbuffer[index].normal);
	  //float factor = max(glm::dot(depthbuffer[index].normal, lightDirection), 0.0f);
	  float factor = abs(glm::dot(depthbuffer[index].normal, lightDirection));
	  depthbuffer[index].color =  6 * factor * lightColor * depthbuffer[index].color;
	  depthbuffer[index].scissored = false;
  }
}


__global__ void scissorTest(glm::vec2 resolution, fragment* depthbuffer, glm::vec2 rectPos, glm::vec2 dimensions){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){
		if(x > rectPos.x && x < rectPos.x + dimensions.x && y > rectPos.y && y < rectPos.y + dimensions.y)
			depthbuffer[index].scissored = false;
		else
			depthbuffer[index].scissored = true;
	}
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y && depthbuffer[index].scissored == false){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){

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

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  glm::vec3 camPos = glm::vec3(0,0,5);
  glm::mat4 model = glm::mat4(1.0f);
  glm::mat4 view = glm::lookAt(camPos, glm::vec3(0,0,0), glm::vec3(0,1,0));
  glm::mat4 projection = glm::perspective(75.0f, resolution.x/resolution.y, 0.1f, 100.0f);
  glm::vec4 viewport = glm::vec4(0, 0, resolution.x, resolution.y);
  glm::mat4 MVP = projection * view * model;

  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, MVP);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //backface culling
  //------------------------------
  backfaceCullingKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, camPos, model, view, projection, viewport);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  glm::vec3 lpos = glm::vec3(5,5,10);
  glm::vec3 lcol = glm::vec3(1,1,1);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lpos, lcol);

  cudaDeviceSynchronize();
  //------------------------------
  //scissor test
  //------------------------------
  glm::vec2 rectPos = glm::vec2(0,0);
  glm::vec2 dimensions = glm::vec2(1000,1000);
  scissorTest<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, rectPos, dimensions);

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

