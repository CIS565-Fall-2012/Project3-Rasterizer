// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include <thrust/random.h>
#include "glm/glm.hpp"
#include "glm/gtx/vector_access.hpp"
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;
Light* light;
cudaMat4* MVP_matrix;
int* fragLocks;

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
__global__ void vertexShadeKernel(float* vbo, int vbosize, const cudaMat4* MVP_matrix) 
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<vbosize/3){
		int vboIdx = 3*index;

		glm::vec3 newPoint = 
			multiplyMV(*MVP_matrix, glm::vec4(vbo[vboIdx], vbo[vboIdx+1], vbo[vboIdx+2], 1.f));

		vbo[vboIdx] = newPoint.x;
		vbo[vboIdx+1] = newPoint.y;
		vbo[vboIdx+2] = newPoint.z;
	}
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  const int* attr_idx_ptr = &ibo[3*index];

	  int comp_idx0 = 3*attr_idx_ptr[0];
	  int comp_idx1 = 3*attr_idx_ptr[1];
	  int comp_idx2 = 3*attr_idx_ptr[2];

	  glm::set(primitives[index].p0, vbo[comp_idx0], vbo[comp_idx0+1], vbo[comp_idx0+2]);
	  glm::set(primitives[index].p1, vbo[comp_idx1], vbo[comp_idx1+1], vbo[comp_idx1+2]);
	  glm::set(primitives[index].p2, vbo[comp_idx2], vbo[comp_idx2+1], vbo[comp_idx2+2]);

	  glm::set(primitives[index].c0, cbo[0], cbo[1], cbo[2]);
	  glm::set(primitives[index].c1, cbo[3], cbo[4], cbo[5]);
	  glm::set(primitives[index].c2, cbo[6], cbo[7], cbo[8]);
  }
}

__host__ __device__ glm::vec2 convertPixel2NormalCoord(float pixelX, float pixelY, glm::vec2 resolution)
{
	glm::vec2 normalPoint;
	normalPoint.x = (2*pixelX + 1 - resolution.x)/resolution.x;
	normalPoint.y = -(2*pixelY + 1 - resolution.y)/resolution.y;

	return normalPoint;
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(const triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution,
									int* fragLocks)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		triangle targetTriangle = primitives[index];

		// find the rounding box in the pixel coordinates
		glm::vec3 minPoint, maxPoint;
		getAABBForTriangle(targetTriangle, minPoint, maxPoint);

		int minPixel_X = (int)floor((minPoint.x + 1)*resolution.x/2);
		int maxPixel_X = (int)floor((maxPoint.x + 1)*resolution.x/2);
		int maxPixel_Y = (int)floor((-minPoint.y + 1)*resolution.y/2); // min/max are intentionally flipped
		int minPixel_Y = (int)floor((-maxPoint.y + 1)*resolution.y/2);

		if (minPixel_X >= resolution.x) return;
		if (minPixel_Y >= resolution.y) return;
		if (maxPixel_X < 0) return;
		if (maxPixel_Y < 0) return;

		minPixel_X = max(minPixel_X, 0);
		maxPixel_X = min(maxPixel_X, (int)resolution.x - 1);
		minPixel_Y = max(minPixel_Y, 0);
		maxPixel_Y = min(maxPixel_Y, (int)resolution.y - 1);

		glm::vec2 normalPoint;
		glm::vec3 baryCoord;
		fragment tempFragment;
		int pixIdx;
		// for each pixel point
		for (int y = minPixel_Y; y <= maxPixel_Y; y++) {
			pixIdx = y*(int)resolution.x + minPixel_X;
			for (int x = minPixel_X; x <= maxPixel_X; x++) {
				// detect an intersection in the triangle
				normalPoint = convertPixel2NormalCoord((float)x, (float)y, resolution);
				baryCoord = calculateBarycentricCoordinate(targetTriangle, normalPoint);
				if (isBarycentricCoordInBounds(baryCoord)) { // inside triangle
					glm::set(tempFragment.position, 
						normalPoint.x, 
						normalPoint.y, 
						getZAtCoordinate(baryCoord, targetTriangle));

					if (tempFragment.position.z < -0.1f || tempFragment.position.z > 1.f) {
						return;
					}

					tempFragment.color =
						baryCoord.x*targetTriangle.c0
						+ baryCoord.y*targetTriangle.c1
						+ baryCoord.z*targetTriangle.c2;

					tempFragment.normal = glm::normalize(
							glm::cross(targetTriangle.p1 - targetTriangle.p0, 
										targetTriangle.p2 - targetTriangle.p0));

					bool leaveLoop = false;
					while (!leaveLoop) {
						if (atomicCAS(&fragLocks[pixIdx], 0, 1) == 0) {
							// start of critical section
							if (depthbuffer[pixIdx].position.z > tempFragment.position.z) {
								memcpy(&depthbuffer[pixIdx], &tempFragment, sizeof(fragment));
							}
							// end of critical section	
							leaveLoop = true;
							__threadfence();
							atomicExch(&fragLocks[pixIdx], 0);
						}
					}
				}

				pixIdx++;
			} // for x
		} // for y
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, 
									const Light* light, glm::vec3 eyePosition, bool isCboEnabled, 
									bool isAntiAliasingEnabled, glm::vec3* framebuffer){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  //glm::set(depthbuffer[index].color, 0.5f, 0.f, 0.f);
	  //float depth = depthbuffer[index].position.z;
	  //if (depth < 1.f && depth > -1.f) {
		 // depth += 1.f;
		 // depth /= 2.f;
		 // glm::set(depthbuffer[index].color, depth, depth, depth);
	  //} else {
		 // glm::set(depthbuffer[index].color, 0.f, 0.f, 0.f);
	  //}

	  glm::vec3 currPoint = depthbuffer[index].position;
	  glm::vec3 normal = depthbuffer[index].normal;

	  glm::vec3 L = glm::normalize(light->position - currPoint);
	  float diffuse = max(glm::dot(normal, L), 0.f);

	  eyePosition = glm::vec3(0.f);
	  glm::vec3 V = glm::normalize(eyePosition - currPoint);
	  glm::vec3 H = glm::normalize(L + V);
	  float specular = max(pow(glm::dot(H, normal), 5.f), 0.f);

	  if (isCboEnabled) {
		  depthbuffer[index].color *= (0.4f*diffuse + 0.6f*specular) * light->color;
	  } else {
		  depthbuffer[index].color = (0.4f*diffuse + 0.6f*specular) * light->color;
	  }

	  if (isAntiAliasingEnabled) {
		  // use super-sampling
		  thrust::default_random_engine rng(hash(index));
		  thrust::uniform_real_distribution<float> u01(0,1);
		  int numOfSuperSamples = 0;
		  glm::vec3 colorAccumulator = depthbuffer[index].color;

		  if (x > 0 && x < (int)resolution.x && y > 0 && y < (int)resolution.y) {			  
			  numOfSuperSamples += 8;
			  colorAccumulator += depthbuffer[index-(int)resolution.x-1].color; // upper-left
			  colorAccumulator += depthbuffer[index-(int)resolution.x].color; // upper
			  colorAccumulator += depthbuffer[index-(int)resolution.x+1].color; // upper-right
			  colorAccumulator += depthbuffer[index-1].color; // left
			  colorAccumulator += depthbuffer[index+1].color; // right
			  colorAccumulator += depthbuffer[index+(int)resolution.x-1].color; // bottom-left
			  colorAccumulator += depthbuffer[index+(int)resolution.x].color; // bottom
			  colorAccumulator += depthbuffer[index+(int)resolution.x+1].color; // bottom-right
		  } else if ((x == 0 && y == 0)) {
			  numOfSuperSamples += 3;
			  colorAccumulator += depthbuffer[1].color; // right
			  colorAccumulator += depthbuffer[(int)resolution.x].color; // bottom
			  colorAccumulator += depthbuffer[(int)resolution.x+1].color; // bottom-right
		  } else if (x == (int)resolution.x-1 && y == (int)resolution.y-1) {
			  numOfSuperSamples += 3;
			  colorAccumulator += depthbuffer[index-(int)resolution.x-1].color; // upper-left
			  colorAccumulator += depthbuffer[index-(int)resolution.x].color; // upper
			  colorAccumulator += depthbuffer[index-1].color; // left
		  } else if (x == 0 && y == (int)resolution.y-1) {
			  numOfSuperSamples += 3;
			  colorAccumulator += depthbuffer[index-(int)resolution.x].color; // upper
			  colorAccumulator += depthbuffer[index-(int)resolution.x+1].color; // upper-right
			  colorAccumulator += depthbuffer[index+1].color; // right
		  } else if (x == (int)resolution.x-1 && y == 0) {
			  numOfSuperSamples += 3;
			  colorAccumulator += depthbuffer[index-1].color; // left
			  colorAccumulator += depthbuffer[index+(int)resolution.x-1].color; // bottom-left
			  colorAccumulator += depthbuffer[index+(int)resolution.x].color; // bottom
		  } else if (x == 0) {
			  numOfSuperSamples += 5;
			  colorAccumulator += depthbuffer[index-(int)resolution.x].color; // upper
			  colorAccumulator += depthbuffer[index-(int)resolution.x+1].color; // upper-right
			  colorAccumulator += depthbuffer[index+1].color; // right
			  colorAccumulator += depthbuffer[index+(int)resolution.x].color; // bottom
			  colorAccumulator += depthbuffer[index+(int)resolution.x+1].color; // bottom-right
		  } else if (x == (int)resolution.x-1) {
			  numOfSuperSamples += 5;
			  colorAccumulator += depthbuffer[index-(int)resolution.x-1].color; // upper-left
			  colorAccumulator += depthbuffer[index-(int)resolution.x].color; // upper
			  colorAccumulator += depthbuffer[index-1].color; // left
			  colorAccumulator += depthbuffer[index+(int)resolution.x-1].color; // bottom-left
			  colorAccumulator += depthbuffer[index+(int)resolution.x].color; // bottom
		  } else if (y == 0) {
			  numOfSuperSamples += 5;
			  colorAccumulator += depthbuffer[index-(int)resolution.x].color; // upper
			  colorAccumulator += depthbuffer[index-(int)resolution.x+1].color; // upper-right
			  colorAccumulator += depthbuffer[index+1].color; // right
			  colorAccumulator += depthbuffer[index+(int)resolution.x].color; // bottom
			  colorAccumulator += depthbuffer[index+(int)resolution.x+1].color; // bottom-right
		  } else if (y == (int)resolution.y-1) {
			  numOfSuperSamples += 5;
			  colorAccumulator += depthbuffer[index-(int)resolution.x-1].color; // upper-left
			  colorAccumulator += depthbuffer[index-(int)resolution.x].color; // upper
			  colorAccumulator += depthbuffer[index-1].color; // left
			  colorAccumulator += depthbuffer[index+(int)resolution.x-1].color; // bottom-left
			  colorAccumulator += depthbuffer[index+(int)resolution.x].color; // bottom
		  }

		  colorAccumulator /= (float)(numOfSuperSamples + 1);
		  framebuffer[index] = colorAccumulator;
	  }
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer, bool isAntiAliasingEnabled){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){
		if (!isAntiAliasingEnabled) {
			framebuffer[index] = depthbuffer[index].color;
		}
	}
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, 
                       float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,
					   const cudaMat4* hostMVP_mat, glm::vec3 eyePosition, 
					   bool isCboEnabled, bool isAntiAliasingEnabled)
{

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
  frag.position = glm::vec3(0,0,10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  Light hostLight;
  glm::set(hostLight.color, 1.f, 1.f, 1.f);
  hostLight.position = eyePosition;

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

  light = NULL;
  cudaMalloc((void**)&light, sizeof(Light));
  cudaMemcpy( light, &hostLight, sizeof(Light), cudaMemcpyHostToDevice);

  MVP_matrix = NULL;
  cudaMalloc((void**)&MVP_matrix, sizeof(cudaMat4));
  cudaMemcpy( MVP_matrix, hostMVP_mat, sizeof(cudaMat4), cudaMemcpyHostToDevice);

  fragLocks = NULL;
  cudaMalloc((void**)&fragLocks, (int)resolution.x*(int)resolution.y*sizeof(int));
  cudaMemset(fragLocks, 0, (int)resolution.x*(int)resolution.y*sizeof(int));

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, MVP_matrix);
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
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, fragLocks);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, light, eyePosition, isCboEnabled, 
	  isAntiAliasingEnabled, framebuffer);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer, isAntiAliasingEnabled);
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
  cudaFree( light );
  cudaFree( MVP_matrix );
  cudaFree( fragLocks );
}

