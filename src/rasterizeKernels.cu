// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
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

struct IsInvalidPixel
{
	__host__ __device__
	bool operator()(const fragment& fragmentVal)
	{
		return !fragmentVal.valid;
	}
};

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

__host__ __device__ glm::vec3 ScreenPointInWorldCoordinates(glm::vec2 resolution, float x, float y,
	                                                   glm::vec3 M, glm::vec3 A, glm::vec3 B)
{
	float sx = (x/(float)(resolution.x-1));
	float sy = (y/(float)(resolution.y-1));
	return (M - A*(resolution.x/2.0f)*(2.0f*sx - 1) - B*(resolution.y/2.0f)*(2.0f*sy - 1));
} 

__host__ __device__ glm::vec2 WorldToScreen(glm::vec2 resolution, glm::vec3 pt, glm::vec2 eyeLeftRight, glm::vec2 eyeTopBottom)
{
	/*float sx = (((M.x - pt.x)/(A.x * resolution.x/2)) + 1)/2;
	float sy = (((M.y - pt.y)/(B.y * resolution.y/2)) + 1)/2;
	glm::vec2 screenPoint;
	screenPoint.x = sx*(resolution.x - 1);
	screenPoint.y = sy*(resolution.y - 1);
	return screenPoint;*/
	
	glm::vec2 screenPoint;
	screenPoint.x = (pt.x - (eyeLeftRight.x))*(resolution.x/2);
	screenPoint.y = (pt.y - (eyeTopBottom.x))*(resolution.y/2);
	return screenPoint;
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
      
	  int PBOIndex = resolution.x - 1 - x + ((resolution.y - 1 - y) * resolution.x);

      // Each thread writes one pixel location in the texture (textel)
      PBOpos[PBOIndex].w = 0.5;
      PBOpos[PBOIndex].x = color.x;     
      PBOpos[PBOIndex].y = color.y;
      PBOpos[PBOIndex].z = color.z;
  }
}


//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, float* vboOriginal, float* normals, int vbosize, cudaMat4 transform, cudaMat4 modelView){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  /*{
		  printf("\nVBO %i: %f, %f, %f\n", index, vbo[3*index], vbo[3*index+1], vbo[3*index+2]);
	  }*/
	  float4 point = make_float4(vbo[3*index], vbo[3*index+1], vbo[3*index+2], 1);
	  float4 transformW = make_float4(transform.w[0], transform.w[1], transform.w[2], transform.w[3]);
	  float w = dot(transformW, point);

	  float4 transformX = make_float4(transform.x.x, transform.x.y, transform.x.z, transform.x.w);
	  float4 transformY = make_float4(transform.y.x, transform.y.y, transform.y.z, transform.y.w);
	  float4 transformZ = make_float4(transform.z.x, transform.z.y, transform.z.z, transform.z.w);

	  vbo[3*index]     = dot(transformX, point);
	  vbo[3*index + 1] = dot(transformY, point);
	  vbo[3*index + 2] = dot(transformZ, point);
	  
	  // Divide by w to create homogeneous coordinates
	  if (w != 0)
	  {
		  vbo[3*index]     /= w;
		  vbo[3*index + 1] /= w;
		  vbo[3*index + 2] /= w;
	  }

	  float4 modelViewX = make_float4(modelView.x.x, modelView.x.y, modelView.x.z, modelView.x.w);
	  float4 modelViewY = make_float4(modelView.y.x, modelView.y.y, modelView.y.z, modelView.y.w);
	  float4 modelViewZ = make_float4(modelView.z.x, modelView.z.y, modelView.z.z, modelView.z.w);
	  vboOriginal[3*index]     = dot(modelViewX, point);
	  vboOriginal[3*index + 1] = dot(modelViewY, point);
	  vboOriginal[3*index + 2] = dot(modelViewZ, point);

	  float4 normal = make_float4(normals[3*index], normals[3*index+1], normals[3*index+2], 0);
	  normals[3*index]     = dot(modelViewX, normal);
	  normals[3*index + 1] = dot(modelViewY, normal);
	  normals[3*index + 2] = dot(modelViewZ, normal);
  }
}
__device__ void PrintVec3(int index, glm::vec3& value)
{
	printf("Index: %i,  Value: (%f, %f, %f)\n", index, value.x, value.y, value.z);
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, 
	                                    float* vboOriginal, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int vboIndex = 3*ibo[3*index];
	  primitives[index].p0 = glm::vec3(vbo[vboIndex], vbo[vboIndex+1], vbo[vboIndex+2]);
	  primitives[index].p0Original = glm::vec3(vboOriginal[vboIndex], vboOriginal[vboIndex+1], vboOriginal[vboIndex+2]);
	  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].n0 = glm::vec3(nbo[vboIndex], nbo[vboIndex+1], nbo[vboIndex+2]);
	  vboIndex = 3*ibo[3*index + 1];
	  primitives[index].p1 = glm::vec3(vbo[vboIndex], vbo[vboIndex+1], vbo[vboIndex+2]);
	  primitives[index].p1Original = glm::vec3(vboOriginal[vboIndex], vboOriginal[vboIndex+1], vboOriginal[vboIndex+2]);
	  primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  primitives[index].n1 = glm::vec3(nbo[vboIndex], nbo[vboIndex+1], nbo[vboIndex+2]);
	  vboIndex = 3*ibo[3*index + 2];
	  primitives[index].p2 = glm::vec3(vbo[vboIndex], vbo[vboIndex+1], vbo[vboIndex+2]);
	  primitives[index].p2Original = glm::vec3(vboOriginal[vboIndex], vboOriginal[vboIndex+1], vboOriginal[vboIndex+2]);
	  primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
	  primitives[index].n2 = glm::vec3(nbo[vboIndex], nbo[vboIndex+1], nbo[vboIndex+2]);
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, 
	                                glm::vec2 eyeLeftRight, glm::vec2 eyeTopBottom, int* depthBufferLock, bool antialias, bool colorInterp){

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  float3 p0 = make_float3(primitives[index].p0.x, primitives[index].p0.y, primitives[index].p0.z);
	  float3 p1 = make_float3(primitives[index].p1.x, primitives[index].p1.y, primitives[index].p1.z);
	  float3 p2 = make_float3(primitives[index].p2.x, primitives[index].p2.y, primitives[index].p2.z);
	  //printf("Depth -- Index: %i, P0: %f, P1: %f, P2: %f\n", index, primitives[index].p0.z, primitives[index].p1.z, primitives[index].p2.z);
	  
	  // Perform Back Culling to ignore vertices with normals facing the other way
	  if (glm::dot(-primitives[index].p0Original,primitives[index].n0) < 0 &&
		  glm::dot(-primitives[index].p1Original,primitives[index].n1) < 0 &&
		  glm::dot(-primitives[index].p2Original,primitives[index].n2) < 0)
	  {
		  return;
	  }

	  /*printf("Pos: Index: %i P0: (%f, %f, %f) P1: (%f, %f, %f) P2: (%f, %f, %f)\n", index, 
						  primitives[index].p0Original.x, primitives[index].p0Original.y, primitives[index].p0Original.z,
						  primitives[index].p1Original.x, primitives[index].p1Original.y, primitives[index].p1Original.z,
						  primitives[index].p2Original.x, primitives[index].p2Original.y, primitives[index].p2Original.z);*/

	  // Convert the coordinates into screen space
	  glm::vec2 p0Screen = WorldToScreen(resolution, primitives[index].p0, eyeLeftRight, eyeTopBottom);
	  glm::vec2 p1Screen = WorldToScreen(resolution, primitives[index].p1, eyeLeftRight, eyeTopBottom);
	  glm::vec2 p2Screen = WorldToScreen(resolution, primitives[index].p2, eyeLeftRight, eyeTopBottom);

	  glm::vec2 p0ScreenActual = p0Screen;
	  glm::vec2 p1ScreenActual = p1Screen;
	  glm::vec2 p2ScreenActual = p2Screen;

	  // Find min and max y-values to loop through and sort by y
	  float yMax = p0Screen.y;
	  if (yMax < p1Screen.y)
	  {
		  yMax = p1Screen.y;
		  // Swap
		  glm::vec2 temp = p0Screen;
		  p0Screen = p1Screen;
		  p1Screen = temp;
	  }
	  if (yMax < p2Screen.y)
	  {
		  yMax = p2Screen.y;
		  // Swap
		  glm::vec2 temp = p0Screen;
		  p0Screen = p2Screen;
		  p2Screen = temp;
	  }

	  // Sort p1 and p2
	  if (p1Screen.y < p2Screen.y)
	  {
		  // Swap
		  glm::vec2 temp = p1Screen;
		  p1Screen = p2Screen;
		  p2Screen = temp;
	  }

	  float& y0 = p0Screen.y;
	  float& y1 = p1Screen.y;
	  float& y2 = p2Screen.y;
	  float& x0 = p0Screen.x;
	  float& x1 = p1Screen.x;
	  float& x2 = p2Screen.x;

	  int yStart = floor(y0);
	  int yEnd   = ceil(y2);
	  yStart = min(yStart, (int)(resolution.y-0.9f));
	  yEnd = max(yEnd, 0);
	  if (yStart == 0 && yEnd == 0)
		  return;

	  // Loop through yMin to yMax - the scanlines
	  for (int y=yStart; y>=yEnd; --y)
	  {
		  float xMin = 100000;
		  float xMax = -xMin;
		  if ((float)y<=y0 && (float)y>=y1)
		  {
			  float x;
			  if (y1 != y0)
			  {
				  x = x0 + ((float)y-y0)*(x1-x0)/(y1-y0);
				  if (x<xMin)
					  xMin = x;
				  if (x>xMax)
					  xMax = x;
			  }
			  else
			  {
				  // Line parallel to x
				  if (x0<xMin)
					  xMin = x0;
				  if (x0>xMax)
					  xMax = x0;
				  if (x1<xMin)
					  xMin = x1;
				  if (x1>xMax)
					  xMax = x1;
			  }
			  
		  }

		  if ((float)y<=y0 && (float)y>=y2)
		  {
			  if (y2 != y0)
			  {
				  float x = x0 + ((float)y-y0)*(x2-x0)/(y2-y0);
				  if (x<xMin)
					  xMin = x;
				  if (x>xMax)
					  xMax = x;
			  }
			  else
			  {
				  // Line parallel to x
				  if (x0<xMin)
					  xMin = x0;
				  if (x0>xMax)
					  xMax = x0;
				  if (x2<xMin)
					  xMin = x2;
				  if (x2>xMax)
					  xMax = x2;
			  }
		  }

		  if ((float)y<=y1 && (float)y>=y2)
		  {
			  if (y1 != y2)
			  {
				  float x = x1 + ((float)y-y1)*(x2-x1)/(y2-y1);
				  if (x<xMin)
					  xMin = x;
				  if (x>xMax)
					  xMax = x;
			  }
			  else
			  {
				  // Line parallel to x
				  if (x2<xMin)
					  xMin = x2;
				  if (x2>xMax)
					  xMax = x2;
				  if (x1<xMin)
					  xMin = x1;
				  if (x1>xMax)
					  xMax = x1;
			  }
		  }

		  // Check if the depth is smallest
		  int xStart = ceil(xMin);
		  int xEnd   = floor(xMax);

		  xStart = max(xStart, 0);
		  xEnd = min(xEnd, (int)resolution.x);

		  //if(y==yStart-1 && (index == 1 || index ==2))
		  /*{
			  printf("y: %i, xStart: %i, xEnd: %i\n", y, xStart, xEnd);
		  }*/

		  for (int x=xStart; x<=xEnd; ++x)
		  {
			  if (x<resolution.x && y >=0 && y<resolution.y)
			  {
				  int pixelIndex = x + y*resolution.y;

				  glm::vec2 pointOnScreen = glm::vec2(x,y);
				  triangle tri;
				  tri.p0 = glm::vec3(p0ScreenActual.x, p0ScreenActual.y, 1);
				  tri.p1 = glm::vec3(p1ScreenActual.x, p1ScreenActual.y, 1);
				  tri.p2 = glm::vec3(p2ScreenActual.x, p2ScreenActual.y, 1);
				 
				  glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tri, pointOnScreen);
				  
				  // If the point is outside the triangle, then just continue to the next point
				  if (!antialias && !isBarycentricCoordInBounds(barycentricCoord))
				  {
					  continue;
				  }
				  // Find the depth of the point
				  float pointZ = primitives[index].p0.z * barycentricCoord.x + primitives[index].p1.z * barycentricCoord.y + primitives[index].p2.z * barycentricCoord.z;
				  //pointZ = -pointZ;
				  // Store into depth buffer only if this is the minimum
				  if (pointZ < depthbuffer[pixelIndex].depthVal)
				  {
					  glm::vec3 newColor = primitives[index].c0;
					  if (antialias)
					  {
						  glm::vec2 pointOnScreenAntialias;
						  glm::vec3 barycentricCoordAntialias;
						  newColor = glm::vec3(0,0,0);
						  bool outside = true;

						  pointOnScreenAntialias = glm::vec2((float)x-0.25,(float)y-0.25);
						  barycentricCoordAntialias = calculateBarycentricCoordinate(tri, pointOnScreenAntialias);
						  if (isBarycentricCoordInBounds(barycentricCoordAntialias))
						  {
							  if (colorInterp)
								  newColor += primitives[index].c0 * barycentricCoordAntialias.x + primitives[index].c1 * barycentricCoordAntialias.y + primitives[index].c2 * barycentricCoordAntialias.z;
							  else
								  newColor += primitives[index].c0;

							  outside = false;
						  }

						  pointOnScreenAntialias = glm::vec2((float)x-0.25,(float)y+0.25);
						  barycentricCoordAntialias = calculateBarycentricCoordinate(tri, pointOnScreenAntialias);
						  if (isBarycentricCoordInBounds(barycentricCoordAntialias))
						  {
							  if (colorInterp)
								  newColor += primitives[index].c0 * barycentricCoordAntialias.x + primitives[index].c1 * barycentricCoordAntialias.y + primitives[index].c2 * barycentricCoordAntialias.z;
							  else
								  newColor += primitives[index].c0;

							  outside = false;
						  }

						  pointOnScreenAntialias = glm::vec2((float)x+0.25,(float)y-0.25);
						  barycentricCoordAntialias = calculateBarycentricCoordinate(tri, pointOnScreenAntialias);
						  if (isBarycentricCoordInBounds(barycentricCoordAntialias))
						  {
							  if (colorInterp)
								  newColor += primitives[index].c0 * barycentricCoordAntialias.x + primitives[index].c1 * barycentricCoordAntialias.y + primitives[index].c2 * barycentricCoordAntialias.z;
							  else
								  newColor += primitives[index].c0;

							  outside = false;
						  }

						  pointOnScreenAntialias = glm::vec2((float)x+0.25,(float)y+0.25);
						  barycentricCoordAntialias = calculateBarycentricCoordinate(tri, pointOnScreenAntialias);
						  if (isBarycentricCoordInBounds(barycentricCoordAntialias))
						  {
							  if (colorInterp)
								  newColor += primitives[index].c0 * barycentricCoordAntialias.x + primitives[index].c1 * barycentricCoordAntialias.y + primitives[index].c2 * barycentricCoordAntialias.z;
							  else
								  newColor += primitives[index].c0;

							  outside = false;
						  }

						  //newColor += (primitives[index].c0 * barycentricCoord.x + primitives[index].c1 * barycentricCoord.y + primitives[index].c2 * barycentricCoord.z)*4.f;
						  if (outside)
							  continue;

						  newColor /= 4.f;
					  }
					  else if (colorInterp)
					  {
						  newColor = primitives[index].c0 * barycentricCoord.x + primitives[index].c1 * barycentricCoord.y + primitives[index].c2 * barycentricCoord.z;
					  }
					  glm::vec3 newNormal = primitives[index].n0 * barycentricCoord.x + primitives[index].n1 * barycentricCoord.y + primitives[index].n2 * barycentricCoord.z;
					  glm::vec3 newPosition = primitives[index].p0Original * barycentricCoord.x + primitives[index].p1Original * barycentricCoord.y + primitives[index].p2Original * barycentricCoord.z;
					  
					  bool loop =true;
					  
					  do
					  {
						  // Apply a lock, the threads go into this only if the old value was 0
						  // i.e. if some other thread is accessing the critical section, other threads be remain in the while loop
						  if (atomicExch(&(depthBufferLock[pixelIndex]), 1) == 0)
						  //if (atomicExch(&(lock[0]), 1) == 0)
						  {
							  // Crticial Section
							  if (pointZ < depthbuffer[pixelIndex].depthVal)
							  {
								  depthbuffer[pixelIndex].color = newColor;

								  depthbuffer[pixelIndex].normal = newNormal;
								  depthbuffer[pixelIndex].position = newPosition;
								  depthbuffer[pixelIndex].depthVal = pointZ;
								  depthbuffer[pixelIndex].pixelIndex = pixelIndex;
								  depthbuffer[pixelIndex].valid = true;
							  }
							  loop = false;
							  atomicExch(&(depthBufferLock[pixelIndex]), 0);
						  }
					  } while (loop);
				  }
			  }
		  }
	  }
  }
}

__global__ void FindMinMaxDepth(fragment* depthbuffer, int depthBufferSize, float* depthValue)
{
	depthValue[0] = 10000;
	depthValue[1] = -10000;
	for (int i=0; i< depthBufferSize; ++i)
	{
		if (depthbuffer[i].valid && depthValue[0] > depthbuffer[i].depthVal)
			depthValue[0] = depthbuffer[i].depthVal;

		if (depthbuffer[i].valid && depthValue[1] < depthbuffer[i].depthVal)
			depthValue[1] = depthbuffer[i].depthVal;
	}

	depthValue[1] -= depthValue[0];
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, float3* lightDir, float3* lightColor, int lightCount, 
	                                bool* stencil, bool stencilPresent, float* boundZ, bool depthShade, bool toShade){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  if (depthbuffer[index].valid)
	  {
		  if (!stencilPresent || (stencilPresent && stencil[depthbuffer[index].pixelIndex]))
		  {
			  if (depthShade)
			  {
				  float color = 1.f - (0.15+(0.7*(depthbuffer[index].depthVal - boundZ[0])/boundZ[1]));
				  depthbuffer[index].color = glm::vec3(color,color,color);
			  }
			  else if(toShade)
			  {
				  // View Direction is camera - point. Camera is at 0,0,0
				  float3 viewDir = make_float3(-depthbuffer[index].position.x, -depthbuffer[index].position.y, -depthbuffer[index].position.z);
				  if (viewDir.x != 0 && viewDir.y != 0 && viewDir.z != 0)
					  viewDir = normalize(viewDir);
		  
				  float3 normal = make_float3(depthbuffer[index].normal.x, depthbuffer[index].normal.y, depthbuffer[index].normal.z);
				  if (normal.x != 0 && normal.y != 0 && normal.z != 0)
					  normal = normalize(normal);
		  
				  float3 finalColor = make_float3(0, 0, 0);
				  float specularity = 20;
				  float3 color = make_float3(depthbuffer[index].color.x, depthbuffer[index].color.y, depthbuffer[index].color.z);
				  float kd = 0.6f;
				  float ks = 0.1f;
				  for (int k = 0; k < lightCount; ++k)
				  {
					  float3 halfVec = normalize(viewDir + lightDir[k]);
					  float cosTh = dot(normal, halfVec);
					  float cosTi = dot(normal, lightDir[k]);
					  if (cosTi < 0) cosTi = 0;
					  if (cosTi > 1) cosTi = 1;
					  if (cosTh < 0) cosTh = 0;
					  if (cosTh > 1) cosTh = 1;
					  finalColor += (kd*color/*/3.14*/ + ks/*(specularity + 8)/(8*3.14)*/ * pow(cosTh, specularity)) * lightColor[k] * cosTi;
				  }

				  depthbuffer[index].color.x = finalColor.x;
				  depthbuffer[index].color.y = finalColor.y;
				  depthbuffer[index].color.z = finalColor.z;
			  }
		  }
		  else
		  {
			  depthbuffer[index].color = glm::vec3(0,0,0);
		  }
	  }
	  else
	  {
		  depthbuffer[index].color = glm::vec3(0,0,0);
	  }
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
	  if (depthbuffer[index].valid)
		  framebuffer[depthbuffer[index].pixelIndex] = depthbuffer[index].color;
  }
}

//Writes fragment colors to the framebuffer
__global__ void renderAntialias(glm::vec2 resolution, glm::vec2 newResolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
	  int depthbufferOffsetX = 2*x + 1;
	  int depthbufferOffsetY = 2*y + 1;
	  int depthbufferIndex = (depthbufferOffsetX) + (depthbufferOffsetY)*resolution.y*2.f;

	  // X
	  if (depthbuffer[depthbufferIndex].valid)
	  {
		  framebuffer[index] += depthbuffer[depthbufferIndex].color;
	  }
	  if (depthbufferOffsetY-1 >= 0)
	  {
		  depthbufferIndex = (depthbufferOffsetX) + (depthbufferOffsetY-1)*resolution.y*2.f;
		  if (depthbuffer[depthbufferIndex].valid)
		  {
			  framebuffer[index] += depthbuffer[depthbufferIndex].color;
		  }
	  }
	  if (depthbufferOffsetY+1 < newResolution.y)
	  {
		  depthbufferIndex = (depthbufferOffsetX-1) + (depthbufferOffsetY+1)*resolution.y*2.f;
		  if (depthbuffer[depthbufferIndex].valid)
		  {
			  framebuffer[index] += depthbuffer[depthbufferIndex].color;
		  }
	  }

	  // X-1
	  if (depthbufferOffsetX - 1 >= 0)
	  {
		  if (depthbufferOffsetY-1 >= 0)
		  {
			  depthbufferIndex = (depthbufferOffsetX-1) + (depthbufferOffsetY-1)*resolution.y*2.f;
			  if (depthbuffer[depthbufferIndex].valid)
			  {
				  framebuffer[index] += depthbuffer[depthbufferIndex].color;
			  }
		  }

		  depthbufferIndex = (depthbufferOffsetX-1) + (depthbufferOffsetY)*resolution.y*2.f;
		  if (depthbuffer[depthbufferIndex].valid)
		  {
			  framebuffer[index] += depthbuffer[depthbufferIndex].color;
		  }

		  if (depthbufferOffsetY+1 < newResolution.y)
		  {
			  depthbufferIndex = (depthbufferOffsetX-1) + (depthbufferOffsetY+1)*resolution.y*2.f;
			  if (depthbuffer[depthbufferIndex].valid)
			  {
				  framebuffer[index] += depthbuffer[depthbufferIndex].color;
			  }
		  }
	  }

	  // X+1
	  if (depthbufferOffsetX + 1 < newResolution.x)
	  {
		  if (depthbufferOffsetY-1 >= 0)
		  {
			  depthbufferIndex = (depthbufferOffsetX+1) + (depthbufferOffsetY-1)*resolution.y*2.f;
			  if (depthbuffer[depthbufferIndex].valid)
			  {
				  framebuffer[index] += depthbuffer[depthbufferIndex].color;
			  }
		  }

		  depthbufferIndex = (depthbufferOffsetX+1) + (depthbufferOffsetY)*resolution.y*2.f;
		  if (depthbuffer[depthbufferIndex].valid)
		  {
			  framebuffer[index] += depthbuffer[depthbufferIndex].color;
		  }

		  if (depthbufferOffsetY+1 < newResolution.y)
		  {
			  depthbufferIndex = (depthbufferOffsetX+1) + (depthbufferOffsetY+1)*resolution.y*2.f;
			  if (depthbuffer[depthbufferIndex].valid)
			  {
				  framebuffer[index] += depthbuffer[depthbufferIndex].color;
			  }
		  }
	  }
	  
	  framebuffer[index] /= 9.f;
  }
}

__global__ void printCulled(int* culled)
{
	printf("Culled: %i\n", culled[0]);
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, 
	                   int* ibo, int ibosize, float* nbo, int nbosize, Eye eye, cudaMat4 transform, cudaMat4 modelView, cudaMat4 perspective)
{
	glm::vec3 M, A, B;
	float distImagePlaneFromCamera;
	eye.GetParametersForScreenTransform(resolution, M, A, B, distImagePlaneFromCamera);

	//printf("M: (%f, %f, %f)   A: (%f, %f, %f)   B: (%f, %f, %f)", M.x, M.y, M.z, A.x, A.y, A.z, B.x, B.y, B.z);

	bool objAntialias = false;
	bool colorInterp = true;
	bool antialias = false;
	bool depthShade = false;
	bool toBeShaded = true;
	bool stencilPresent = false;
	
	int totalPixels = (int)resolution.x*(int)resolution.y;
	int newTotalPixels = totalPixels;
	glm::vec2 newResolution = resolution;
	if (objAntialias)
	{
		newResolution *= 2.f;
		newTotalPixels *= 4;
	}

  // set up crucial magic
  int tileSize = 8;
  int tileSizeInitial = tileSize;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
  dim3 fullBlocksPerGridNew((int)ceil(float(newResolution.x)/float(tileSize)), (int)ceil(float(newResolution.y)/float(tileSize)));
  dim3 fullBlocksPerGridAntialias((int)ceil(float(newResolution.x)/float(tileSize)), (int)ceil(float(newResolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, newTotalPixels*sizeof(fragment));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.valid = false;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,0);
  frag.depthVal = 10000.f;
  frag.pixelIndex = 0;
  clearDepthBuffer<<<fullBlocksPerGridAntialias, threadsPerBlock>>>(newResolution, depthbuffer, frag);

  bool* stencil = new bool[newTotalPixels];
  bool* device_stencil = NULL;

  if (stencilPresent)
  {
	  // Create Stencil
	  for (int i = 0; i<newResolution.x; ++i)
	  {
		  for (int j = 0; j<newResolution.y; ++j)
		  {
			  int indexVal = i+j*newResolution.x;
			  stencil[indexVal] = false;
			  if (i > 150 && i < 300 && j > 300 && j < 525)
				  stencil[indexVal] = true;
			  if (i > 350 && i < 550 && j > 550 && j < 700)
				  stencil[indexVal] = true;
		  }
	  }

	  cudaMalloc((void**)&device_stencil, newTotalPixels*sizeof(bool));
	  cudaMemcpy( device_stencil, stencil, newTotalPixels*sizeof(bool), cudaMemcpyHostToDevice);
  }
  int* depthBufferLock = NULL;
  cudaMalloc((void**)&depthBufferLock, newTotalPixels*sizeof(int));
  cudaMemset(depthBufferLock, 0, newTotalPixels*sizeof(int));
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

  float* device_vboWorld = NULL;
  cudaMalloc((void**)&device_vboWorld, vbosize*sizeof(float));
  cudaMemcpy( device_vboWorld, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

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
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vboWorld, device_nbo, vbosize, transform, modelView);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, device_vboWorld, primitives);

  cudaDeviceSynchronize();
  
  //------------------------------
  //rasterization
  //------------------------------
  glm::vec2 eyeLeftRight = glm::vec2(eye.l, eye.r);
  glm::vec2 eyeTopBottom = glm::vec2(eye.b, eye.t);
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, newResolution, eyeLeftRight, eyeTopBottom, 
	                                                 depthBufferLock, antialias, colorInterp);
  cudaDeviceSynchronize();
  
  float totalValidPixels = newTotalPixels;
  if (!objAntialias)
  {
	  // wrap raw pointer with a device_ptr for Stream compaction
	  thrust::device_ptr<fragment> devDepthBufferPtr(depthbuffer);
	  thrust::device_ptr<fragment> devDepthBufferEndPtr = thrust::remove_if(devDepthBufferPtr, devDepthBufferPtr+totalPixels, IsInvalidPixel());
	  totalValidPixels = devDepthBufferEndPtr.get() - devDepthBufferPtr.get();
  
	  // Create blocks for lesser number of pixels found by stream compaction
	  fullBlocksPerGridNew = dim3((int)ceil(float(newResolution.x)/float(tileSizeInitial)), (int)ceil((float(totalValidPixels)/float(newResolution.y))/float(tileSizeInitial)));
  }

  float* device_depthValue = NULL;
  cudaMalloc((void**)&device_depthValue, 2*sizeof(float));
  FindMinMaxDepth<<<1,1>>>(depthbuffer, totalValidPixels, device_depthValue);
  
  //------------------------------
  //fragment shader
  //------------------------------
  int numLights = 4;
  float3* lightDir = new float3[numLights];
  lightDir[0] = make_float3(0, 0, 1);
  lightDir[1] = make_float3(0, 0, -1);
  lightDir[2] = make_float3(1, 1, 0);
  lightDir[3] = make_float3(-1, -1, 0);
  for (int i = 0; i < numLights; ++i)
  {
	  lightDir[i] = normalize(lightDir[i]);
  }

  float3* lightColor = new float3[numLights];
  lightColor[0] = make_float3(1, 1, 1);
  lightColor[1] = make_float3(1,1,1);
  lightColor[2] = make_float3(1, 1, 1);
  lightColor[3] = make_float3(1, 1, 1);

  float3* device_lightDir = NULL;
  cudaMalloc((void**)&device_lightDir, numLights*sizeof(float3));
  cudaMemcpy( device_lightDir, lightDir, numLights*sizeof(float3), cudaMemcpyHostToDevice);
  
  float3* device_lightColor = NULL;
  cudaMalloc((void**)&device_lightColor, numLights*sizeof(float3));
  cudaMemcpy( device_lightColor, lightColor, numLights*sizeof(float3), cudaMemcpyHostToDevice);
  
  if (toBeShaded || depthShade || stencilPresent)
  {
	  fragmentShadeKernel<<<fullBlocksPerGridNew, threadsPerBlock>>>(depthbuffer, newResolution, device_lightDir, device_lightColor, numLights, 
		                                                             device_stencil, stencilPresent, device_depthValue, depthShade, toBeShaded);
  }

  cudaDeviceSynchronize();
  delete[] lightDir;
  delete[] lightColor;
  cudaFree(device_lightDir);
  cudaFree(device_lightColor);
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  if (!objAntialias)
	  render<<<fullBlocksPerGridNew, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  else
	  renderAntialias<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, newResolution, depthbuffer, framebuffer);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

 // delete[] boundZ;
  cudaFree(device_depthValue);
  cudaFree(device_stencil);
  cudaFree(depthBufferLock);
  cudaFree(device_vboWorld);
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
}

