// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

//#include <sm_11_atomic_functions.h>
//#include <sm_12_atomic_functions.h>
//#include <sm_13_double_functions.h>
//#include <sm_20_atomic_functions.h>
//#include <sm_20_intrinsics.h>
//#include <sm_30_intrinsics.h>

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
//ADDED
float *device_nbo;
//float *device_fnbo;
//float *device_originalDepth;
triangle *originalPrimitives;
Light *cudaLights;
//ADDED
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
	int index = (y * resolution.x) + x;
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

__device__ void atomicCompareAndSwapFrag(int screenSpaceX, int screenSpaceY, glm::vec2 resolution, fragment *depthbuffer, fragment val)
{
	unsigned int old;
	unsigned int *lockAddress = &(depthbuffer[screenSpaceY * (int)(resolution.x + 0.1f) + screenSpaceX].lockVariable);
	do {
		old = atomicCAS(lockAddress, 0, 1);
		if(old == 0)	//Can Swap Fragment after depth test
		{
			if(val.worldDepth < depthbuffer[screenSpaceY * (int)(resolution.x + 0.1f) + screenSpaceX].worldDepth)
			{
				depthbuffer[screenSpaceY * (int)(resolution.x + 0.1f) + screenSpaceX] = val;
			}
			else
			{
				depthbuffer[screenSpaceY * (int)(resolution.x + 0.1f) + screenSpaceX].lockVariable = 0;
			}
		}
	} while(old != 0);
}

__device__ void modifyFragmentAfterDepthTest(int screenSpaceX, int screenSpaceY, triangle screenSpaceTriangle, triangle originalTriangle,
	glm::vec3 eye, glm::vec2 resolution, fragment *depthbuffer)
{
	glm::vec3 barycentricCoords = calculateBarycentricCoordinate(screenSpaceTriangle, glm::vec2((float)screenSpaceX, (float)screenSpaceY));
	float z = giveWorldSpaceDepth(barycentricCoords, originalTriangle, eye);

	fragment f = getFromDepthbuffer(screenSpaceX, screenSpaceY, depthbuffer, resolution);
	
	if(f.worldDepth > z)
	{
		f.color = getColorAtBarycentricCoordinate(barycentricCoords, screenSpaceTriangle);
		//f.normal = screenSpaceTriangle.faceNormal;
		f.screenSpaceX = screenSpaceX;
		f.screenSpaceY = screenSpaceY;
		f.worldDepth = z;
		f.lockVariable = 0;
		f.worldPosition = getPositionAtBarycentricCoordinate(barycentricCoords, originalTriangle);
		f.normal = getNormalAtBarycentricCoordinate(barycentricCoords, originalTriangle);
		f.normal = glm::normalize(f.normal);
		//writeToDepthbuffer(screenSpaceX, screenSpaceY, f, depthbuffer, resolution);
		atomicCompareAndSwapFrag(screenSpaceX, screenSpaceY, resolution, depthbuffer, f);
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
      f.screenSpaceX = x;
      f.screenSpaceY = y;
      buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int index2 = (resolution.x - x) + ((resolution.y - y) * resolution.x);	//To inveert in X and Y to get correct image

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
      PBOpos[index2].w = 0;
      PBOpos[index2].x = color.x;     
      PBOpos[index2].y = color.y;
      PBOpos[index2].z = color.z;
  }
}


////Original Depth Computing Kernel
//__global__ void originalDepthComputingKernel(float *vbo, int vbosize, float *originalDepth, glm::vec3 eye)
//{
//	//Remember, this works only if model matrix is Identity, otherwise, vbo has to be first brought to world co-ordinates, then depth must be computed
//	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//	if(index < vbosize / 3)
//	{
//		glm::vec3 vertex(vbo[index], vbo[index + 1], vbo[index + 2]);
//		originalDepth[index] = glm::distance(vertex, eye);
//	}
//}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, cudaMat4 modelViewProjection)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < vbosize / 3)
  {
	  glm::vec4 vertex4(vbo[3 * index], vbo[3 * index + 1], vbo[3 * index + 2], 1.0f);
	  glm::vec3 vertex = multiplyMVWithHomogenization(modelViewProjection, vertex4);
	  //Remember to homogenize the resulting vector, since perspective projection is not an affine transform
	  vbo[3 * index] = vertex.x;
	  vbo[3 * index + 1] = vertex.y;
	  vbo[3 * index + 2] = vertex.z;

	  //printf("%s\t%d\t%f\t%f\t%f\n", "Index", index, vbo[3 * index], vbo[3 * index + 1], vbo[3 * index + 2]);
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float *nbo, int nbosize, triangle* primitives)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index < primitivesCount)
  {
	  int a = 3 * ibo[3 * index];
	  int b = 3 * ibo[3 * index + 1];
	  int c = 3 * ibo[3 * index + 2];

	 // //Adding normals to triangles before bringing to NDC
	 // glm::vec3 A = glm::vec3(vbo[b], vbo[b + 1], vbo[b + 2]) - glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]);
	 // glm::vec3 B = glm::vec3(vbo[c], vbo[c + 1], vbo[c + 2]) - glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]);
	 // //Reusing A instead of a new normal vector
	 // A = glm::cross(A, B);
	 // if(glm::dot(glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]), A) > 0.0f)
	 // {
		//  A = -A;
	 // }
	 // if(glm::length(A) > 0.001f)
	 // {
		//A = glm::normalize(A);
	 // }
	 // primitives[index].faceNormal = A;


	  primitives[index].p0 = glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]);
	  primitives[index].p1 = glm::vec3(vbo[b], vbo[b + 1], vbo[b + 2]);
	  primitives[index].p2 = glm::vec3(vbo[c], vbo[c + 1], vbo[c + 2]);
	  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
	  primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
	  primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);

	  primitives[index].n0 = glm::vec3(nbo[a], nbo[a + 1], nbo[a + 2]);
	  primitives[index].n1 = glm::vec3(nbo[b], nbo[b + 1], nbo[b + 2]);
	  primitives[index].n2 = glm::vec3(nbo[c], nbo[c + 1], nbo[c + 2]);

	 // //Adding normals to triangles
	 // glm::vec3 A = glm::vec3(vbo[b], vbo[b + 1], vbo[b + 2]) - glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]);
	 // glm::vec3 B = glm::vec3(vbo[c], vbo[c + 1], vbo[c + 2]) - glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]);
	 // //Reusing A instead of a new normal vector
	 // A = glm::cross(A, B);
	 // if(glm::dot(glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]), A) > 0.0f)
	 // {
		//  A = -A;
	 // }
	 // if(glm::length(A) > 0.001f)
	 // {
		//A = glm::normalize(A);
	 // }
	 // primitives[index].faceNormal = A;

	  //printf("%s\t%d\t%d\t%d\n", "IBO Indices", ibo[index], ibo[index + 1], ibo[index + 2]);
  }
}

__global__ void geometryShaderKernel(float *fnbo, triangle *primitives, int numberOfPrimitives)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < numberOfPrimitives)
	{
		triangle tri = primitives[index];
		glm::vec3 a = tri.p1 - tri.p0;
		glm::vec3 b = tri.p2 - tri.p0;
		glm::vec3 faceNormal = glm::cross(a, b);

		//Vector from light to one vertex of triangle surface, which is located in the same place as eye i.e. origin, is just the position itself
		if(glm::dot(tri.p0, faceNormal) > 0.0f)
		{
			faceNormal = -faceNormal;
		}

		fnbo[index] = faceNormal.x;
		fnbo[index + 1] = faceNormal.y;
		fnbo[index + 2] = faceNormal.z;
	}
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, triangle *originalPrimitives, fragment* depthbuffer, glm::vec2 resolution, glm::vec3 worldEye)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < primitivesCount)
  {
	  triangle tri = primitives[index];
	  triangle originalTriangle = originalPrimitives[index];
	  glm::vec2 s0, s1, s2;
	  s0.x = (tri.p0.x + 1.0f) * resolution.x / 2.0f;
	  s0.y = (tri.p0.y + 1.0f) * resolution.y / 2.0f;
	  s1.x = (tri.p1.x + 1.0f) * resolution.x / 2.0f;
	  s1.y = (tri.p1.y + 1.0f) * resolution.y / 2.0f;
	  s2.x = (tri.p2.x + 1.0f) * resolution.x / 2.0f;
	  s2.y = (tri.p2.y + 1.0f) * resolution.y / 2.0f;

	  //Bringing to screen space from NDC
	  tri.p0 = glm::vec3(s0, 0.0f);
	  tri.p1 = glm::vec3(s1, 0.0f);
	  tri.p2 = glm::vec3(s2, 0.0f);

	  //Finding if 2 vertices have same Y co-ordinate
	  float yDanger = -10000.0f;
	  float dangerX0, dangerX1;
	  if(s0.y >= s1.y - 0.001f && s0.y <= s1.y + 0.001f)
	  {
		  yDanger = s0.y;
		  dangerX0 = s0.x;
		  dangerX1 = s1.x;
	  }
	  else if(s1.y >= s2.y - 0.001f && s1.y <= s2.y + 0.001f)
	  {
		  yDanger = s1.y;
		  dangerX0 = s1.x;
		  dangerX1 = s2.x;
	  }
	  else if(s2.y >= s0.y - 0.001f && s2.y <= s0.y + 0.001f)
	  {
		  yDanger = s2.y;
		  dangerX0 = s0.x;
		  dangerX1 = s2.x;
	  }


	  float minY = min(s0.y, min(s1.y, s2.y));
	  float maxY = max(s0.y, max(s1.y, s2.y));

	  //Converting some values to int by taking floor (implicit, since x and y are always positive in screen space)
	  int iMaxY = (int)(maxY);

	  //There are indeed vertices with same Y co-ordinates
	  if(yDanger > 0.0f)
	  {
		  int iYDanger = (int)(yDanger);
		  for(int y = (int)(minY); y <= iMaxY; ++y)
		  {
			  if(y == iYDanger)
			  {
				  int x = (int)min(dangerX0, dangerX1);
				  int maxDanger = (int)max(dangerX0, dangerX1);
				  for(; x <= maxDanger; ++x)
				  {
					  //printf("%s\t%d\t%d\n", "Danger Y", x, y);
					  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
				  }
			  }
			  else
			  {
				  float xIntersection0 = -100000.0f, xIntersection1 = -100000.0f, xIntersection2 = -100.0f;
				  if(y >= min(s0.y, s1.y) && y <= max(s0.y, s1.y))
				  {
					  xIntersection2 = ((s0.x - s1.x) / (s0.y - s1.y)) * ((float)y - s1.y) + s1.x;
				  }
				  if(y >= min(s1.y, s2.y) && y <= max(s1.y, s2.y))
				  {
					  xIntersection0 = ((s1.x - s2.x) / (s1.y - s2.y)) * ((float)y - s2.y) + s2.x;
				  }
				  if(y >= min(s0.y, s2.y) && y <= max(s0.y, s2.y))
				  {
					  xIntersection1 = ((s0.x - s2.x) / (s0.y - s2.y)) * ((float)y - s2.y) + s2.x;
				  }
				  if(xIntersection0 < 0.0f && xIntersection1 < 0.0f && xIntersection2 < 0.0f)
				  {
				  	  //Do Nothing, No Intersection
				  } 
				  else if(xIntersection0 < 0.0f && xIntersection1 < 0.0f)
				  {
					  //Intersection with just one point i.e. one vertex
					  //printf("%s\t%d\t%d\n", "Just one Point 2", (int)xIntersection2, y);
					  modifyFragmentAfterDepthTest((int)xIntersection2, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
				  }
				  else if(xIntersection1 < 0.0f && xIntersection2 < 0.0f)
				  {
					  //printf("%s\t%d\t%d\n", "Just one Point 0", (int)xIntersection0, y);
					  modifyFragmentAfterDepthTest((int)xIntersection0, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
				  }
				  else if(xIntersection0 < 0.0f && xIntersection2 < 0.0f)
				  {
					  //printf("%s\t%d\t%d\n", "Just one Point 1", (int)xIntersection1, y);
					  modifyFragmentAfterDepthTest((int)xIntersection1, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
				  }
				  else if(xIntersection0 < 0.0f)
				  {
					  int x = (int)min(xIntersection1, xIntersection2);
					  int maxIntersection = (int)max(xIntersection1, xIntersection2);
					  for(; x <= maxIntersection; ++x)
					  {
						  //printf("%s\t%d\t%d\n", "12 Vertices", x, y);
						  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
					  }
				  }
				  else if(xIntersection1 < 0.0f)
				  {
					  int x = (int)min(xIntersection0, xIntersection2);
					  int maxIntersection = (int)max(xIntersection0, xIntersection2);
					  for(; x <= maxIntersection; ++x)
					  {
						  //printf("%s\t%d\t%d\n", "02 Vertices", x, y);
						  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
					  }
				  }
				  else if(xIntersection2 < 0.0f)
				  {
					  int x = (int)min(xIntersection1, xIntersection0);
					  int maxIntersection = (int)max(xIntersection1, xIntersection0);
					  for(; x <= maxIntersection; ++x)
					  {
						  //printf("%s\t%d\t%d\n", "10 Vertices", x, y);
						  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
					  }
				  }
			  }
		  }
	  }
	  //There are no vertices with same Y co-ordinates
	  else
	  {
		    for(int y = (int)(minY); y <= iMaxY; ++y)
		    {
				float xIntersection0 = -10000.0f, xIntersection1 = -100.0f, xIntersection2 = -100.0f;
				if(y >= min(s0.y, s1.y) && y <= max(s0.y, s1.y))
				{
					xIntersection2 = ((s0.x - s1.x) / (s0.y - s1.y)) * ((float)y - s1.y) + s1.x;
				}
				if(y >= min(s1.y, s2.y) && y <= max(s1.y, s2.y))
				{
					xIntersection0 = ((s1.x - s2.x) / (s1.y - s2.y)) * ((float)y - s2.y) + s2.x;
				}
				if(y >= min(s0.y, s2.y) && y <= max(s0.y, s2.y))
				{
					xIntersection1 = ((s0.x - s2.x) / (s0.y - s2.y)) * ((float)y - s2.y) + s2.x;
				}
				if(xIntersection0 < 0.0f && xIntersection1 < 0.0f && xIntersection2 < 0.0f)
				{
					//Do Nothing, No Intersection
				}
				else if(xIntersection0 < 0.0f && xIntersection1 < 0.0f)
				{
					//Intersection with just one point i.e. one vertex
					//printf("%s\t%d\t%d\n", "Just one Point 2", (int)xIntersection2, y);
					modifyFragmentAfterDepthTest((int)xIntersection2, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
				}
				else if(xIntersection1 < 0.0f && xIntersection2 < 0.0f)
				{
					//printf("%s\t%d\t%d\n", "Just one Point 0", (int)xIntersection0, y);
					modifyFragmentAfterDepthTest((int)xIntersection0, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
				}
				else if(xIntersection0 < 0.0f && xIntersection2 < 0.0f)
				{
					//printf("%s\t%d\t%d\n", "Just one Point 1", (int)xIntersection1, y);
					modifyFragmentAfterDepthTest((int)xIntersection1, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
				}
				else if(xIntersection0 < 0.0f)
				{
					int x = (int)min(xIntersection1, xIntersection2);
					int maxIntersection = (int)max(xIntersection1, xIntersection2);
					for(; x <= maxIntersection; ++x)
					{
						//printf("%s\t%d\t%d\n", "12 Vertices", x, y);
						modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
					}
				}
				else if(xIntersection1 < 0.0f)
				{
					int x = (int)min(xIntersection0, xIntersection2);
					int maxIntersection = (int)max(xIntersection0, xIntersection2);
					for(; x <= maxIntersection; ++x)
					{
						//printf("%s\t%d\t%d\n", "02 Vertices", x, y);
						modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
					}
				}
				else if(xIntersection2 < 0.0f)
				{
					int x = (int)min(xIntersection1, xIntersection0);
					int maxIntersection = (int)max(xIntersection1, xIntersection0);
					for(; x <= maxIntersection; ++x)
					{
						//printf("%s\t%d\t%d\n", "10 Vertices", x, y);
						modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer);
					}
				}
			}
	   }
   }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec3 *framebuffer, glm::vec2 resolution, Light *lights, unsigned int numberOfLights)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y)
  {
	  fragment f = getFromDepthbuffer(x, y, depthbuffer, resolution);
	  //Computing color due to lambert shading and specular highlight
	  glm::vec3 col(0, 0, 0);
	  for(unsigned int i = 0; i < numberOfLights; ++i)
	  {
		  glm::vec3 lightVec = lights[i].pos - f.worldPosition;
		  if(glm::length(lightVec) > 0.001f)
		  {
			  lightVec = glm::normalize(lightVec);
		  }
		  col += glm::dot(f.normal, lightVec) * f.color * lights[i].col;
	  }

	  float KA = 0.1f;	//Ambient Light coefficient
	  f.color = col + KA * f.color;
	  writeToDepthbuffer(x, y, f, depthbuffer, resolution);

	  /*if(f.normal.x > 0 || f.normal.y > 0 || f.normal.z > 0)
	  {
		  printf("%s\t%f\t%f\t%f\n", "Fragment normals", f.normal.x, f.normal.y, f.normal.z);
	  }*/
	  /*if(f.color.x > 0 || f.color.y > 0 || f.color.z > 0)
	  {
		  printf("%s\t%f\t%f\t%f\n", "Fragment colors", f.color.x, f.color.y, f.color.z);
	  }*/
	  /*if(col.x > 0 || col.y > 0 || col.z > 0)
	  {
		  printf("%s\t%f\t%f\t%f\n", "Computed colors", col.x, col.y, col.z);
	  }*/
	  /*if(lightVec.x > 0 || lightVec.y > 0 || lightVec.z > 0)
	  {
		  printf("%s\t%f\t%f\t%f\n", "Light Vector", lightVec.x, lightVec.y, lightVec.z);
	  }*/
	  /*if(lights[0].pos.x > 0 || lights[0].pos.y > 0 || lights[0].pos.z > 0)
	  {
		  printf("%s\t%f\t%f\t%f\n", "Light Pos", lights[0].pos.x, lights[0].pos.y, lights[0].pos.z);
	  }*/
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //int index = (resolution.x - x) + ((resolution.y - y) * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float *nbo, int nbosize,
	cudaMat4 modelViewProjection, glm::vec3 eye, Light *lights, unsigned int numberOfLights)
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
  //frag.position = glm::vec3(0,0,-10000);
  frag.worldDepth = 10000.0f;
  frag.screenSpaceX = 0;
  frag.screenSpaceY = 0;
  frag.lockVariable = 0;
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

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  /*device_fnbo = NULL;
  cudaMalloc((void**)&device_fnbo, ibosize * sizeof(float));*/

  tileSize = 32;
  int primitiveBlocks;

  /*device_originalDepth = NULL;
  cudaMalloc((void**)&device_originalDepth, ibosize * sizeof(float));*/

  originalPrimitives = NULL;
  cudaMalloc((void**)&originalPrimitives, (ibosize / 3) * sizeof(triangle));

  cudaLights = NULL;
  cudaMalloc((void**)&cudaLights, numberOfLights * sizeof(Light));
  cudaMemcpy( cudaLights, lights, numberOfLights*sizeof(Light), cudaMemcpyHostToDevice);

  ////------------------------------
  ////Computing original depth in world
  ////------------------------------
  //originalDepthComputingKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_originalDepth, eye);
  //cudaDeviceSynchronize();

  //------------------------------
  //Original primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, originalPrimitives);
  cudaDeviceSynchronize();

  //------------------------------
  //vertex shader
  //------------------------------
  primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, modelViewProjection);
  cudaDeviceSynchronize();

  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives);
  cudaDeviceSynchronize();

  ////------------------------------
  ////geometry shader
  ////------------------------------
  //geometryShaderKernel<<<primitiveBlocks, tileSize>>>(device_fnbo, primitives, (ibosize / 3));
  //cudaDeviceSynchronize();

  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, originalPrimitives, depthbuffer, resolution, eye);
  cudaDeviceSynchronize();

  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, framebuffer, resolution, cudaLights, numberOfLights);
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

  //ADDED
  cudaFree(originalPrimitives);
  cudaFree(device_nbo);
  //ADDED
}