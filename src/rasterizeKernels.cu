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
//ADDED
float *device_nbo;
float *device_vto;
triangle *originalPrimitives;
Light *cudaLights;
unsigned char *cudaTextureImage;
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

__device__ void modifyFragmentAfterDepthTest(const int &screenSpaceX, const int &screenSpaceY, triangle &screenSpaceTriangle, const triangle &originalTriangle,
	const glm::vec3 &eye, const glm::vec2 &resolution, fragment *depthbuffer,
	unsigned char *textureImage, unsigned int textureWidth, unsigned int textureHeight)
{
	glm::vec3 barycentricCoords = calculateBarycentricCoordinate(screenSpaceTriangle, glm::vec2((float)screenSpaceX, (float)screenSpaceY));
	float z = giveWorldSpaceDepth(barycentricCoords, originalTriangle, eye);

	fragment f = getFromDepthbuffer(screenSpaceX, screenSpaceY, depthbuffer, resolution);
	
	if(f.worldDepth > z)
	{
		f.screenSpaceX = screenSpaceX;
		f.screenSpaceY = screenSpaceY;
		f.worldDepth = z;
		f.lockVariable = 0;
		f.worldPosition = getPositionAtBarycentricCoordinate(barycentricCoords, originalTriangle);
		f.normal = getNormalAtBarycentricCoordinate(barycentricCoords, originalTriangle);

		//Normal coloring
		f.color = getColorAtBarycentricCoordinate(barycentricCoords, screenSpaceTriangle);

		//Texturing
		//printf("UV 0 \t%f\t%f\n", originalTriangle.uv0.x , originalTriangle.uv0.y);
		//printf("UV 1 \t%f\t%f\n", originalTriangle.uv1.x , originalTriangle.uv1.y);
		//printf("UV 2 \t%f\t%f\n", originalTriangle.uv2.x , originalTriangle.uv2.y);
		
		//glm::vec2 uv = getTextureAtBarycentricCoordinate(barycentricCoords, originalTriangle);
		
		//int imageCoordX = (int)(uv.x * (float)textureWidth);
		//int imageCoordY = (int)(uv.y * (float)textureHeight);
		//if(imageCoordX < 1023 && imageCoordY < 1023 && imageCoordX >= 0 && imageCoordY >= 0)
		//{
			//f.color = glm::vec3(-100, -100, -100);
			//(float)textureImage[3 * (1023 * 1024 + 1023) - 1];
			//f.color = glm::vec3((float)textureImage[3 * 1023 * 1024 + 1023 - 1], (float)textureImage[3 * 1023 * 1024 + 1023 - 1], (float)textureImage[3 * 1023 * 1024 + 1023 - 1]);
			//f.color.x = (float)textureImage[3 * (imageCoordY * textureWidth + imageCoordX)] / 255.0f;
			//f.color.y = (float)textureImage[3 * (imageCoordY * textureWidth + imageCoordX) + 1] / 255.0f;
			//f.color.z = (float)textureImage[3 * (imageCoordY * textureWidth + imageCoordX) + 2] / 255.0f;
		//}

		/*if(uv.x > 1 || uv.y > 1)
			printf("Image Co-ordinates %d\t%d\t%d\n", uv.x, uv.y, 3 * (imageCoordY * textureWidth + imageCoordX));*/

		/*if(imageCoordX > 1023 || imageCoordY > 1023)
			printf("Image Co-ordinates %d\t%d\t%d\n", imageCoordX, imageCoordY, 3 * (imageCoordY * textureWidth + imageCoordX));*/
		//printf("Hmm %f\n", (float)textureImage[3 * (1023 * 1024 + 1023) - 1]);

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
__global__ void primitiveAssemblyKernel(float* vbo, float* cbo, int* ibo, int ibosize, float *nbo, float *vto, triangle* primitives)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index < primitivesCount)
  {
	  int a = 3 * ibo[3 * index];
	  int b = 3 * ibo[3 * index + 1];
	  int c = 3 * ibo[3 * index + 2];

	  triangle tri;

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


	  tri.p0 = glm::vec3(vbo[a], vbo[a + 1], vbo[a + 2]);
	  tri.p1 = glm::vec3(vbo[b], vbo[b + 1], vbo[b + 2]);
	  tri.p2 = glm::vec3(vbo[c], vbo[c + 1], vbo[c + 2]);

	  //When only 3 colors are given
	  /*if(vbosize == 9)
	  {
		  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
		  primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
		  primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
	  }
	  else
	  {
		  //When more than 3 colors are given
		  primitives[index].c0 = glm::vec3(cbo[a], cbo[a + 1], cbo[a + 2]);
		  primitives[index].c1 = glm::vec3(cbo[b], cbo[b + 1], cbo[b + 2]);
		  primitives[index].c2 = glm::vec3(cbo[c], cbo[c + 1], cbo[c + 2]);
	  }*/

	  tri.c0 = glm::vec3(cbo[a], cbo[a + 1], cbo[a + 2]);
	  tri.c1 = glm::vec3(cbo[b], cbo[b + 1], cbo[b + 2]);
	  tri.c2 = glm::vec3(cbo[c], cbo[c + 1], cbo[c + 2]);

	  tri.n0 = glm::vec3(nbo[a], nbo[a + 1], nbo[a + 2]);
	  tri.n1 = glm::vec3(nbo[b], nbo[b + 1], nbo[b + 2]);
	  tri.n2 = glm::vec3(nbo[c], nbo[c + 1], nbo[c + 2]);

	  tri.uv0 = glm::vec2(vto[a], vto[a + 1]);
	  tri.uv1 = glm::vec2(vto[b], vto[b + 1]);
	  tri.uv2 = glm::vec2(vto[c], vto[c + 1]);

	  primitives[index] = tri;

	  //printf("%s\t%f\t%f\t%f\n", "VTO UV 0", vto[m], vto[m + 1], vto[m + 2]);
	  //printf("%s\t%f\t%f\t%f\n", "VTO UV 1", vto[n], vto[n + 1], vto[n + 2]);
	  //printf("%s\t%f\t%f\t%f\n", "VTO UV 2", vto[l], vto[l + 1], vto[l + 2]);
  }
}

__global__ void geometryShaderKernel(triangle *originalPrimitives, int numberOfPrimitives, glm::vec3 eye)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < numberOfPrimitives)
	{
		triangle tri = originalPrimitives[index];
		glm::vec3 avgNorm = glm::normalize((tri.n0 + tri.n1 + tri.n2) / 3.0f);
		glm::vec3 avgPos = (tri.p0 + tri.p1 + tri.p2) / 3.0f;
		if(glm::dot(avgNorm, glm::normalize(eye - avgPos)) < -0.5f)
		{
			originalPrimitives[index].c0.x = -10000.0f;
		}
	}
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, triangle *originalPrimitives, fragment* depthbuffer, glm::vec2 resolution, glm::vec3 worldEye,
	unsigned char *textureImage, unsigned int textureWidth, unsigned int textureHeight)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < primitivesCount)
  {
	  triangle tri = primitives[index];
	  triangle originalTriangle = originalPrimitives[index];
	  //Back Face Culling
	  if(originalTriangle.c0.x < -9999.0f)
	  {
		  return;
	  }
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
					  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
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
					  modifyFragmentAfterDepthTest((int)xIntersection2, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
				  }
				  else if(xIntersection1 < 0.0f && xIntersection2 < 0.0f)
				  {
					  //printf("%s\t%d\t%d\n", "Just one Point 0", (int)xIntersection0, y);
					  modifyFragmentAfterDepthTest((int)xIntersection0, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
				  }
				  else if(xIntersection0 < 0.0f && xIntersection2 < 0.0f)
				  {
					  //printf("%s\t%d\t%d\n", "Just one Point 1", (int)xIntersection1, y);
					  modifyFragmentAfterDepthTest((int)xIntersection1, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
				  }
				  else if(xIntersection0 < 0.0f)
				  {
					  int x = (int)min(xIntersection1, xIntersection2);
					  int maxIntersection = (int)max(xIntersection1, xIntersection2);
					  for(; x <= maxIntersection; ++x)
					  {
						  //printf("%s\t%d\t%d\n", "12 Vertices", x, y);
						  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
					  }
				  }
				  else if(xIntersection1 < 0.0f)
				  {
					  int x = (int)min(xIntersection0, xIntersection2);
					  int maxIntersection = (int)max(xIntersection0, xIntersection2);
					  for(; x <= maxIntersection; ++x)
					  {
						  //printf("%s\t%d\t%d\n", "02 Vertices", x, y);
						  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
					  }
				  }
				  else if(xIntersection2 < 0.0f)
				  {
					  int x = (int)min(xIntersection1, xIntersection0);
					  int maxIntersection = (int)max(xIntersection1, xIntersection0);
					  for(; x <= maxIntersection; ++x)
					  {
						  //printf("%s\t%d\t%d\n", "10 Vertices", x, y);
						  modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
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
					modifyFragmentAfterDepthTest((int)xIntersection2, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
				}
				else if(xIntersection1 < 0.0f && xIntersection2 < 0.0f)
				{
					//printf("%s\t%d\t%d\n", "Just one Point 0", (int)xIntersection0, y);
					modifyFragmentAfterDepthTest((int)xIntersection0, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
				}
				else if(xIntersection0 < 0.0f && xIntersection2 < 0.0f)
				{
					//printf("%s\t%d\t%d\n", "Just one Point 1", (int)xIntersection1, y);
					modifyFragmentAfterDepthTest((int)xIntersection1, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
				}
				else if(xIntersection0 < 0.0f)
				{
					int x = (int)min(xIntersection1, xIntersection2);
					int maxIntersection = (int)max(xIntersection1, xIntersection2);
					for(; x <= maxIntersection; ++x)
					{
						//printf("%s\t%d\t%d\n", "12 Vertices", x, y);
						modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
					}
				}
				else if(xIntersection1 < 0.0f)
				{
					int x = (int)min(xIntersection0, xIntersection2);
					int maxIntersection = (int)max(xIntersection0, xIntersection2);
					for(; x <= maxIntersection; ++x)
					{
						//printf("%s\t%d\t%d\n", "02 Vertices", x, y);
						modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
					}
				}
				else if(xIntersection2 < 0.0f)
				{
					int x = (int)min(xIntersection1, xIntersection0);
					int maxIntersection = (int)max(xIntersection1, xIntersection0);
					for(; x <= maxIntersection; ++x)
					{
						//printf("%s\t%d\t%d\n", "10 Vertices", x, y);
						modifyFragmentAfterDepthTest(x, y, tri, originalTriangle, worldEye, resolution, depthbuffer, textureImage, textureWidth, textureHeight);
					}
				}
			}
	   }
   }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec3 *framebuffer, glm::vec2 resolution, Light *lights, unsigned int numberOfLights, glm::vec3 eye)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if(x<=resolution.x && y<=resolution.y)
  {
	  fragment f = getFromDepthbuffer(x, y, depthbuffer, resolution);
	  if(!(f.worldDepth >= 99999.999f && f.worldDepth <= 100000.001f))
	  {
		  //Computing color due to lambert shading and specular highlight
		  glm::vec3 col(0, 0, 0);
		  float specularCoefficient = 10;
		  glm::vec3 fragmentToEye = glm::normalize(eye - f.worldPosition);
		  for(unsigned int i = 0; i < numberOfLights; ++i)
		  {
			  glm::vec3 lightVec = lights[i].pos - f.worldPosition;
			  if(glm::length(lightVec) > 0.001f)
			  {
	  			  lightVec = glm::normalize(lightVec);
			  }
			  glm::vec3 reflectedRay = calculateReflectionDirection(f.normal, -lightVec);
			  //Diffuse
			  col += glm::dot(f.normal, lightVec) * f.color * lights[i].col;

			  //Specular
			  if(glm::length(f.color) > 0.01f)
			  {
				  col += pow(glm::dot(fragmentToEye, reflectedRay), specularCoefficient) * lights[i].col;
			  }
		  }

		  float KA = 0.1f;	//Ambient Light coefficient
		  f.color = col + KA * f.color;
		  writeToDepthbuffer(x, y, f, depthbuffer, resolution);
	  }
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
	float *vto, int vtosize, cudaMat4 modelViewProjection, glm::vec3 eye, Light *lights, unsigned int numberOfLights,
	unsigned char *textureImage, unsigned int textureWidth, unsigned int textureHeight)
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
  frag.color = glm::vec3(0.5f, 0.5f, 0.5f);
  frag.normal = glm::vec3(0,0,0);
  frag.worldDepth = 100000.0f;
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

  device_vto = NULL;
  cudaMalloc((void**)&device_vto, vtosize*sizeof(float));
  cudaMemcpy( device_vto, vto, vtosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks;

  originalPrimitives = NULL;
  cudaMalloc((void**)&originalPrimitives, (ibosize / 3) * sizeof(triangle));

  cudaLights = NULL;
  cudaMalloc((void**)&cudaLights, numberOfLights * sizeof(Light));
  cudaMemcpy( cudaLights, lights, numberOfLights*sizeof(Light), cudaMemcpyHostToDevice);

  cudaTextureImage = NULL;
  cudaMalloc((void**)&cudaTextureImage, 3 * textureWidth * textureHeight * sizeof(unsigned char));
  cudaMemcpy( cudaTextureImage, textureImage, 3 * textureWidth * textureHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

  //------------------------------
  //Original primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_cbo, device_ibo, ibosize, device_nbo, device_vto, originalPrimitives);
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
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_cbo, device_ibo, ibosize, device_nbo, device_vto, primitives);
  cudaDeviceSynchronize();

  //------------------------------
  //geometry shader, used for back face culling
  //------------------------------
  geometryShaderKernel<<<primitiveBlocks, tileSize>>>(originalPrimitives, (ibosize / 3), eye);
  cudaDeviceSynchronize();

  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, originalPrimitives, depthbuffer, resolution, eye, textureImage, textureWidth, textureHeight);
  cudaDeviceSynchronize();

  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, framebuffer, resolution, cudaLights, numberOfLights, eye);
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
  cudaFree(cudaLights);
  cudaFree(device_vto);
  cudaFree(cudaTextureImage);
  //ADDED
}