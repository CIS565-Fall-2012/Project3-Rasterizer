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
float* device_wbo;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 modelMatrix, glm::mat4 ViewMatrix,  glm::mat4 Projection, glm::vec4 ViewPort){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<vbosize/3){
		//printf("\n\n------Vertex Shader-------");
		//printf("\nVertex %d = %f\t%f\t%f", index, vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2]);
		glm::vec3 V = glm::vec3(vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2]);
		V = glm::project(V, ViewMatrix * modelMatrix, Projection, ViewPort);
		vbo[index * 3]		=	V.x;
		vbo[index * 3 + 1]	=	V.y;
		vbo[index * 3 + 2]	=	V.z;
		//printf("\nProjection %d = %f\t%f\t%f", index, vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2]);
	}
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float* wbo, glm::mat4 modelMatrix, glm::vec3 CameraPosition){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int primitivesCount = ibosize/3;
	if(index<primitivesCount){
		int iboIndex = 3 * index;
		//printf("\n\n------Primitive Assembly-------");
		//Arrange Normals
		primitives[index].n0 = glm::vec3(nbo[3 * iboIndex], nbo[3 * iboIndex + 1], nbo[3 * iboIndex + 2]);
		primitives[index].n1 = glm::vec3(nbo[3 * (iboIndex + 1)], nbo[3 * (iboIndex + 1) + 1], nbo[3 * (iboIndex + 1) + 2]);
		primitives[index].n2 = glm::vec3(nbo[3 * (iboIndex + 2)], nbo[3 * (iboIndex + 2) + 1], nbo[3 * (iboIndex + 2) + 2]);
		//Multiply Normals by Model Matrix
		primitives[index].n0 = glm::normalize(glm::vec3(modelMatrix * glm::vec4(primitives[index].n0, 0.0)));		
		primitives[index].n1 = glm::normalize(glm::vec3(modelMatrix* glm::vec4(primitives[index].n1, 0.0)));			
		primitives[index].n2 = glm::normalize(glm::vec3(modelMatrix* glm::vec4(primitives[index].n2, 0.0)));

		//Copy Original Vertices
		primitives[index].w0 = glm::vec3(wbo[3 * iboIndex], wbo[3 * iboIndex + 1], wbo[3 * iboIndex + 2]);
		primitives[index].w1 = glm::vec3(wbo[3 * (iboIndex + 1)], wbo[3 * (iboIndex + 1) + 1], wbo[3 * (iboIndex + 1) + 2]);
		primitives[index].w2 = glm::vec3(wbo[3 * (iboIndex + 2)], wbo[3 * (iboIndex + 2) + 1], wbo[3 * (iboIndex + 2) + 2]);

		//Multiply Vertices by Model Matrix
		primitives[index].w0 = glm::vec3(modelMatrix * glm::vec4(primitives[index].w0, 1.0f));		//Point in world space
		primitives[index].w1 = glm::vec3(modelMatrix * glm::vec4(primitives[index].w1, 1.0f));		//Point in world space
		primitives[index].w2 = glm::vec3(modelMatrix * glm::vec4(primitives[index].w2, 1.0f));	

		glm::vec3 FaceNormal = glm::vec3(primitives[index].n0 + primitives[index].n1 + primitives[index].n2) / 3.0f;
		glm::vec3 FaceCenter = glm::vec3(primitives[index].w0 + primitives[index].w1 + primitives[index].w2) / 3.0f;
		FaceNormal = glm::normalize(FaceNormal);
		glm::vec3 Direction = glm::normalize(FaceCenter - CameraPosition);
		
		//Check For Back Face Culling
		if(glm::dot(FaceNormal, Direction) > 0)
		{
			primitives[index].isCulled = true;
			return;
		}

		else
		{
			primitives[index].isCulled = false;
		}

		primitives[index].p0 = glm::vec3(vbo[3 * iboIndex], vbo[3 * iboIndex + 1], vbo[3 * iboIndex + 2]);
		primitives[index].p1 = glm::vec3(vbo[3 * (iboIndex + 1)], vbo[3 * (iboIndex + 1) + 1], vbo[3 * (iboIndex + 1) + 2]);
		primitives[index].p2 = glm::vec3(vbo[3 * (iboIndex + 2)], vbo[3 * (iboIndex + 2) + 1], vbo[3 * (iboIndex + 2) + 2]);
		

		//Vertex Color
		if(cbosize == 9)
		{
			primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
			primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
			primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
		}
		else if(cbosize == ibosize * 3)
		{
			primitives[index].c0 = glm::vec3(cbo[3 * iboIndex], cbo[3 * iboIndex + 1], cbo[3 * iboIndex + 2]);
			primitives[index].c1 = glm::vec3(cbo[3 * (iboIndex + 1)], cbo[3 * (iboIndex + 1) + 1], cbo[3 * (iboIndex + 1) + 2]);
			primitives[index].c2 = glm::vec3(cbo[3 * (iboIndex + 2)], cbo[3 * (iboIndex + 2) + 1], cbo[3 * (iboIndex + 2) + 2]);
		}
	  
		//Print Vertices
		//printf("\nPrimitive %d.A = %f\t%f\t%f", index, primitives[index].p0.x, primitives[index].p0.y, primitives[index].p0.z);
		//printf("\nPrimitive %d.B = %f\t%f\t%f", index, primitives[index].p1.x, primitives[index].p1.y, primitives[index].p1.z);
		//printf("\nPrimitive %d.C = %f\t%f\t%f", index, primitives[index].p2.x, primitives[index].p2.y, primitives[index].p2.z);

		//printf("\nNormal %d.A = %f\t%f\t%f", index, primitives[index].n0.x, primitives[index].n0.y, primitives[index].n0.z);
		//printf("\nNormal %d.B = %f\t%f\t%f", index, primitives[index].n1.x, primitives[index].n1.y, primitives[index].n1.z);
		//printf("\nNormal %d.C = %f\t%f\t%f", index, primitives[index].n2.x, primitives[index].n2.y, primitives[index].n2.z);

		//printf("\nPrimitive Color %d.A = %f\t%f\t%f", index, primitives[index].c0.x, primitives[index].c0.y, primitives[index].c0.z);
		//printf("\nPrimitive Color %d.B = %f\t%f\t%f", index, primitives[index].c1.x, primitives[index].c1.y, primitives[index].c1.z);
		//printf("\nPrimitive Color %d.C = %f\t%f\t%f", index, primitives[index].c2.x, primitives[index].c2.y, primitives[index].c2.z);
	}
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::mat4 modelMatrix, glm::mat4 ViewMatrix,  glm::mat4 Projection, glm::vec4 ViewPort, glm::vec3 CameraPosition){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		if(primitives[index].isCulled)
			return;
		//printf("\n\n------Rasterization-------");
		glm::vec3 minPoint(0.0, 0.0, 0.0);
		glm::vec3 maxPoint(0.0, 0.0, 0.0);
		getAABBForTriangle(primitives[index], minPoint, maxPoint);
		
		glm::vec3 WPoint0 = primitives[index].w0;
		glm::vec3 WPoint1 = primitives[index].w1;
		glm::vec3 WPoint2 = primitives[index].w2;

		glm::vec3 NPoint0 = primitives[index].n0;
		glm::vec3 NPoint1 = primitives[index].n1;
		glm::vec3 NPoint2 = primitives[index].n2;

		//Calculate Points after Model View Transformation
		//glm::vec3 WPoint0 = glm::vec3(modelMatrix * glm::vec4(primitives[index].w0, 1.0f));		//Point in world space
		//glm::vec3 WPoint1 = glm::vec3(modelMatrix * glm::vec4(primitives[index].w1, 1.0f));			//Point in world space
		//glm::vec3 WPoint2 = glm::vec3(modelMatrix * glm::vec4(primitives[index].w2, 1.0f));		

		//Calculate the Original Points -> Even before tansformations are applied
		//glm::vec3 OPoint0 = glm::unProject(primitives[index].p0, ViewMatrix * modelMatrix, Projection, ViewPort);		//Point in world space
		//glm::vec3 OPoint1 = glm::unProject(primitives[index].p1, ViewMatrix * modelMatrix, Projection, ViewPort);		//Point in world space
		//glm::vec3 OPoint2 = glm::unProject(primitives[index].p2, ViewMatrix * modelMatrix, Projection, ViewPort);		//Point in world space

		//printf("\nNormal Before %d.A = %f\t%f\t%f", index, primitives[index].n0.x, primitives[index].n0.y, primitives[index].n0.z);
		//printf("\nNormal Before %d.B = %f\t%f\t%f", index, primitives[index].n1.x, primitives[index].n1.y, primitives[index].n1.z);
		//printf("\nNormal Before %d.C = %f\t%f\t%f", index, primitives[index].n2.x, primitives[index].n2.y, primitives[index].n2.z);

		//printf("\nNormal After %d.A = %f\t%f\t%f", index, NPoint0.x, NPoint0.y, NPoint0.z);
		//printf("\nNormal After %d.B = %f\t%f\t%f", index, NPoint1.x, NPoint1.y, NPoint1.z);
		//printf("\nNormal After %d.C = %f\t%f\t%f", index, NPoint2.x, NPoint2.y, NPoint2.z);

		//printf("\nMaxPoint %d = %f\t%f\t%f", index, maxPoint.x, maxPoint.y, maxPoint.z);
		//printf("\nMinPoint %d = %f\t%f\t%f", index, minPoint.x, minPoint.y, minPoint.z);

		//Calculate the ModelView Normals

		//Normals in world space
		//glm::vec3 NPoint0 = glm::normalize(glm::vec3(modelMatrix * glm::vec4(primitives[index].n0, 0.0)));		
		//glm::vec3 NPoint1 = glm::normalize(glm::vec3(modelMatrix* glm::vec4(primitives[index].n1, 0.0)));			
		//glm::vec3 NPoint2 = glm::normalize(glm::vec3(modelMatrix* glm::vec4(primitives[index].n2, 0.0)));
		//Above-> Normals in World Space


		glm::vec3 CPoints[4] = {primitives[index].p0, primitives[index].p1, primitives[index].p2, primitives[index].p0};
		
		for(int j = minPoint.y; j <= maxPoint.y; j++)
		{
			glm::vec3 FirstPoint = glm::vec3(10000, j, 0);	//Setting it inverted because we need space of scanline area covered
			glm::vec3 LastPoint = glm::vec3(-10000, j, 0);
			bool inter = true;
			float t = -1.0f;
			for(int k = 0; k < 3 && inter; k++)
			{
				glm::vec3 StartPoint;
				glm::vec3 EndPoint;
				
				StartPoint = CPoints[k];
				EndPoint = CPoints[k+1];
				
				glm::vec3 LineUnit(0.0);
				float LineLength = glm::length(EndPoint - StartPoint);
				if(LineLength > 0.001)
					LineUnit = glm::normalize(EndPoint - StartPoint);
				t = -1.0f;

				//If the following condition is true, then we cannot divide by LineUnit.y in the else part ->Divide by 0 error
				if(LineUnit.y < 0 + 0.0000001 && LineUnit.y > 0 - 0.0000001)
				{
					if(StartPoint.x < EndPoint.x)
					{
						FirstPoint = StartPoint;
						LastPoint = EndPoint;
					}
					else
					{
						FirstPoint = EndPoint;
						LastPoint = StartPoint;
					}
					inter = false;
				}
				else
				{
					t = (j - StartPoint.y) / LineUnit.y;
					glm::vec3 IntersectionPoint;
					
					if(t >=0 && t <= LineLength)
					{
						IntersectionPoint = StartPoint + LineUnit * t;
						if(IntersectionPoint.x < FirstPoint.x)
							FirstPoint = IntersectionPoint;
						if(IntersectionPoint.x > LastPoint.x)
								LastPoint = IntersectionPoint;
					}
				}
			}
			//printf("\nFirstPoint %d = %f\t%f\t%f", index, FirstPoint.x, FirstPoint.y, FirstPoint.z);
			//printf("\nLastPoint %d = %f\t%f\t%f", index, LastPoint.x, LastPoint.y, LastPoint.z);
			if(FirstPoint.x < resolution.x && LastPoint.x > 0)
			{
				if(FirstPoint.x > LastPoint.x)		//Check if first point is greater than the last point
				{
					glm::vec3 temp = LastPoint;
					LastPoint = FirstPoint;
					FirstPoint = temp;
				}
				
				float ScanlineLength = glm::length(LastPoint - FirstPoint);
				glm::vec3 ScanlineUnit(0.0, 0.0, 0.0);
				if(ScanlineLength > 0.001f)
					ScanlineUnit = glm::normalize(LastPoint - FirstPoint);
			
				float t = 0;
				
				for(int i = 0; i <= ScanlineLength; i++)
				{
					glm::vec3 SPoint = FirstPoint + (float)i * ScanlineUnit;
					
					int xpix = resolution.x - SPoint.x;//Point in screen space 0,0,800,800
					int ypix = resolution.y - SPoint.y;
					//glm::vec3 WPoint = glm::unProject(SPoint, ViewMatrix * modelMatrix, Projection, ViewPort);		//Point in world space
					glm::vec3 BPoint = calculateBarycentricCoordinate(primitives[index], glm::vec2(SPoint.x, SPoint.y));
					//printf("\nBarycentric Point %d = %f\t%f\t%f", index, BPoint.x, BPoint.y, BPoint.z);
			
					int bufIndex = ypix * resolution.x + xpix;
					
					fragment fragXY;
					fragXY.color = BPoint.x * primitives[index].c0 + BPoint.y * primitives[index].c1 + BPoint.z * primitives[index].c2;
					fragXY.normal = glm::normalize(BPoint.x * NPoint0 + BPoint.y * NPoint1 + BPoint.z * NPoint2);
					//fragXY.orig_position = glm::vec3(modelMatrix * glm::vec4(WPoint, 1.0));
					
					fragXY.orig_position = BPoint.x * WPoint0 + BPoint.y * WPoint1 + BPoint.z * WPoint2;

					fragXY.position = SPoint;
					fragXY.Lock = 1;
										
					//Atomic Compare and swap
					bool dontLeaveLoop = true;
					fragXY.distance = glm::length(fragXY.orig_position - CameraPosition);
									
					while(dontLeaveLoop)
					{
						float BufferDistanceFromEye = glm::length(depthbuffer[bufIndex].orig_position - CameraPosition);
						if(fragXY.distance < BufferDistanceFromEye)
						{	
							if(atomicExch(&(depthbuffer[bufIndex].Lock), 1) == 0)
							{
								depthbuffer[bufIndex] = fragXY;
								dontLeaveLoop = false;
								atomicExch(&(depthbuffer[bufIndex].Lock), 0);
							}
						}
						else
						{
							dontLeaveLoop = false;
						}
					}
				}
			}
		}
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 Camera, glm::vec3 LightPosition, glm::vec3 LightColor, glm::vec3 AmbientColor, float specularCoefficient, bool UseDiffuseShade, bool UseSpecularShade, bool UseAmbientShade, bool UseDepthShade){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		if(depthbuffer[index].distance > -9000)
		{
			if(UseDepthShade)
			{
				glm::vec3 out_Color = 300.0f * glm::vec3(1.0, 1.0, 1.0) / (depthbuffer[index].distance * depthbuffer[index].distance);
				glm::clamp(out_Color, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
				depthbuffer[index].color = out_Color;
				return;
			}
			//glm::vec3 LightPosition = glm::vec3(0.0, 10.0, 10.0);
			//glm::vec3 LightColor	= glm::vec3(1.0, 1.0, 1.0);
			//glm::vec3 AmbientColor	= glm::vec3(0.2, 0.2, 0.2);

			glm::vec3 Incident = glm::normalize(Camera - depthbuffer[index].orig_position);
			glm::vec3 normal = glm::normalize(depthbuffer[index].normal);
			glm::vec3 Reflected = glm::normalize(Incident - 2.0f * normal * glm::dot(Incident, normal));//Ri - 2 * N *(Ri . N)

			//calculate diffuse term and clamp to the range [0, 1]
			float diffuseTerm = clamp(glm::dot(normal, glm::normalize(LightPosition - depthbuffer[index].orig_position)), 0.0, 1.0);
			float specularTerm = clamp(glm::dot(Reflected, Incident), 0.0, 1.0);
			
			if(diffuseTerm == 0)
				specularTerm = 0;
			
			glm::vec3 out_Color = glm::vec3(0.0);
			
			
			glm::vec3 Color = depthbuffer[index].color;
			if(UseDiffuseShade)
				out_Color += Color * LightColor * diffuseTerm;
			if(UseAmbientShade)
				out_Color += Color * AmbientColor;
			if(UseSpecularShade)
				out_Color += Color * LightColor * pow(specularTerm, specularCoefficient);
			
			glm::clamp(out_Color, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
			depthbuffer[index].color = out_Color;
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
	framebuffer[index] = depthbuffer[index].color;
	}
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, glm::mat4 modelMatrix, glm::mat4 ViewMatrix, glm::mat4 Projection, glm::vec4 ViewPort, glm::vec3 CameraPosition, glm::vec3 LightPosition, glm::vec3 LightColor, glm::vec3 AmbientColor, float specularCoefficient)
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
	frag.position = glm::vec3(0,0,-10000);
	frag.orig_position = glm::vec3(0, 0, -10000);
	frag.distance = -10000;
	frag.Lock = 0;
	clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);
  
	//------------------------------
	//Set Up Camera Projection Matrix
	//------------------------------
	
	//cudaMat4 cudaModelMatrix	= utilityCore::glmMat4ToCudaMat4(modelMatrix);
	//cudaMat4 cudaViewMatrix			= utilityCore::glmMat4ToCudaMat4(ViewMatrix);
	//cudaMat4 cudaProjection			= utilityCore::glmMat4ToCudaMat4(Projection);
	//glm::vec4 ViewPort remains same

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

	device_wbo = NULL;
	cudaMalloc((void**)&device_wbo, vbosize*sizeof(float));
	cudaMemcpy( device_wbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

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
	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, modelMatrix, ViewMatrix, Projection, ViewPort);

	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, device_wbo, modelMatrix, CameraPosition);
	
	cudaDeviceSynchronize();
	//------------------------------
	//rasterization
	//------------------------------
	rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, modelMatrix, ViewMatrix, Projection, ViewPort, CameraPosition);
	
	cudaDeviceSynchronize();
	//------------------------------
	//fragment shader
	//------------------------------
	if(UseFragmentShader)
	{
		fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, CameraPosition, LightPosition, LightColor, AmbientColor, specularCoefficient, UseDiffuseShade, UseSpecularShade, UseAmbientShade, UseDepthShade);

		cudaDeviceSynchronize();
	}
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
  cudaFree( device_wbo );
  cudaFree( device_nbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}