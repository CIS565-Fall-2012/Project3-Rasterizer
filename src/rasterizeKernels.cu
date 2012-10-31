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
float* device_vbo_orig;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, cudaMat4 cudaProjection){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
		//printf("\nVertex %d = %f\t%f\t%f", index / 3, vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2]);
		glm::vec3 V(vbo[index * 3], vbo[index * 3 + 1], vbo[index * 3 + 2]);
		V = multiplyMVPoint(cudaProjection, glm::vec4(V, 1));
		vbo[index * 3]		=	V.x;
		vbo[index * 3 + 1]	=	V.y;
		vbo[index * 3 + 2]	=	V.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float* device_vbo_orig){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int iboIndex = 3 * index;
	  //Transformed Vertices
	  primitives[index].p0 = glm::vec3(vbo[3 * iboIndex], vbo[3 * iboIndex + 1], vbo[3 * iboIndex + 2]);
	  primitives[index].p1 = glm::vec3(vbo[3 * (iboIndex + 1)], vbo[3 * (iboIndex + 1) + 1], vbo[3 * (iboIndex + 1) + 2]);
	  primitives[index].p2 = glm::vec3(vbo[3 * (iboIndex + 2)], vbo[3 * (iboIndex + 2) + 1], vbo[3 * (iboIndex + 2) + 2]);
	  //Original Vertices
	  primitives[index].orig_p0 = glm::vec3(device_vbo_orig[3 * iboIndex], device_vbo_orig[3 * iboIndex + 1], device_vbo_orig[3 * iboIndex + 2]);
	  primitives[index].orig_p1 = glm::vec3(device_vbo_orig[3 * (iboIndex + 1)], device_vbo_orig[3 * (iboIndex + 1) + 1], device_vbo_orig[3 * (iboIndex + 1) + 2]);
	  primitives[index].orig_p2 = glm::vec3(device_vbo_orig[3 * (iboIndex + 2)], device_vbo_orig[3 * (iboIndex + 2) + 1], device_vbo_orig[3 * (iboIndex + 2) + 2]);
	  //Vertex Color
	  primitives[index].c0 = glm::vec3(cbo[3 * iboIndex], cbo[3 * iboIndex + 1], cbo[3 * iboIndex + 2]);
	  primitives[index].c1 = glm::vec3(cbo[3 * (iboIndex + 1)], cbo[3 * (iboIndex + 1) + 1], cbo[3 * (iboIndex + 1) + 2]);
	  primitives[index].c2 = glm::vec3(cbo[3 * (iboIndex + 2)], cbo[3 * (iboIndex + 2) + 1], cbo[3 * (iboIndex + 2) + 2]);
	  //if(index <= 100)
	  //Print Vertices
	  printf("\nPrimitive %d.A = %f\t%f\t%f", index, primitives[index].p0.x, primitives[index].p0.y, primitives[index].p0.z);
	  printf("\nPrimitive %d.B = %f\t%f\t%f", index, primitives[index].p1.x, primitives[index].p1.y, primitives[index].p1.z);
	  printf("\nPrimitive %d.C = %f\t%f\t%f", index, primitives[index].p2.x, primitives[index].p2.y, primitives[index].p2.z);

	  printf("\nOrig Primitive %d.A = %f\t%f\t%f", index, primitives[index].orig_p0.x, primitives[index].orig_p0.y, primitives[index].orig_p0.z);
	  printf("\nOrig Primitive %d.B = %f\t%f\t%f", index, primitives[index].orig_p1.x, primitives[index].orig_p1.y, primitives[index].orig_p1.z);
	  printf("\nOrig Primitive %d.C = %f\t%f\t%f", index, primitives[index].orig_p2.x, primitives[index].orig_p2.y, primitives[index].orig_p2.z);
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount){
		
		glm::vec3 minPoint(0.0, 0.0, 0.0);
		glm::vec3 maxPoint(0.0, 0.0, 0.0);
		getAABBForTriangle(primitives[index], minPoint, maxPoint);

		glm::vec3 CPoints[4] = {primitives[index].p0, primitives[index].p1, primitives[index].p2, primitives[index].p0};
		float incY = 1.0f / resolution.y;
		for(float j = minPoint.y - 1; j < maxPoint.y + 1; j+=incY)
		{
			glm::vec3 FirstPoint = glm::vec3(-1, j, 0);	//Setting it inverted because we need space of scanline area covered
			glm::vec3 LastPoint = glm::vec3(2, j, 0);

			for(int k = 0; k < 3; k++)
			{
				glm::vec3 StartPoint = CPoints[k];
				glm::vec3 EndPoint = CPoints[k+1];
				glm::vec3 LineUnit = glm::normalize(EndPoint - StartPoint);
				
				if((LineUnit.x != 0 && LineUnit.z != 0) || (LineUnit.y == j))			//Check for Parallel Lines
				{
					float LineLength = glm::length(EndPoint - StartPoint);
					float t = (j - StartPoint.y) / LineUnit.y;
					glm::vec3 IntersectionPoint;
					
					if(t >=0 && t <= LineLength)
						IntersectionPoint = StartPoint + LineUnit * t;
					if(IntersectionPoint.x < FirstPoint.x)
						FirstPoint = IntersectionPoint;
					if(IntersectionPoint.x > LastPoint.x)
						LastPoint = IntersectionPoint;
				}
			}
			glm::vec3 ScanlineUnit = glm::normalize(LastPoint - FirstPoint);
			float ScanlineLenght = glm::length(LastPoint - FirstPoint);
			float incX = 1.0f / resolution.x;
			int n = ScanlineLength / incX;
			float t = 0;
			for(float i = FirstPoint.x; i <= LastPoint.x; i+=incX, t+=n)
			{
				int x = i * resolution.x;		//pixel values
				int y = j * resolution.y;

				index = y * resolution.x + x;
				glm::vec3 Point = FirstPoint + t * ScanlineUnit;

				//Atomic Compare and swap
			}
		}
		
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 Camera){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		if(depthbuffer[index].position.z > 0)
		{
			glm::vec3 LightPosition = glm::vec3(3.0, 3.0, -3.0);
			glm::vec3 LightColor	= glm::vec3(1.0, 1.0, 1.0);
			glm::vec3 AmbientColor	= glm::vec3(0.1, 0.1, 0.1);
			glm::vec3 Incident = Camera - depthbuffer[index].orig_position;
			glm::vec3 Reflected = Incident - 2.0f * glm::normalize(depthbuffer[index].normal) * (glm::dot(glm::normalize(Incident), glm::normalize(depthbuffer[index].normal)));//Ri – 2 N (Ri • N)
			
			//calculate diffuse term and clamp to the range [0, 1]
			float diffuseTerm = clamp(glm::dot(glm::normalize(depthbuffer[index].normal), glm::normalize(LightPosition - depthbuffer[index].orig_position)), 0.0, 1.0);
			float specularTerm = clamp(glm::dot(glm::normalize(Reflected), glm::normalize(Incident)), 0.0, 1.0);
			if(diffuseTerm == 0)
				specularTerm = 0;
			glm::vec3 out_Color = depthbuffer[index].color * LightColor;
			out_Color = out_Color * diffuseTerm 
						+ out_Color * AmbientColor 
						+ out_Color * pow(specularTerm, 30.0f);
			glm::clamp(out_Color, glm::vec3(0.0f), glm::vec3(1.0f));
			depthbuffer[index].color = out_Color;
		}
		else
		{
			depthbuffer[index].color = glm::vec3(1,1,0);
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize){

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
	clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);
  
	//------------------------------
	//Set Up Camera Projection Matrix
	//------------------------------
	glm::vec3 CameraPosition = glm::vec3(0.0f, 0.5f, 5.0f);
	int width = resolution.x;
	int height = resolution.y;
	
	glm::mat4 projection = glm::perspective(30.0f, static_cast<float>(width) / static_cast<float>(height), 0.1f, 50.0f);
	glm::mat4 camera = glm::lookAt(CameraPosition, glm::vec3(0.0, 0.5, 0), glm::vec3(0, 1, 0));
	projection = projection * camera;
	cudaMat4 cudaProjection = utilityCore::glmMat4ToCudaMat4(projection);
	glm::mat4 invProjection = glm::inverse(projection);
	cudaMat4 cudaInvProjection = utilityCore::glmMat4ToCudaMat4(invProjection);

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

	device_vbo_orig = NULL;
	cudaMalloc((void**)&device_vbo_orig, vbosize*sizeof(float));
	cudaMemcpy( device_vbo_orig, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

	device_cbo = NULL;
	cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
	cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

	tileSize = 32;
	int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

	//------------------------------
	//vertex shader
	//------------------------------
	vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, cudaProjection);

	cudaDeviceSynchronize();
	//------------------------------
	//primitive assembly
	//------------------------------
	primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
	primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, device_vbo_orig);

	cudaDeviceSynchronize();
	//------------------------------
	//rasterization
	//------------------------------
	rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

	cudaDeviceSynchronize();
	//------------------------------
	//fragment shader
	//------------------------------
	fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, CameraPosition);

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
  cudaFree( device_vbo_orig);
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}
