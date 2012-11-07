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
float * device_nbo;
int* device_ibo;
triangle* primitives;
 cudaMat4 projectionMatrix;
 cudaMat4 viewMatrix;
 glm::vec3 lightPosition = glm::vec3(4,10,4);
  blendType blendtype = NONE;

 glm::vec4 scissorWindow;
bool scissortest = false;
bool blending = false;
 void Toggle(test testType)
{
	switch(testType)
	{
	case SCISSOR_TEST:
		scissortest = !scissortest;
		break;
	case BLENDING:
		blending = !blending;
		break;
    
	}
}

blendType ReadBlendType()
{
	return blendtype;
}
void SetBlendType(blendType type)
{
	if(blendtype == ADD && type == ADD)
	{
	   blendtype = NONE;
	   return;
	}
	if(blendtype == NONE && type == ADD)
	{
		blendtype = ADD;
	}
	
}
 void SetScissorWindow(glm::vec4 windowsize)
 {
	 scissorWindow = windowsize;
 }

void setProjectionMatrix(glm::mat4  & trans)
{
	
  projectionMatrix.x = trans[0];
  projectionMatrix.y = trans[1];
  projectionMatrix.z = trans[2];
  projectionMatrix.w = trans[3];
  
}
void setViewMatrix(glm::mat4 & trans)
{
  viewMatrix.x = trans[0];
  viewMatrix.y = trans[1];
  viewMatrix.z = trans[2];
  viewMatrix.w = trans[3];
}

void setUniformLightPosition(glm::vec3 position)
{
	lightPosition = position;
}


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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image,glm::vec4 windowsize, blendType blendtype){
  
    int x = (blockIdx.x * blockDim.x) + threadIdx.x + windowsize.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y + windowsize.y;
  int index = x + (y * resolution.x);
  
  if( x >= windowsize.x && y >= windowsize.y && x <= windowsize.z && y<= windowsize.w){

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

	  uchar4  result= PBOpos[index];
	  switch(blendtype)
	  {
	  case NONE:
		  result.x = color.x;
		  result.y = color.y;
		  result.z = color.z;
		  result.w = 0;
		  break;
	  case ADD:
		 /* result.x = 0.5f * result.x + 0.5f * color.x;
		  result.y = 0.5f * result.y + 0.5f * color.x;
		  result.z = 0.5f * result.z + 0.5f * color.z;
		  result.w = 0;*/
		 /*result.x =  result.x + color.x;
		  result.y =  result.y + color.y;
		  result.z =  result.z + color.z;
		  result.w = 0;*/

		  result.x =  result.x * color.x;
		  result.y =  result.y * color.y;
		  result.z =  result.z * color.z;
		  result.w = 0;
		  break;
	  }
	  
      PBOpos[index] = result;
 
  }
}



//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, float * nbo, int nbosize,  cudaMat4 projectionMatrix, cudaMat4 viewMatrix, float * wbo){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec4 v(vbo[index * 3 ],vbo[index * 3 + 1] ,vbo[index * 3 + 2],1);
	  v = multiplyMV4(viewMatrix, v);
	  vbo[index * 3] = v.x ;
	  vbo[index * 3 + 1] = v.y ;
	  vbo[index * 3 + 2] = v.z ;

	  viewMatrix.w = glm::vec4(0,0,0,1);
	  glm::vec4 n(nbo[index * 3 ],nbo[index * 3 + 1] ,nbo[index * 3 + 2],0);
	  n = multiplyMV4(viewMatrix, n);
	  nbo[index * 3] = n.x  ;
	  nbo[index * 3 + 1] = n.y;
	  nbo[index * 3 + 2] = n.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float * nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives,float * wbo){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	
	  triangle tmp;
	 //vertices
	 tmp.p0 = glm::vec3(vbo[ibo[index * 3] * 3],     vbo[ibo[index * 3]*3 + 1],    vbo[ibo[index * 3]*3 + 2]);
	 tmp.p1 = glm::vec3(vbo[ibo[index * 3 + 1] * 3], vbo[ibo[index * 3 + 1]*3 + 1],vbo[ibo[index * 3 + 1]*3 + 2]);
	 tmp.p2 = glm::vec3(vbo[ibo[index * 3 + 2] * 3], vbo[ibo[index * 3 + 2]*3 + 1],vbo[ibo[index * 3 + 2]*3 + 2]);
	 tmp.w3 = glm::vec3( wbo[ibo[index * 3]], wbo[ibo[index * 3 + 1]], wbo[ibo[index * 3 + 2]]);
	 //color
	// tmp.c0 = glm::vec3(cbo[ibo[index * 3] * 3], cbo[ibo[index * 3]*3 + 1],cbo[ibo[index * 3]*3 + 2]);
	// tmp.c1 = glm::vec3(cbo[ibo[index * 3] * 3], cbo[ibo[index * 3]*3 + 1],cbo[ibo[index * 3]*3 + 2]);
	 //tmp.c2 = glm::vec3(cbo[ibo[index * 3] * 3], cbo[ibo[index * 3]*3 + 1],cbo[ibo[index * 3]*3 + 2]);
	 tmp.c0 = glm::vec3(1,1,1);
	 tmp.c1 = glm::vec3(1,1,1);
	 tmp.c2 = glm::vec3(1,1,1);

	
	 //normals
	 tmp.n0 = glm::vec3(nbo[ibo[index * 3] * 3], nbo[ibo[index * 3]*3 + 1],nbo[ibo[index * 3]*3 + 2]);
	 tmp.n1 = glm::vec3(nbo[ibo[index * 3 + 1] * 3 ], nbo[ibo[index * 3 + 1]*3 + 1],nbo[ibo[index * 3 + 1]*3 + 2]);
	 tmp.n2 = glm::vec3(nbo[ibo[index * 3 + 2] * 3 ], nbo[ibo[index * 3 + 2]*3 + 1],nbo[ibo[index * 3 + 2]*3 + 2]);
	 
	 primitives[index] = tmp;
	
  }
}
__device__ glm::vec3 MultiplyMatrix(glm::vec3 vec, cudaMat4 & mat)
{
	glm::vec4 newvec(vec, 1);
	newvec = multiplyMV4(mat, newvec);
	newvec.x = newvec.x / newvec.w;
	newvec.y = newvec.y / newvec.w;
	newvec.z = newvec.z / newvec.w;

	return glm::vec3(newvec.x, newvec.y,newvec.z);
	
}


//TODO: Implement a rasterization method, such as scanline.
/*__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, cudaMat4 projectionMatrix){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	 
	  //initialize buffer

	  //projection 
	  triangle initialtri = primitives[index];
	  triangle tri = initialtri;
	  tri.p0 =  MultiplyMatrix(tri.p0,projectionMatrix);
	  tri.p1 =  MultiplyMatrix(tri.p1,projectionMatrix);
	  tri.p2 =  MultiplyMatrix(tri.p2,projectionMatrix);

	  tri.p0.x = 0.5f * resolution.x * (tri.p0.x + 1.0);
	  tri.p0.y = 0.5f * resolution.y * (tri.p0.y + 1.0);
	  tri.p1.x = 0.5f * resolution.x * (tri.p1.x + 1.0);
	  tri.p1.y = 0.5f * resolution.y * (tri.p1.y + 1.0);
	  tri.p2.x = 0.5f * resolution.x * (tri.p2.x + 1.0);
	  tri.p2.y = 0.5f * resolution.y * (tri.p2.y + 1.0);
	  int maxX, maxY, minX, minY;


	  //clip 
	  maxX = glm::min(glm::max(tri.p0.x, glm::max(tri.p1.x, tri.p2.x)),resolution.x - 1);
	  minX = glm::max(glm::min(tri.p0.x, glm::min(tri.p1.x, tri.p2.x)), 0.0f);
	  maxY = glm::min(glm::max(tri.p0.y, glm::max(tri.p1.y, tri.p2.y)),resolution.y - 1);
	  minY = glm::max(glm::min(tri.p0.y, glm::min(tri.p1.y, tri.p2.y)), 0.0f);



	  for(int y = minY ; y <= maxY; y ++)
	  {
		  for(int x = minX; x <= maxX; x++)
		  {

			 //  int ndcx = (x + 1) /2.0 * resolution.x;
				//  int ndcy = (y + 1) / 2.0 * resolution.y;
			  glm::vec3 baryCord = calculateBarycentricCoordinate(tri,glm::vec2(x,y));
			 
			  if(isBarycentricCoordInBounds(baryCord))
			  {
				  float tmpdepth = (baryCord.x * initialtri.p0 + baryCord.y * initialtri.p1 + baryCord.z * initialtri.p2).z;
				  //NDC COORDINATE

				 // depthbuffer[int(x + resolution.x * y)].normal = tri.n0;
				  //transform to cartesin
				  
				  tmpdepth = atomicAdd(&depthbuffer[int(x + resolution.x * y)].position.z, -tmpdepth);
				  if(depthbuffer[int(x + resolution.x * y)].position.z < 0)
				  {
					  
					 
					 fragment tmpfra;
					  tmpfra.color =(baryCord.x * initialtri.c0 + baryCord.y * initialtri.c1 + baryCord.z * initialtri.c2) ;
					  // tmpfra.color = glm::vec3(1.0,1.0,1.0);
				       tmpfra.normal = glm::normalize((baryCord.x * initialtri.n0 + baryCord.y * initialtri.n1 + baryCord.z * initialtri.n2));
				      tmpfra.position = (baryCord.x * initialtri.p0 + baryCord.y * initialtri.p1 + baryCord.z * initialtri.p2);
					  //tmpfra.position = glm::vec3(1,0,0);
					  // tmpfra.lightDir = glm::vec3(1,0,0);
					   depthbuffer[int(x + resolution.x * y)] = tmpfra;
				  }
				  else
				  {
					  depthbuffer[int(x + resolution.x * y)].position.z = tmpdepth;
				  }
			  }

		  }
	  }

  }
}*/


__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::vec4 windowsize, cudaMat4 projectionMatrix){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	 
	  //initialize buffer

	  //projection 
	  triangle initialtri = primitives[index];
	  triangle tri = initialtri;
	  tri.p0 =  MultiplyMatrix(tri.p0,projectionMatrix);
	  tri.p1 =  MultiplyMatrix(tri.p1,projectionMatrix);
	  tri.p2 =  MultiplyMatrix(tri.p2,projectionMatrix);

	  tri.p0.x = 0.5f * resolution.x * (tri.p0.x + 1.0);
	  tri.p0.y = 0.5f * resolution.y * (tri.p0.y + 1.0);
	  tri.p1.x = 0.5f * resolution.x * (tri.p1.x + 1.0);
	  tri.p1.y = 0.5f * resolution.y * (tri.p1.y + 1.0);
	  tri.p2.x = 0.5f * resolution.x * (tri.p2.x + 1.0);
	  tri.p2.y = 0.5f * resolution.y * (tri.p2.y + 1.0);
	  int maxX, maxY, minX, minY;


	  //clip 
	  maxX = glm::min(glm::max(tri.p0.x, glm::max(tri.p1.x, tri.p2.x)),windowsize.z);
	  minX = glm::max(glm::min(tri.p0.x, glm::min(tri.p1.x, tri.p2.x)), windowsize.x);
	  maxY = glm::min(glm::max(tri.p0.y, glm::max(tri.p1.y, tri.p2.y)),windowsize.w);
	  minY = glm::max(glm::min(tri.p0.y, glm::min(tri.p1.y, tri.p2.y)), windowsize.y);



	  for(int y = minY ; y <= maxY; y ++)
	  {
		  for(int x = minX; x <= maxX; x++)
		  {

			 //  int ndcx = (x + 1) /2.0 * resolution.x;
				//  int ndcy = (y + 1) / 2.0 * resolution.y;
			  glm::vec3 baryCord = calculateBarycentricCoordinate(tri,glm::vec2(x,y));
			 
			  if(isBarycentricCoordInBounds(baryCord))
			  {
				  float tmpdepth = (baryCord.x * initialtri.p0 + baryCord.y * initialtri.p1 + baryCord.z * initialtri.p2).z;
				  //NDC COORDINATE

				 // depthbuffer[int(x + resolution.x * y)].normal = tri.n0;
				  //transform to cartesin
				  
				  tmpdepth = atomicAdd(&depthbuffer[int(x + resolution.x * y)].position.z, -tmpdepth);
				  if(depthbuffer[int(x + resolution.x * y)].position.z < 0)
				  {
					  
					 
					 fragment tmpfra;
					  tmpfra.color =(baryCord.x * initialtri.c0 + baryCord.y * initialtri.c1 + baryCord.z * initialtri.c2) ;
					  // tmpfra.color = glm::vec3(1.0,1.0,1.0);
				       tmpfra.normal = glm::normalize((baryCord.x * initialtri.n0 + baryCord.y * initialtri.n1 + baryCord.z * initialtri.n2));
				      tmpfra.position = (baryCord.x * initialtri.p0 + baryCord.y * initialtri.p1 + baryCord.z * initialtri.p2);
					  //tmpfra.position = glm::vec3(1,0,0);
					  // tmpfra.lightDir = glm::vec3(1,0,0);
					   depthbuffer[int(x + resolution.x * y)] = tmpfra;
				  }
				  else
				  {
					  depthbuffer[int(x + resolution.x * y)].position.z = tmpdepth;
				  }
			  }

		  }
	  }

  }
}
//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution,glm::vec3 lightPosition,glm::vec4 windowsize){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x + windowsize.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y + windowsize.y;
  int index = x + (y * resolution.x);
  if( x >= windowsize.x && y >= windowsize.y && x <= windowsize.z && y<= windowsize.w){

	  fragment frag = depthbuffer[index];
	  glm::vec3 lightDir = glm::normalize(lightPosition - frag.position);
	  depthbuffer[index].lightDir = lightDir;
	  depthbuffer[index].color.z = glm::dot(frag.normal,lightDir);
	// depthbuffer[index].color = frag.color * glm::max(glm::dot(frag.normal,lightDir), 0.0f);

	  
	 // glm::vec3 lightDir = light
	 // glm::vec3 color = depthbuffer[index].color * glm::dot(depthbuffer[index].normal, );
	 // color = color * glm::dot()
	 // depthbuffer[index].color = 
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer, glm::vec4 windowsize){

    int x = (blockIdx.x * blockDim.x) + threadIdx.x + windowsize.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y + windowsize.y;
  int index = x + (y * resolution.x);

  if(x >= windowsize.x && y >= windowsize.y && x <= windowsize.z && y<= windowsize.w)
  {
	  framebuffer[index] = depthbuffer[index].color;
  }
}

__global__ void cudaclearPBOpos( uchar4 * pbobuffer, glm::vec2 resolution)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x <= resolution.x && y <= resolution.y)
  {
	  pbobuffer[index].x = 1;
	  pbobuffer[index].y = 1;
	  pbobuffer[index].w = 0;
	  pbobuffer[index].z = 1;

  }
}
void clearPBOpos(uchar4 * PBOpos, int width, int height)
{
	int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(width)/float(tileSize)), (int)ceil(float(height)/float(tileSize)));
  cudaclearPBOpos<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, glm::vec2(width, height));

}

__global__ void cudadrawTexture(uchar4 * pbobuffer, glm::vec2 resolution, unsigned char * pictures, glm::vec3 pictureRes)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x <= resolution.x && y <= resolution.y)
  {
	  int picx = (x / resolution.x) * pictureRes.x;
	  int picy = (y/ resolution.y) * pictureRes.y;

	  int startindex = (picx + pictureRes.x * picy)*3;
	  pbobuffer[index].x = pictures[startindex];
	  pbobuffer[index].y = pictures[startindex+1];
	  pbobuffer[index].z = pictures[startindex + 2];
	  pbobuffer[index].w = 0;
  }
}


void drawTexture(uchar4 * PBOpos, int width, int height, Picture pics)
{
	int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  unsigned char * cudapic = NULL;
  cudaMalloc((void **)&cudapic, pics.width * pics.height * pics.depth * sizeof(unsigned char));
  cudaMemcpy(cudapic, pics.mapptr,pics.width * pics.height * pics.depth * sizeof(unsigned char),cudaMemcpyHostToDevice);

  dim3 fullBlocksPerGrid((int)ceil(float(width)/float(tileSize)), (int)ceil(float(height)/float(tileSize)));
  cudadrawTexture<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, glm::vec2(width, height), cudapic, glm::vec3(pics.width, pics.height, pics.depth));

  cudaFree(cudapic);
}
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float * nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){

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
  triangle * cpuprimitive = new triangle[ibosize/3];


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
  cudaMalloc((void **) & device_nbo, nbosize * sizeof(float));
  cudaMemcpy(device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);


 // tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  float * device_wbo;
  cudaMalloc((void **) & device_wbo, vbosize/3 * sizeof(float));
  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, projectionMatrix,viewMatrix, device_wbo);

  cudaDeviceSynchronize();

 /*   float * newvbo = new float[vbosize];
   cudaMemcpy(newvbo, device_nbo, vbosize * sizeof(float), cudaMemcpyDeviceToHost);

   for(int i = 0;i < vbosize / 3; i=i+3)
   {

	   std::cout <<"old " <<nbo[i] <<" "<<nbo[i+1] <<" "<<nbo[i+2]<<std::endl;
	   std::cout <<"new " << newvbo[i] <<" "<<newvbo[i+1] <<" "<<newvbo[i+2]<<std::endl;
   }*/
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives,device_wbo);



  cudaDeviceSynchronize();


 /*cudaMemcpy(cpuprimitive, primitives, ibosize/3 * sizeof(triangle), cudaMemcpyDeviceToHost);
   int * cpuindices = new int[ibosize];
  // cudaMemcpy(cpuindices, ibo, ibosize* sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < ibosize / 3; i++)
  {
	  std::cout<< "indices " << ibo[i] <<" " << ibo[i+1] <<" " <<ibo[i+2] <<std::endl;

	  std::cout << " cpu " << vbo[ibo[i * 3] * 3] <<" " << vbo[ibo[i* 3] * 3 + 1] <<" " << vbo[ibo[i* 3] * 3 + 2]<<std::endl;
	  std::cout << cpuprimitive[i].p0.x <<"  " <<  cpuprimitive[i].p0.y<<" " <<cpuprimitive[i].p0.z<< std::endl;
	  std::cout << " cpu " << vbo[ibo[i* 3+1] * 3] <<" " << vbo[ibo[i* 3+1] * 3 + 1] <<" "<< vbo[ibo[i* 3+1] * 3 + 2]<<std::endl;
	  std::cout << cpuprimitive[i].p1.x <<"  " <<  cpuprimitive[i].p1.y<<" " <<cpuprimitive[i].p1.z<<std::endl;
	  std::cout << " cpu " << vbo[ibo[i* 3+2] * 3] <<" " << vbo[ibo[i* 3+2] * 3 + 1] <<" "<< vbo[ibo[i* 3+2] * 3 + 2]<<std::endl;
	  std::cout << cpuprimitive[i].p2.x <<"  " <<  cpuprimitive[i].p2.y<<" " <<cpuprimitive[i].p2.z<<std::endl;
  }
  
  delete [] cpuprimitive;
  delete [] cpuindices;*/
  //------------------------------
  //rasterization
  //------------------------------
  glm::vec4 windowSize(0,0,resolution.x, resolution.y);
  if(scissortest)
  {
       windowSize = scissorWindow; 
	   fullBlocksPerGrid  = dim3((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
	  
  }

  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, windowSize,projectionMatrix);
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  glm::vec3 viewLightPos = multiplyMV(viewMatrix,glm::vec4(lightPosition,1));
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, viewLightPos,windowSize);

  cudaDeviceSynchronize();
  //  fragment * cpufrag = new fragment[(int)resolution.x*(int)resolution.y];
/*  cudaMemcpy(cpufrag, depthbuffer,(int)resolution.x*(int)resolution.y*sizeof(fragment), cudaMemcpyDeviceToHost);
    for(int i = 0; i < (int)resolution.x * (int)resolution.y; i++)
  {
	  /*if(cpufrag[i].position.z != -10000)
	  std::cout<<cpufrag[i].position.x<<" " <<cpufrag[i].position.y<<" "<<cpufrag[i].position.z<<std::endl;*/
	 // std::cout<<cpufrag[i].lightDir.x<<" " <<cpufrag[i].lightDir.y<<" "<<cpufrag[i].lightDir.z<<std::endl;
	/*  if(cpufrag[i].color.x != 0)
	  {
	   std::cout<<cpufrag[i].color.x<<" " <<cpufrag[i].color.y<<" "<<cpufrag[i].color.z<<std::endl;
	   std::cout<<cpufrag[i].position.x<<" " <<cpufrag[i].position.y<<" "<<cpufrag[i].position.z<<std::endl;
	   std::cout<<cpufrag[i].lightDir.x<<" " <<cpufrag[i].lightDir.y<<" "<<cpufrag[i].lightDir.z<<std::endl;
	   std::cout << cpufrag[i].normal.x<<" " <<cpufrag[i].normal.y<<" "<<cpufrag[i].normal.z<<std::endl;
	  }

  }*/

//  delete []cpufrag;
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer,windowSize);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer,windowSize,blendtype);

  cudaDeviceSynchronize();

  kernelCleanup();

  cudaFree(device_wbo);
  checkCUDAError("Kernel failed!");
}


void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( device_nbo );
}

