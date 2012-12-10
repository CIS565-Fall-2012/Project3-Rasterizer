// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#include "glm/glm.hpp"


struct Picture
{
	unsigned char * mapptr;
	int width;
	int height;
	int depth;
};


 enum test
{
	SCISSOR_TEST,
	BLENDING
};
  enum blendType
 {
	 ADD,
	 SUBTRACT,
	 MAX,
	 NONE
 };
 blendType ReadBlendType();
 void SetBlendType(blendType type);

 void clearPBOpos(uchar4 * PBOpos, int width, int height);
  void SetScissorWindow(glm::vec4 windowsize);
 void Toggle(test testType);
void setProjectionMatrix(glm::mat4  & trans);
void setViewMatrix(glm::mat4 & trans);
void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float * nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize);
void drawTexture(uchar4 * PBOpos, int width, int height, Picture pics);
#endif //RASTERIZEKERNEL_H