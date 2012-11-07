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
#include "cudaMat4.h"
#include "Eye.h"

void kernelCleanup();
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, 
	                   int* ibo, int ibosize, float* nbo, int nbosize, Eye eye, cudaMat4 transform, cudaMat4 modelView, cudaMat4 perspective);

#endif //RASTERIZEKERNEL_H
