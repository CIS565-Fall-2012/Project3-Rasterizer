// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <cutil_math.h>
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

extern bool UseFragmentShader;
extern bool UseDiffuseShade;
extern bool UseSpecularShade;
extern bool UseAmbientShade;

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, glm::mat4 modelMatrix, glm::mat4 ViewMatrix, glm::mat4 Projection, glm::vec4 ViewPort, glm::vec3 CameraPosition, glm::vec3 LightPosition, glm::vec3 LightColor, glm::vec3 AmbientColor, float specularCoefficient);

#endif //RASTERIZEKERNEL_H
