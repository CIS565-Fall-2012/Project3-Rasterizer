// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H

#ifdef __APPLE__
	#include <GL/glfw.h>
#else
	#include <GL/glew.h>
	#include <GL/glut.h>
#endif

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>
#include "glslUtility.h"
#include "glm/glm.hpp"
#include "rasterizeKernels.h"
#include "utilities.h"
#include "ObjCore/objloader.h"
#include "glm/gtc/matrix_transform.hpp"

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

obj* mesh;

float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;

//-------------------------------
//----------CUDA & Camera STUFF-----------
//-------------------------------

int width=800; int height=800;
glm::vec3 CameraPosition = glm::vec3 (0, 2, 5.0);

glm::vec3 mTranslate = glm::vec3(0.0, 0.0, 0.0);
glm::vec3 mRotate = glm::vec3(0.0, 0.0, 0.0);
glm::vec3 mScale = glm::vec3(1.0, 1.0, 1.0);
glm::mat4 modelMatrix = utilityCore::buildTransformationMatrix(mTranslate, mRotate, mScale);

glm::vec3 LookAtCenter = glm::vec3(0.0, 0.5, 0.0);
glm::mat4 ViewMatrix = glm::lookAt(CameraPosition, LookAtCenter, glm::vec3(0.0, 1.0, 0.0));

glm::vec4 ViewPort = glm::vec4(0.0, 0.0, width, height);

glm::mat4 Projection = glm::perspective(30.0f, static_cast<float>(width) / static_cast<float>(height), 0.1f, 50.0f);


//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

#ifdef __APPLE__
	void display();
#else
	void display();
	void DrawGrid();
	void DrawOverlay();
	void DrawAxes();
	void keyboard(unsigned char key, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init();
#else
	void init(int argc, char* argv[]);
#endif

void initCamera();

void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath);

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif