// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>

#ifdef __APPLE__
	#include <GL/glfw.h>
#else
	#include <GL/glew.h>
	#include <GL/glut.h>
#endif

#include <cuda_runtime.h>
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
#include "camera.h"
#include "EasyBMP.h"
#include "EasyBMP_BMP.h"
#include "EasyBMP_DataStructures.h"
#include "EasyBMP_VariousBMPutilities.h"

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
float* nbo;
int nbosize;
float* tbo;
int tbosize;

//Texture Map Variables
float* tMap;
int tMapsize = 0;
int tMapWidth = 0;
int tMapHeight = 0;

//-------------------------------
//----------CUDA & Camera STUFF-----------
//-------------------------------

//int width=800; int height=800;
int width=1280; int height=720;

bool UseFragmentShader = true;
bool UseDiffuseShade = true;
bool UseSpecularShade = true;
bool UseAmbientShade = true;
bool UseDepthShade = false;
bool UseTextures = true;
//////////////////////////////Specify Mesh Model Matrix Values Here///////////////////////////////////
glm::vec3 mTranslate = glm::vec3(0.0, 0.0, 0.0);
glm::vec3 mRotate = glm::vec3(0.0, 0.0, 0.0);
glm::vec3 mScale = glm::vec3(1.0, 1.0, 1.0);
glm::mat4 modelMatrix = utilityCore::buildTransformationMatrix(mTranslate, mRotate, mScale);
glm::vec3 Obj_Color = glm::vec3(1.0, 1.0, 1.0);
/////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////Camera Matrices and Vectoes///////////////////////////////////////////
glm::vec3 CameraPosition;
glm::mat4 ViewMatrix;
glm::mat4 Projection;
glm::vec4 ViewPort;
/////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////Specify Light Values Here///////////////////////////////////
glm::vec3 LightPosition = glm::vec3(5.0, 10.0, 5.0);
glm::vec3 LightColor	= glm::vec3(1.0, 1.0, 1.0);
glm::vec3 AmbientColor	= glm::vec3(0.1, 0.1, 0.1);
float specularCoefficient = 20.0f;
/////////////////////////////////////////////////////////////////////////////////////////

glm::vec3  Camera::dfltEye(0.0, 7.5, 15.0);
glm::vec3  Camera::dfltUp(0.0, 1.0, 0.0);
glm::vec3  Camera::dfltLook(0.0, 0.0, 0.0);
float Camera::dfltVfov = 30.0;
float Camera::dfltAspect = width / (float)height;
float Camera::dfltNear = 0.1;
float Camera::dfltFar = 100.0;
float Camera::dfltSpeed = 0.1;
float Camera::dfltTurnRate = 1.0*(M_PI/180.0);
glm::vec4 Camera::Viewport = glm::vec4(0,0,width, height);
Camera theCamera = Camera();

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------
//UI Helpers
// UI Helpers
int lastX = 0, lastY = 0;
int theMenu = 0;
int theButtonState = 0;
int theModifierState = 0;


void runCuda();

#ifdef __APPLE__
	void display();
#else
	void display();
	void DrawGrid();
	void DrawOverlay();
	void DrawAxes();
	void keyboard(unsigned char key, int x, int y);
	void onMouseMotionCb(int x, int y);
	void onMouseCb(int button, int state, int x, int y);
	void initCamera();
	void setMatrices();
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init();
#else
	void init(int argc, char* argv[]);
#endif

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