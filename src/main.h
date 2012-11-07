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
#include "glm/gtc/matrix_transform.hpp"
#include "rasterizeKernels.h"
#include "utilities.h"
#include "ObjCore/objloader.h"

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

float *nbo;
int nbosize;
float* vbo;
int vbosize;
float* cbo;
int cbosize;
int* ibo;
int ibosize;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=800; int height=800;
class Camera
{
public:
	glm::vec3 position;
	glm::vec3 up;
	glm::vec3 view;
	glm::vec3 right;
	float fov;
	Camera(glm::vec3 p = glm::vec3(0,-0.5,2), glm::vec3 v = glm::vec3(0,0,-1), glm::vec3 u = glm::vec3(0,-1,0), float f =20):position(p),view(v), up(u), fov(f){ right = glm::cross(up,-1.0f * view);};
};
Camera eye;
glm::vec3 center = glm::vec3(0,0,0);
float front = 0;
float back = 100;


Picture texture;
//Interactive Camera
int currentX, currentY;
bool dragging = false;
bool rotating = false;
bool clipping = false;
float rotateSpeed = 0.04f;
float draggingSpeed = 0.01f;
float minAngle = -20;
float maxAngle = 20;
glm::vec4 windowSize;
float zoomspeed = 0.2;
glm::mat4 transMatrix = glm::mat4(1.0);
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
	void keyboard(unsigned char key, int x, int y);
	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
	void init();
#else
	void init(int argc, char* argv[]);
#endif

void calcuatetransformationMatrix( Camera eye, glm::vec2 resolution, float front, float back);
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