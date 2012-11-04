
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <GL/glut.h>
#include <GL/gl.h>

#include "camera.h"

/*
glm::vec3  Camera::dfltEye(0.0, 0.0, -5.0);
glm::vec3  Camera::dfltUp(0.0, 1.0, 0.0);
glm::vec3  Camera::dfltLook(0.0, 0.0, 0.0);
float Camera::dfltVfov = 60.0;
float Camera::dfltAspect = 1.0;
float Camera::dfltNear = 0.5;
float Camera::dfltFar = 50.0;
float Camera::dfltSpeed = 0.1;
float Camera::dfltTurnRate = 1.0*(M_PI/180.0);
*/

Camera::Camera() 
{   
   myDir = NONE; myTurnDir = NONE;
   reset();
}

Camera::~Camera() {}

void Camera::reset()
{
   mSpeed = dfltSpeed;
   mTurnRate = dfltTurnRate;
   mVfov = dfltVfov;
   mAspect = dfltAspect;
   mNear = dfltNear;
   mFar = dfltFar;

   // Calculate the initial heading & pitch
   // Note that  eye[0] = radius*cos(h)*cos(p); and  eye[1] = radius*sin(p);
   mPitch = -asin(dfltEye[1]/glm::length(dfltEye));
   mHeading = acos(dfltEye[0]/(glm::length(dfltEye)*cos(mPitch)));
   //printf("INIT: %f %f\n", mPitch, mHeading);

   set(dfltEye, dfltLook, dfltUp);
}

const glm::vec3& Camera::getUp() const
{
   return u;
}

const glm::vec3& Camera::getBackward() const
{
   return n;
}

const glm::vec3& Camera::getRight() const
{
   return v;
}

glm::vec3 Camera::getRelativePosition(float left, float up, float forward)
{
   glm::vec3 direction = up*u + left*v - forward*n;
   return eye + direction;  // Move along forward axis 
}

void Camera::setViewport(glm::vec4 V)
{
	myViewport = V;
}

glm::vec4 Camera::getViewport()
{
	return myViewport;
}

glm::mat4 Camera::getViewMatrix()
{
	return glm::lookAt(eye, -n, u);
}

void Camera::setPosition(const glm::vec3& pos)
{
   eye = pos;
}

const glm::vec3& Camera::getPosition() const
{
   return eye;
}

glm::mat4 Camera::getProjection()
{
   //*vfov = mVfov; *aspect = mAspect; *zNear = mNear; *zFar = mFar;
	return glm::perspective(mVfov, mAspect, mNear, mFar);
}

void Camera::setProjection(float vfov, float aspect, float zNear, float zFar)
{
   mVfov = vfov;
   mAspect = aspect;
   mNear = zNear;
   mFar = zFar;
}

void Camera::setProjection()
{
	mVfov = dfltVfov;
	mAspect = dfltAspect;
	mNear = dfltNear;
	mFar = dfltFar;
}
float Camera::heading() const
{
   return mHeading;
}

float Camera::pitch() const
{
   return mPitch;
}

void Camera::set(const glm::vec3& eyepos, const glm::vec3& look, const glm::vec3& up)
{
   eye = eyepos;
   n = eyepos - look;
   v = glm::cross(up, n);
   u = glm::cross(n, v);
   mRadius = n.length(); // cache this distance

   u = glm::normalize(u);
   v = glm::normalize(v);
   n = glm::normalize(n);
}

void Camera::move(float dV, float dU, float dN)
{
   eye += dU*u + dV*v + dN*n;
}

void Camera::orbit(float h, float p)
{
  //printf("PITCH: %f\n", p);
  //printf("HEADING: %f\n", h);
  //printf("RADIUS: %f\n", mRadius);

   glm::vec3 rotatePt; // Calculate new location around sphere having mRadius
   rotatePt[0] = mRadius*cos(h)*cos(p);
   rotatePt[1] = mRadius*sin(p);
   rotatePt[2] = mRadius*sin(h)*cos(p);

   glm::vec3 lookAt = eye-n*mRadius;
   set(lookAt-rotatePt, lookAt /* look */, axisY /* up Approx */);
}

void Camera::orbitLeft(int scale) 
{
   myTurnDir = TL;
   mHeading += mTurnRate*scale;
   orbit(mHeading, pitch());
}

void Camera::moveLeft(int scale) // => move along v
{    
   myDir = L;
   move(-mSpeed*scale, 0.0, 0.0);
}

void Camera::orbitRight(int scale)
{
   myTurnDir = TR;
   mHeading -= mTurnRate*scale;
   orbit(mHeading, pitch());
}

void Camera::moveRight(int scale) // => move along v
{
   myDir = R;
   move(mSpeed*scale, 0.0, 0.0);   
}

void Camera::orbitUp(int scale)
{
   myTurnDir = TU; 
   mPitch = min(-0.1, mPitch + mTurnRate*scale);
   orbit(heading(), mPitch);
}

void Camera::moveUp(int scale) // => move along +u
{
   myDir = U;
   move(0.0, mSpeed*scale, 0.0);   
}

void Camera::orbitDown(int scale)
{
   myTurnDir = TD; 
   mPitch = max(-M_PI/2.0+0.01, mPitch - mTurnRate*scale);
   orbit(heading(), mPitch);
}

void Camera::moveDown(int scale) // => move along -u
{
   myDir = D;
   move(0.0, -mSpeed*scale, 0.0);   
}

void Camera::moveForward(int scale) // => move along -n
{
   myDir = F; 
   move(0.0, 0.0, -mSpeed*scale);      
   mRadius += -mSpeed*scale;  // Also "zoom" into radius
}

void Camera::moveBack(int scale) // => move along n
{
   myDir = B; 
   move(0.0, 0.0, mSpeed*scale);   
   mRadius += mSpeed*scale;  // Also "zoom" out radius
}

void Camera::turn(glm::vec3& v1, glm::vec3& v2, float amount)
{
   double cosTheta = cos(amount);
   double sinTheta = sin(amount);

   float vX =  cosTheta*v1[0] + sinTheta*v2[0]; 
   float vY =  cosTheta*v1[1] + sinTheta*v2[1]; 
   float vZ =  cosTheta*v1[2] + sinTheta*v2[2]; 

   float nX = -sinTheta*v1[0] + cosTheta*v2[0]; 
   float nY = -sinTheta*v1[1] + cosTheta*v2[1]; 
   float nZ = -sinTheta*v1[2] + cosTheta*v2[2]; 

   v1 = glm::vec3(vX, vY, vZ);
   v2 = glm::vec3(nX, nY, nZ);
}

void Camera::turnLeft(int scale) // rotate around u
{
   myTurnDir = TL; 
   turn(v, n, -mTurnRate*scale);
}

void Camera::turnRight(int scale) // rotate neg around u
{
   myTurnDir = TR;
   turn(v, n, mTurnRate*scale);
}

void Camera::turnUp(int scale) // rotate around v
{
   myTurnDir = TU; 
   turn(n, u, mTurnRate*scale);
}

void Camera::turnDown(int scale) // rotate around v
{
   myTurnDir = TD; 
   turn(n, u, -mTurnRate*scale);
}

/* UNUSED FUNCTIONS
bool Camera::screenToWorld(int screenX, int screenY, glm::vec3& worldCoords)
{
   double x, y, z;
   GLint result = gluUnProject(screenX, screenY, 0.0, 
							   myModelMatrix, myProjMatrix, myViewport, 
							   &x, &y, &z);

   worldCoords = glm::vec3(x, y, z);
   return result == GL_TRUE;
}

bool Camera::worldToScreen(const glm::vec3& worldCoords, int& screenX, int& screenY)
{
   double x, y, z;
   GLint result = gluProject(worldCoords[0], worldCoords[1], worldCoords[2],
							 myModelMatrix, myProjMatrix, myViewport, 
							 &x, &y, &z);

   screenX = (int) x;
   screenY = (int) y;
   return result == GL_TRUE;
}

math::matrix<double> Camera::cameraToWorldMatrix()
{
   math::matrix<double> tmp;
   tmp.Set(4, 4, myModelMatrix);
   tmp = tmp.Inv();
   return tmp;
}

void Camera::draw()
{
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(mVfov, mAspect, mNear, mFar);

   float m[16];
   m[0] = v[0]; m[4] = v[1]; m[8] = v[2];  m[12] = -glm::dot(eye, v); 
   m[1] = u[0]; m[5] = u[1]; m[9] = u[2];  m[13] = -glm::dot(eye, u); 
   m[2] = n[0]; m[6] = n[1]; m[10] = n[2]; m[14] = -glm::dot(eye, n); 
   m[3] = 0.0;  m[7] = 0.0;  m[11] = 0.0;  m[15] = 1.0;
   glMatrixMode(GL_MODELVIEW);
   glLoadMatrixf(m); 

   glGetDoublev(GL_MODELVIEW_MATRIX, myModelMatrix);
   glGetDoublev(GL_PROJECTION_MATRIX, myProjMatrix);
   glGetIntegerv(GL_VIEWPORT, myViewport);
}


/*void Camera::print()
{
   eye.Print("EYE: ");
   v.Print("RIGHT: ");
   u.Print("UP: ");
   n.Print("N: ");
   printf("-----------------------\n");
}
*/