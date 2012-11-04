
#ifndef camera_H_
#define camera_H_
#include <windows.h>

#include "GL/gl.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"


#ifndef M_PI
const double M_PI = 3.14159265358979323846f;		// per CRC handbook, 14th. ed.
#endif
const double M_PI_2 = double(M_PI/2.0f);				// PI/2
const double M2_PI = double(M_PI*2.0f);				// PI*2
const double Rad2Deg = double(180.0f / M_PI);			// Rad to Degree
const double Deg2Rad = double(M_PI / 180.0f);			// Degree to Rad

#ifndef EPSILON
#define EPSILON 0.001
#endif

#ifndef __MINMAX_DEFINED
#  define max(a,b)    (((a) > (b)) ? (a) : (b))
#  define min(a,b)    (((a) < (b)) ? (a) : (b))
#endif

const glm::vec3 axisX(1.0f, 0.0f, 0.0f);
const glm::vec3 axisY(0.0f, 1.0f, 0.0f);
const glm::vec3 axisZ(0.0f, 0.0f, 1.0f);
const glm::vec3 vec3Zero(0.0f, 0.0f, 0.0f);

class Camera
{
public:
   Camera();
   virtual ~Camera();

   // Draw projection and eyepoint
   //virtual void draw();

   // Print eyepoint position and basis
   //virtual void print();

   // Initialize the camera with glyLookAt parameters
   virtual void set(const glm::vec3& eyepos, const glm::vec3& look, const glm::vec3& up);

   // Get camera state
   virtual void setPosition(const glm::vec3& pos);
   virtual const glm::vec3& getPosition() const;
   virtual const glm::vec3& getUp() const;
   virtual const glm::vec3& getBackward() const;
   virtual const glm::vec3& getRight() const;
   virtual glm::vec3 getRelativePosition(float left, float up, float forward);
   virtual float heading() const;
   virtual float pitch() const;

   // Camera frustrum managements
   virtual void setProjection(float vfov, float aspect, float zNear, float zFar);
   virtual void setProjection();
   virtual glm::mat4 getProjection();
   virtual void setViewport(glm::vec4 V);
   virtual glm::vec4 getViewport();
   virtual glm::mat4 getViewMatrix();

   // Relative movement commands
   virtual void moveLeft(int scale = 1.0);
   virtual void moveRight(int scale = 1.0);
   virtual void moveUp(int scale = 1.0);
   virtual void moveDown(int scale = 1.0);
   virtual void moveForward(int scale = 1.0);
   virtual void moveBack(int scale = 1.0);

   virtual void turnLeft(int scale = 1.0);
   virtual void turnRight(int scale = 1.0);
   virtual void turnUp(int scale = 1.0);
   virtual void turnDown(int scale = 1.0);

   virtual void orbitLeft(int scale = 1.0);
   virtual void orbitRight(int scale = 1.0);
   virtual void orbitUp(int scale = 1.0);
   virtual void orbitDown(int scale = 1.0);

   // Reset to original state
   virtual void reset();

   // Conversion utilities between screen and world coordinates
   //virtual bool screenToWorld(int screenX, int screenY, glm::vec3& worldCoords);
   //virtual bool worldToScreen(const glm::vec3& worldCoords, int& screenX, int& screenY);

   // Get camera to world matrix
   //virtual math::matrix<double> cameraToWorldMatrix();

protected:
   enum Dir { NONE, F, B, L, R, U, D, TL, TR, TU, TD} myDir, myTurnDir;
   virtual void turn(glm::vec3& v, glm::vec3& n, float amount);
   virtual void move(float dU, float dV, float dN);
   virtual void orbit(float h, float p);

protected:
   float mSpeed, mTurnRate;

   glm::vec3 eye; // camera position
   float mHeading, mPitch, mRadius;
   float mVfov, mAspect, mNear, mFar; // projection parameters
   
   // Basis of camera local coord system
   glm::vec3 u; // up
   glm::vec3 v; // v points right
   glm::vec3 n; // -n points forward -> towards LookAt Point

   // Cache useful values
   glm::mat4 myProjMatrix;
   glm::mat4 myModelMatrix;
   glm::vec4 myViewport;

public:

   // Defaults
   static glm::vec3 dfltEye, dfltUp, dfltLook;
   static float dfltVfov, dfltAspect, dfltNear, dfltFar; 
   static float dfltSpeed, dfltTurnRate;
   static glm::vec4 Viewport;
};

#endif
