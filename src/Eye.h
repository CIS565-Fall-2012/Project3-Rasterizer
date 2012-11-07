#ifndef EYE_H
#define EYE_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include "utilities.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"

class Eye
{
public:
	glm::vec3 pos; // wrt the world
	glm::vec3 rot; // wrt the world
	glm::vec3 up;
	glm::vec3 viewDir;
	glm::vec2 fov;
	glm::vec2 resolution;

	float l; // left plane
	float r; // right plane
	float b; // bottom plane
	float t; // top plane
	float n; // near plane
	float f; // far plane

	glm::mat4 transformIntoPerspective;

	Eye()
	{
		pos = glm::vec3(0, 0, 20);
		rot = glm::vec3(0,0,0);
		up = glm::vec3(0, 1, 0);
		viewDir = glm::vec3(0, 0, -1);
		resolution = glm::vec2(800, 800);
		fov = glm::vec2(30, 30);
	}

	void SetResolution(glm::vec2 fResolution)
	{
		resolution = fResolution;
	}

	void SetBoundariesOfView(float f_l, float f_r, float f_b, float f_t, float f_n, float f_f)
	{
		l = f_l;
		r = f_r;
		b = f_b;
		t = f_t;
		n = f_n;
		f = f_f;
		TransformIntoPerspectiveView();
	}

	glm::mat4 TransformWorldToEye()
	{
		// ModelView matrix
		glm::mat4 transformWorld;
		
		glm::mat4 translationMat = glm::translate(glm::mat4(), -pos);
		glm::mat4 rotationMat = glm::rotate(glm::mat4(), rot.x, glm::vec3(1,0,0));
		rotationMat = rotationMat*glm::rotate(glm::mat4(), rot.y, glm::vec3(0,1,0));
		rotationMat = rotationMat*glm::rotate(glm::mat4(), rot.z, glm::vec3(0,0,1));
		glm::mat4 transformWorld1 = translationMat*rotationMat;
		transformWorld[0] = glm::vec4(transformWorld1[0].x, transformWorld1[1].x, transformWorld1[2].x, transformWorld1[3].x);
		transformWorld[1] = glm::vec4(transformWorld1[0].y, transformWorld1[1].y, transformWorld1[2].y, transformWorld1[3].y);
		transformWorld[2] = glm::vec4(transformWorld1[0].z, transformWorld1[1].z, transformWorld1[2].z, transformWorld1[3].z);
		transformWorld[3] = glm::vec4(transformWorld1[0].w, transformWorld1[1].w, transformWorld1[2].w, transformWorld1[3].w);

		return transformWorld;
	}

	void TransformIntoPerspectiveView()
	{
		transformIntoPerspective[0] = glm::vec4(2*n/(r-l), 0, (r+l)/(r-l), 0);
		transformIntoPerspective[1] = glm::vec4(0, 2*n/(t-b), (t+b)/(t-b), 0);
		transformIntoPerspective[2] = glm::vec4(0, 0, -(f+n)/(f-n), -2*(f*n)/(f-n));
		transformIntoPerspective[3] = glm::vec4(0, 0, -1, 0);
	}

	glm::mat4 GetTransformWorldToPerspective()
	{
		return TransformWorldToEye()*transformIntoPerspective;
	}

	void GetParametersForScreenTransform(glm::vec2 resolution, glm::vec3& M, glm::vec3& A, glm::vec3& B, float& distImagePlaneFromCamera)
	{
	  viewDir = glm::normalize(viewDir);
	  up = glm::normalize(up);

	  A = glm::normalize(glm::cross(viewDir, up));
	  B = glm::normalize(glm::cross(A, viewDir));

	  float tanVert = tan(fov.y*PI/180);
	  
	  float camDistFromScreen = (float)((resolution.y/2.0)/tanVert);
	  fov.x = atan(resolution.x/(2*camDistFromScreen)) * 180/PI;
	  glm::vec3 C = viewDir*camDistFromScreen;
	  M = pos + C;

	  distImagePlaneFromCamera = camDistFromScreen;
	}
};

#endif
