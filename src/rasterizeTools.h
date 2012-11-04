// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"

//ADDED
struct Point
{
	glm::vec3 p0;
	glm::vec3 c0;
};

struct Line
{
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 c0;
	glm::vec3 c1;
};
//ADDED

struct triangle {
  glm::vec3 p0;
  glm::vec3 p1;
  glm::vec3 p2;
  glm::vec3 c0;
  glm::vec3 c1;
  glm::vec3 c2;
  //ADDED
  //glm::vec3 faceNormal;	// World Normal
  glm::vec3 n0;
  glm::vec3 n1;
  glm::vec3 n2;
  //ADDED
};

struct fragment{
  glm::vec3 color;
  glm::vec3 normal;
  glm::vec3 worldPosition;
  int screenSpaceX, screenSpaceY;
  float worldDepth;
  unsigned int lockVariable;
};

//Multiplies a cudaMat4 matrix and a vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v)
{
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Special Multiply for taking care of projection
__host__ __device__ glm::vec3 multiplyMVWithHomogenization(cudaMat4 m, glm::vec4 v)
{
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  float w = (m.w.x*v.x)+(m.w.y*v.y)+(m.w.z*v.z)+(m.w.w*v.w);
  r.x /= w;
  r.y /= w;
  r.z /= w;
  return r;
}

//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getAABBForTriangle(triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint){
  minpoint = glm::vec3(min(min(tri.p0.x, tri.p1.x),tri.p2.x), 
        min(min(tri.p0.y, tri.p1.y),tri.p2.y),
        min(min(tri.p0.z, tri.p1.z),tri.p2.z));
  maxpoint = glm::vec3(max(max(tri.p0.x, tri.p1.x),tri.p2.x), 
        max(max(tri.p0.y, tri.p1.y),tri.p2.y),
        max(max(tri.p0.z, tri.p1.z),tri.p2.z));
}

//LOOK: calculates the signed area of a given triangle
__host__ __device__ float calculateSignedArea(triangle tri){
  return 0.5*((tri.p2.x - tri.p0.x)*(tri.p1.y - tri.p0.y) - (tri.p1.x - tri.p0.x)*(tri.p2.y - tri.p0.y));
}

//LOOK: helper function for calculating barycentric coordinates
__host__ __device__ float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri){
  triangle baryTri;
  baryTri.p0 = glm::vec3(a,0); baryTri.p1 = glm::vec3(b,0); baryTri.p2 = glm::vec3(c,0);
  return calculateSignedArea(baryTri)/calculateSignedArea(tri);
}

//LOOK: calculates barycentric coordinates
__host__ __device__ glm::vec3 calculateBarycentricCoordinate(triangle tri, glm::vec2 point){
  float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri.p0.x,tri.p0.y), point, glm::vec2(tri.p2.x,tri.p2.y), tri);
  float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.p0.x,tri.p0.y), glm::vec2(tri.p1.x,tri.p1.y), point, tri);
  float alpha = 1.0-beta-gamma;
  return glm::vec3(alpha,beta,gamma);
}

//LOOK: checks if a barycentric coordinate is within the boundaries of a triangle
__host__ __device__ bool isBarycentricCoordInBounds(glm::vec3 barycentricCoord){
   return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
          barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
          barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

//LOOK: for a given barycentric coordinate, return the corresponding z position on the triangle
__host__ __device__ float getZAtCoordinate(glm::vec3 barycentricCoord, triangle tri){
  return -(barycentricCoord.x*tri.p0.z + barycentricCoord.y*tri.p1.z + barycentricCoord.z*tri.p2.z);
}

__host__ __device__ glm::vec3 getPositionAtBarycentricCoordinate(glm::vec3 barycentricCoord, triangle tri){
  return (barycentricCoord.x*tri.p0 + barycentricCoord.y*tri.p1 + barycentricCoord.z*tri.p2);
}

__host__ __device__ glm::vec3 getColorAtBarycentricCoordinate(glm::vec3 barycentricCoord, triangle tri){
	return (barycentricCoord.x*tri.c0 + barycentricCoord.y*tri.c1 + barycentricCoord.z*tri.c2);
}

__host__ __device__ float giveWorldSpaceDepth(glm::vec3 barycentricCoords, triangle originalTriangle, glm::vec3 eye)
{
	return glm::distance(eye, getPositionAtBarycentricCoordinate(barycentricCoords, originalTriangle));
}

__host__ __device__ glm::vec3 getNormalAtBarycentricCoordinate(glm::vec3 barycentricCoord, triangle tri){
	return (barycentricCoord.x*tri.n0 + barycentricCoord.y*tri.n1 + barycentricCoord.z*tri.n2);
}

#endif