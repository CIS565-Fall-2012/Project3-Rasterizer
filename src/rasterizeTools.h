// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"

struct vertex {
	glm::vec3 position;
	glm::vec3 color;
	glm::vec3 lightdir;
};

struct triangle {
  vertex v0;
  vertex v1;
  vertex v2;
  glm::vec3 normal;
};

struct fragment{
  glm::vec3 color;
  glm::vec3 normal;
  glm::vec3 position;
  glm::vec3 lightdir;
};

//Multiplies a cudaMat4 matrix and a vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Multiplies a cudaMat4 matrix and a vec4
__host__ __device__ glm::vec4 multiplyMV4(cudaMat4 m, glm::vec4 v){
  glm::vec4 r(1,1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  r.w = (m.w.x*v.x)+(m.w.y*v.y)+(m.w.z*v.z)+(m.w.w*v.w);
  return r;
}

//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getAABBForTriangle(triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint){
  minpoint = glm::vec3(min(min(tri.v0.position.x, tri.v1.position.x),tri.v2.position.x), 
        min(min(tri.v0.position.y, tri.v1.position.y),tri.v2.position.y),
        min(min(tri.v0.position.z, tri.v1.position.z),tri.v2.position.z));
  maxpoint = glm::vec3(max(max(tri.v0.position.x, tri.v1.position.x),tri.v2.position.x), 
        max(max(tri.v0.position.y, tri.v1.position.y),tri.v2.position.y),
        max(max(tri.v0.position.z, tri.v1.position.z),tri.v2.position.z));
}

//LOOK: calculates the signed area of a given triangle
__host__ __device__ float calculateSignedArea(triangle tri){
  return 0.5*((tri.v2.position.x - tri.v0.position.x)*(tri.v1.position.y - tri.v0.position.y) - (tri.v1.position.x - tri.v0.position.x)*(tri.v2.position.y - tri.v0.position.y));
}

//LOOK: helper function for calculating barycentric coordinates
__host__ __device__ float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri){
  triangle baryTri;
  baryTri.v0.position = glm::vec3(a,0); baryTri.v1.position = glm::vec3(b,0); baryTri.v2.position = glm::vec3(c,0);
  return calculateSignedArea(baryTri)/calculateSignedArea(tri);
}

//LOOK: calculates barycentric coordinates
__host__ __device__ glm::vec3 calculateBarycentricCoordinate(triangle tri, glm::vec2 point){
  float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri.v0.position.x,tri.v0.position.y), point, glm::vec2(tri.v2.position.x,tri.v2.position.y), tri);
  float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.v0.position.x,tri.v0.position.y), glm::vec2(tri.v1.position.x,tri.v1.position.y), point, tri);
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
	return -(barycentricCoord.x*tri.v0.position.z + barycentricCoord.y*tri.v1.position.z + barycentricCoord.z*tri.v2.position.z);
}

#endif