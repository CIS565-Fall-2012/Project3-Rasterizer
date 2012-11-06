-------------------------------------------------------------------------------
CIS565: Project 3: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Tuesday, 11/6/2012
-------------------------------------------------------------------------------

BLOG Link: http://seunghoon-cis565.blogspot.com/2012/11/project-3-cuda-rasterizer.html
-------------------------------------------------------------------------------
A brief description
-------------------------------------------------------------------------------
The goal of this project is to implement an entire graphic pipeline by using CUDA.

-------------------------------------------------------------------------------
Features
-------------------------------------------------------------------------------
- Basic
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, Blinn-Phong, implemented in the fragment shader 

- Addtional
* Correct color interpretation between points on a primitive
* Anti-aliasing
* Interactive camera via both mouse and keyboard


-------------------------------------------------------------------------------
How to build
-------------------------------------------------------------------------------
I developed this project on Visual Studio 2010.
Its solution file is located in "PROJ1_WIN/565Raytracer.sln".
You should be able to build it without modification.