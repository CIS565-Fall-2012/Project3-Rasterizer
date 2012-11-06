-------------------------------------------------------------------------------
CIS565: Project 3: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Kong Ma
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
Implemented Feature
-------------------------------------------------------------------------------
Finished all the basic features:
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* Implemented Blinn-Phong in the fragment shader

Finished following optional features
* Geometry shader to add more primitives
* Back-face culling
* Anti-aliasing

-------------------------------------------------------------------------------
BLOG LINK:
-------------------------------------------------------------------------------
http://gpuprojects.blogspot.com/2012/10/path-tracer.html
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
Sequence Of Implementation
-------------------------------------------------------------------------------
* Vertex Shader
	* In Vertex Shader, Vertices data are transformed from object coordinates to world coordinates, then to camera coordinates , then through projection transformed to clipped coordinates and finally to NDC coordinates
* Primitive Assembly
	* In Primitive Assembly, vertices are grouped to form primitives. At the end of the Stage, backface culling flag are calculated.
* Geometry Shader(Optional)
	* Implemented a simple version of geomtery shader, for each primitives generate more primitives, if backface culling are enabled, flags are calculated
* rasterization
	* Implemented the rasterization based on tile approach. Before the raseterization, backface primitives and primitives outside viewPort are discarded(clipping). Antialiasing can be enabled in this stage. Depth test are used in this stage, only the nearest primitives colors are stored to frame buffer.	
* Fragment Shader	
	* Implemented a simple Blinn-Phong shader. All the color values are calculated in eye coordinates.

-------------------------------------------------------------------------------
Other
-------------------------------------------------------------------------------
	1. Define the following parameter to enable different features :GEOMETRYSHADER,BackFaceCulling,AntiAliasing,Clipping in rasterizeKernel.cu.
	2. zNear and zFar defined in rasterizeTools.h are used to control the front plane and back plane of the projection, value must>0.
	3. DEPTHBUFFERSIZE defined in rasterizeTools.h is used to define depth buffer size (number of bite).
	4. DEPTHPRECISION defined in rasterizeTools.h is used to define the distance of depth value which would be regarded as "close" in antialiasing.

