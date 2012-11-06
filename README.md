-------------------------------------------------------------------------------
CIS565: Project 3: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Monday 11/06/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Project Submission
-------------------------------------------------------------------------------
Find my blog with screen shot samples at : 
http://mzshehzanayub.blogspot.com/2012/11/cuda-parallel-rasterizer.html

The features and implementation details of the CUDA Rasterizer are as follows:

Rasterization Pipeline
----------------------

**Vertex Shader**
In the vertex shader of the rasterization pipeline, I mulptiplied the model-view-projection with the vertices in the VBO.

**Primitive Assembly**
The primitive structure in my code contains the following: 3 vertices (in screen space coordinates), 3 vertices (in world space), 3 normals (in world space), 3 colors, 3 UV Texture Coordinates, a boolean for Back-Face culling, and a boolean for enabling texture mapping.
The data from the various buffers (VBO, NBO, CBO, TBO etc) are collected and put together in a primitives array. 

**Rasterization**
The rasterization is the most crucial and complex stage of the rasterization pipeline. The parallelization in this stage is per primitive, ie. a kernel is launched for each primitive. I used a scanline algorithm for this stage.
The result of running this algorithm correctly is a fragment buffer of fragment data type of size resolution. Each fragment has a color, a position (in screen space), a position (in world space), a normal (in world space), a float for distance from the camera and a integer used for Atomic exchange.

The algorithm is as follows:

> 1. For each primitive, find the maximum and minimum coordinates of the triangle.
> 2. Iterate through the scanlines between the minimum and maximum y-coordinates.
> 3. Find the intersection of the scanline with the sides of the triangle. In most cases, there are 2 intersection points, except when only 1 vertex lies on the scanline.
> 4. Iterate through the x-coordinates between the 2 intersection points.
> 5. The X and Y coordinates represent the pixel coordinates on the screen.
> 6. Use Barycentric coordinates to compute the fragment values like position, normal and color (or texture).
> 7. Compute the depth test by finding the distance of the fragment in world space and if it is closer to the camera than the value in the depth buffer, then swap the values. (Use atomics to avoid races).


**Fragment Shader**
In the fragment shader I implemented a simple *Phong Shading* model (http://en.wikipedia.org/wiki/Phong_shading). The algorithm parallelizes for each fragment / pixel.
The algorithm uses the world position of the fragment calculated in the Rasterization step of the pipeline. 

Extra Features
--------------
**Tetxure Mapping with Texel Based Super-Sampled Anti-Aliasing**
I implemented planar texture mapping as a part of the rasterization pipeline. The texture (Bitmap file) is read using EasyBMP Library. The UV coordinates are read from the vt values of the OBJ file. The anti-aliasing is a simple super sampling technique wherein each fragment samples the texel value from the texture map and also samples and averages its 8 neighboring texel colors. The texture map must be passed as a command line argument as texture="FilePath/FileName.ext".

**Back-Face Culling**
A simple algorithm where the faces (primitives) not visible to the camera are not processed in the rasterization step. The condition is that if the dot product of the face normal and the direction from camera to the point (generally centeroid) on the primitve is positive, then the face can be culled. It is best to use a epsilon while checking for the values.

**Interaction with Camera and Mesh**
The camera is interactive using the mouse. The mesh can be translated rotated and scaled using the keyboard. One can also switch between the different shading models using the keyboard. The interaction methods are listed in more detail in the interaction section of this document.

-------------------------------------------------------------------------------
How To Make It Work:
-------------------------------------------------------------------------------
**Customization**
**In main.h**

> 1. Use line 75 to specifiy the desired reolution of the screen.
> 2. Use lines 85-89 to set the translation (85), rotation(86) and scale(87) of the mesh. Use line 89 to set the defualt color of the object.
> 3. Use lines 102-105 to set the light position, light color, ambient light color and specular co-efficient of the light.

**In main.cpp**

> 1. Use lines 117 to 124 to set up the camera parameters.

    theCamera.dfltEye    => Default Position
    theCamera.dfltUp     => Defualt Up
    theCamera.dfltLook   => Defualt Center of View
    theCamera.dfltVfov   => Default Feild of View
    theCamera.dfltAspect => Defualt aspect ratio (Computed from width and height -> do not change)
    theCamera.dfltNear   => Default Projection Near
    theCamera.dfltFar    => Defualt Projection Far
    theCamera.dfltSpeed  => Default Speed of Mouse interation
Note: These values will be used again when the camera is reset (see interation section for how to reset).

**Include Texture Map in the command line argument as texture="FilePath/FileName.ext" (after the obj mesh file name argument)**.

-------------------------------------------------------------------------------
Interaction
-------------------------------------------------------------------------------
**Keyboard**

Translate

    case 'w': move the mesh +0.5 along local Y-axis
    case 's': move the mesh -0.5 along local Y-axis
    case 'd': move the mesh +0.5 along local X-axis
    case 'a': move the mesh -0.5 along local X-axis
    case 'q': move the mesh +0.5 along local Z-axis
    case 'e': move the mesh -0.5 along local Z-axis

Rotate

    case 'x': rotate the mesh about local X-axis in anti-clockwise direction
    case 'X': rotate the mesh about local X-axis in clockwise direction
    case 'y': rotate the mesh about local Y-axis in anti-clockwise direction
    case 'Y': rotate the mesh about local Y-axis in clockwise direction
    case 'z': rotate the mesh about local Z-axis in anti-clockwise direction
    case 'Z': rotate the mesh about local Z-axis in clockwise direction

Scale

    case 'j': double the scale of the mesh in local X-Axis
    case 'J': half the scale of the mesh in local X-Axis
    case 'k': double the scale of the mesh in local Y-Axis
    case 'K': half the scale of the mesh in local Y-Axis
    case 'l': double the scale of the mesh in local Z-Axis
    case 'L': half the scale of the mesh in local Z-Axis

Reset
    
case 'r': Reset the model matrix to identity.

Shading

    case '1': Toggle on or off the use of Fragment Shader (Defualt = On)
    case '2': Toggle on or off the use of Diffuse Shading (Defualt = On) -> Fragment Shader must be enabled first
    case '3': Toggle on or off the use of Specular Shading (Defualt = On) -> Fragment Shader must be enabled first
    case '4': Toggle on or off the use of Ambient Light (Defualt = On) -> Fragment Shader must be enabled first
    case '5': Toggle on or off the use of Depth Shading (Defualt = On) -> Fragment Shader must be enabled first. Skips diffuse, specular and ambient.
    case '6': Toggle on or off the use of Textures (Defualt = On) -> Fragment Shader must be enabled first. 

Close
> Escape Key: Close the program.

**Mouse**
Mouse interation is all for the camera.

Left Button Hold and Move:

    Left: Orbit the camera left
    Right: Orbit the camera right
    Up: Orbit the camera up
    Down: Orbit the camera down

Middle Button (Click, not scroll) Hold and Move:

    Left: translate the camera left
    Right: translate the camera right
    Up: translate the camera up
    Down: translate the camera down

Middle Button (Click, not scroll) + Keep Pressed ALT Key Hold and Move:

    Up: Zoom Forwards (In)
    Down: Zoom Backwards (Out)

Right Button + CTRL (or) ALT + Move:

    Moving the mouse with the right mouse button + either of CTRL or ALT held down will reset the Camera to its "Default" Parameters


-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! Any card with CUDA compute capability 1.1 or higher will work fine for this project. For a full list of CUDA capable cards and their compute capability, please consult: http://developer.nvidia.com/cuda/cuda-gpus. If you do not have an NVIDIA graphics card in the machine you are working on, feel free to use any machine in the SIG Lab or in Moore100 labs. All machines in the SIG Lab and Moore100 are equipped with CUDA capable NVIDIA graphics cards. If this too proves to be a problem, please contact Patrick or Karl as soon as possible.

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
In this project, you will implement a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline. In this project, you will implement vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and write the resulting fragments to a framebuffer. More information about the rasterized graphics pipeline can be found in the 10/15 class slides and in your notes from CIS560.

The basecode provided includes an OBJ loader and much of the mundane I/O and bookkeeping code. The basecode also includes some functions that you may find useful, described below. The core rasterization pipeline is left for you to implement.

You MAY NOT use ANY raycasting/raytracing AT ALL in this project, EXCEPT in the fragment shader step. One of the purposes of this project is to see how a rasterization pipeline can generate graphics WITHOUT the need for raycasting! Raycasting may only be used in the fragment shader effect for interesting shading results, but is absolutely not allowed in any other stages of the pipeline.

Also, you MAY NOT use OpenGL ANYWHERE in this project, aside from the given OpenGL code for drawing Pixel Buffer Objects to the screen. Use of OpenGL for any pipeline stage instead of your own custom implementation will result in an incomplete project.

Finally, note that while this basecode is meant to serve as a strong starting point for a CUDA rasterizer, you are not required to use this basecode if you wish, and you may also change any part of the basecode specification as you please, so long as the final rendered result is correct.

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project3 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* objs/ contains an example cow.obj test file: the standard "bovine test".
* renders/ contains an example render of the given example cow.obj file with a z-depth fragment shader. 
* PROJ1_WIN/ contains a Windows Visual Studio 2010 project and all dependencies needed for building and running on Windows 7.
* PROJ1_OSX/ contains a OSX makefile, run script, and all dependencies needed for building and running on Mac OSX 10.8. 

The Windows and OSX versions of the project build and run exactly the same way as in Project0, Project1, and Project2.

-------------------------------------------------------------------------------
REQUIREMENTS:
-------------------------------------------------------------------------------
In this project, you are given code for:

* A library for loading/reading standard Alias/Wavefront .obj format mesh files and converting them to OpenGL style VBOs/IBOs
* A suggested order of kernels with which to implement the graphics pipeline
* Working code for CUDA-GL interop

You will need to implement the following stages of the graphics pipeline and features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

You are also required to implement at least 3 of the following features:

* Additional pipeline stages. Each one of these stages can count as 1 feature:
   * Geometry shader
   * Transformation feedback
   * Back-face culling
   * Scissor test
   * Stencil test
   * Blending

IMPORTANT: For each of these stages implemented, you must also add a section to your README stating what the expected performance impact of that pipeline stage is, and real performance comparisons between your rasterizer with that stage and without.

* Correct color interpretation between points on a primitive
* Texture mapping WITH texture filtering and perspective correct texture coordinates
* Support for additional primitices. Each one of these can count as HALF of a feature.
   * Lines
   * Line strips
   * Triangle fans
   * Triangle strips
   * Points
* Anti-aliasing
* Order-independent translucency using a k-buffer
* MOUSE BASED interactive camera support. Interactive camera support based only on the keyboard is not acceptable for this feature.

-------------------------------------------------------------------------------
BASE CODE TOUR:
-------------------------------------------------------------------------------
You will be working primarily in two files: rasterizeKernel.cu, and rasterizerTools.h. Within these files, areas that you need to complete are marked with a TODO comment. Areas that are useful to and serve as hints for optional features are marked with TODO (Optional). Functions that are useful for reference are marked with the comment LOOK.

* rasterizeKernels.cu contains the core rasterization pipeline. 
	* A suggested sequence of kernels exists in this file, but you may choose to alter the order of this sequence or merge entire kernels if you see fit. For example, if you decide that doing has benefits, you can choose to merge the vertex shader and primitive assembly kernels, or merge the perspective transform into another kernel. There is not necessarily a right sequence of kernels (although there are wrong sequences, such as placing fragment shading before vertex shading), and you may choose any sequence you want. Please document in your README what sequence you choose and why.
	* The provided kernels have had their input parameters removed beyond basic inputs such as the framebuffer. You will have to decide what inputs should go into each stage of the pipeline, and what outputs there should be. 

* rasterizeTools.h contains various useful tools, including a number of barycentric coordinate related functions that you may find useful in implementing scanline based rasterization...
	* A few pre-made structs are included for you to use, such as fragment and triangle. A simple rasterizer can be implemented with these structs as is. However, as with any part of the basecode, you may choose to modify, add to, use as-is, or outright ignore them as you see fit.
	* If you do choose to add to the fragment struct, be sure to include in your README a rationale for why. 

You will also want to familiarize yourself with:

* main.cpp, which contains code that transfers VBOs/CBOs/IBOs to the rasterization pipeline. Interactive camera work will also have to be implemented in this file if you choose that feature.
* utilities.h, which serves as a kitchen-sink of useful functions

-------------------------------------------------------------------------------
SOME RESOURCES:
-------------------------------------------------------------------------------
The following resources may be useful for this project:

* High-Performance Software Rasterization on GPUs
	* Paper (HPG 2011): http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf
	* Code: http://code.google.com/p/cudaraster/ Note that looking over this code for reference with regard to the paper is fine, but we most likely will not grant any requests to actually incorporate any of this code into your project.
	* Slides: http://bps11.idav.ucdavis.edu/talks/08-gpuSoftwareRasterLaineAndPantaleoni-BPS2011.pdf
* The Direct3D 10 System (SIGGRAPH 2006) - for those interested in doing geometry shaders and transform feedback.
	* http://133.11.9.3/~takeo/course/2006/media/papers/Direct3D10_siggraph2006.pdf
* Multi-Fragment Eﬀects on the GPU using the k-Buﬀer - for those who want to do a k-buffer
	* http://www.inf.ufrgs.br/~comba/papers/2007/kbuffer_preprint.pdf
* FreePipe: A Programmable, Parallel Rendering Architecture for Efficient Multi-Fragment Effects (I3D 2010)
	* https://sites.google.com/site/hmcen0921/cudarasterizer
* Writing A Software Rasterizer In Javascript:
	* Part 1: http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-1.html
	* Part 2: http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-2.html

-------------------------------------------------------------------------------
NOTES ON GLM:
-------------------------------------------------------------------------------
This project uses GLM, the GL Math library, for linear algebra. You need to know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but pre-Fermi cards do not play nice with GLM matrices. As such, in this project, GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is provided as multiplyMV() in intersections.h.

-------------------------------------------------------------------------------
BLOG
-------------------------------------------------------------------------------
As mentioned in class, all students should have student blogs detailing progress on projects. If you already have a blog, you can use it; otherwise, please create a blog using www.blogger.com or any other tool, such as www.wordpress.org. Blog posts on your project are due on the SAME DAY as the project, and should include:

* A brief description of the project and the specific features you implemented.
* A link to your github repo if the code is open source.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.  To create the video use http://www.microsoft.com/expression/products/Encoder4_Overview.aspx 

-------------------------------------------------------------------------------
THIRD PARTY CODE POLICY
-------------------------------------------------------------------------------
* Use of any third-party code must be approved by asking on Piazza.  If it is approved, all students are welcome to use it.  Generally, we approve use of third-party code that is not a core part of the project.  For example, for the ray tracer, we would approve using a third-party library for loading models, but would not approve copying and pasting a CUDA function for doing refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will result in you receiving an F for the semester.

-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Karl, yiningli@seas.upenn.edu, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

-------------------------------------------------------------------------------
SUBMISSION
-------------------------------------------------------------------------------
As with the previous project, you should fork this project and work inside of your fork. Upon completion, commit your finished project back to your fork, and make a pull request to the master repository.
You should include a README.md file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running, and at least one screenshot of the final rendered output of your raytracer
* Instructions for building and running your project if they differ from the base code
* A link to your blog post detailing the project
* A list of all third-party code used