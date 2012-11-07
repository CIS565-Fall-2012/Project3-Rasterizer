-------------------------------------------------------------------------------
CIS565: Project 3: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2012
-------------------------------------------------------------------------------
Due Tuesday 11/06/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
BLOG
-------------------------------------------------------------------------------
http://liamboone.blogspot.com/2012/11/project-3-rasterizer.html

-------------------------------------------------------------------------------
Description
-------------------------------------------------------------------------------
In the third project we were tasked with coding a CUDA based rasterization pipeline. I have implemented the actual rasterization with a very simple method of checking barycentric coordinates in an AABB around the triangles being drawn.

Features:	

	-Simple Lambert shading
	-Mouse controlled camera
		-left click and drag horizontally controls model rotation
		-left click and drag controls camera elevation
		-right click and drag vertically controls camera zoom
	-Vertex blending between the head, which is animated by it's own rotation matrix, and the body
	-Color blending
	-Back face culling

The vertex blending that I have added allows free rotation of the cow's head as can be seen in the video using a transformation matrix in addition to the one used on the body.
