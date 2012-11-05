// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  bool loadedScene = false;
  for(int i=1; i<argc; i++){
	string header; string data;
	istringstream liness(argv[i]);
	getline(liness, header, '='); getline(liness, data, '=');
	if(strcmp(header.c_str(), "mesh")==0){
	  //renderScene = new scene(data);
	  mesh = new obj();
	  objLoader* loader = new objLoader(data, mesh);
	  mesh->buildVBOs();
	  delete loader;
	  loadedScene = true;
	}
	else if(strcmp(header.c_str(), "texture")==0){
		BMP texture;
		texture.ReadFromFile(data.c_str());
		tMapWidth = texture.TellWidth();
		tMapHeight = texture.TellHeight();
		tMapsize = 3 * tMapWidth * tMapHeight;
		tMap = new float[tMapsize];

		for(int j = 0; j < tMapHeight; j++)
		{
			for(int i = 0; i < tMapWidth; i++)
			{
				int index = j * tMapWidth + i;
				tMap[3 * index] = texture(i,j)->Red / 255.0f;
				tMap[3 * index + 1] = texture(i,j)->Green / 255.0f;
				tMap[3 * index + 2] = texture(i,j)->Blue / 255.0f;
			}
		}
	}
  }

  if(!loadedScene){
	cout << "Usage: mesh=[obj file]" << endl;
	return 0;
  }

  if(tMapsize != -1)
	  cout << "Texture Loaded" << endl;
  else
	  cout << "No Textures Loaded" << endl;

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  // Launch CUDA/GL
  #ifdef __APPLE__
  // Needed in OSX to force use of OpenGL3.2 
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
  glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  init();
  #else
  init(argc, argv);
  initCamera();
  #endif

  initCuda();

  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  #ifdef __APPLE__
	// send into GLFW main loop
	while(1){
	  display();
	  if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS || !glfwGetWindowParam( GLFW_OPENED )){
		  kernelCleanup();
		  cudaDeviceReset(); 
		  exit(0);
	  }
	}

	glfwTerminate();
  #else
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(onMouseCb);
	glutMotionFunc(onMouseMotionCb); 

	glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void initCamera()
{
	//-------------------------------
	//Initialize Camera Values here
	//-------------------------------
	theCamera.dfltEye = glm::vec3(0.0, 10, 24.0);
	theCamera.dfltUp = glm::vec3(0.0, 1.0, 0.0);
	theCamera.dfltLook = glm::vec3(0.0, 0.0, 0.0);
	theCamera.dfltVfov = 30.0;
	theCamera.dfltAspect = width / (float)height;
	theCamera.dfltNear = 0.1;
	theCamera.dfltFar = 100.0;
	theCamera.dfltSpeed = 0.1;
	theCamera.dfltTurnRate = 0.5*(M_PI/180.0);
	theCamera.setViewport(glm::vec4(0,0,width, height));
	//-------------------------------
	
	theCamera.reset();
	theCamera.set(theCamera.dfltEye, theCamera.dfltLook, theCamera.dfltUp);
	theCamera.setProjection();
	setMatrices();
}

void setMatrices()
{
	CameraPosition = theCamera.getPosition();
	ViewMatrix = theCamera.getViewMatrix();
	Projection = theCamera.getProjection();
	ViewPort = theCamera.getViewport();
}

void runCuda(){
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	dptr=NULL;

	vbo = mesh->getVBO();
	vbosize = mesh->getVBOsize();

	nbo = mesh->getNBO();
	nbosize = mesh->getNBOsize();
	/*
	float newcbo[] = {1.0, 1.0, 1.0, 
					1.0, 1.0, 1.0, 
					1.0, 1.0, 1.0};
	cbo = newcbo;
	cbosize = 9;
	*/

	mesh->setColor(glm::vec3(1.0, 1.0, 1.0));
	cbo = mesh->getCBO();
	cbosize = mesh->getCBOsize();

	tbo = mesh->getTBO();
	tbosize = mesh->getTBOsize();

	ibo = mesh->getIBO();
	ibosize = mesh->getIBOsize();

	cudaGLMapBufferObject((void**)&dptr, pbo);
	
	cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, nbo, nbosize, cbo, cbosize, ibo, ibosize, modelMatrix, ViewMatrix, Projection, ViewPort, CameraPosition, LightPosition, LightColor, AmbientColor, specularCoefficient);
	
	cudaGLUnmapBufferObject(pbo);

	vbo = NULL;
	cbo = NULL;
	ibo = NULL;
	nbo = NULL;

	frame++;
	fpstracker++;
}

#ifdef __APPLE__

  void display(){
	  runCuda();
	  time_t seconds2 = time (NULL);

	  if(seconds2-seconds >= 1){

		fps = fpstracker/(seconds2-seconds);
		fpstracker = 0;
		seconds = seconds2;

	  }

	  string title = "CIS565 Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS";

	  glfwSetWindowTitle(title.c_str());


	  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
	  glBindTexture(GL_TEXTURE_2D, displayImage);
	  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);


	  glClear(GL_COLOR_BUFFER_BIT);   

	  // VAO, shader program, and texture already bound
	  glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

	  glfwSwapBuffers();
  }

#else

void display(){
	runCuda();
	time_t seconds2 = time (NULL);

	if(seconds2-seconds >= 1){

		fps = fpstracker/(seconds2-seconds);
		fpstracker = 0;
		seconds = seconds2;

	}

	string title = "CIS565 Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS";
	glutSetWindowTitle(title.c_str());

	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glClear(GL_COLOR_BUFFER_BIT);   

	// VAO, shader program, and texture already bound
	glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
	
	glutPostRedisplay();
	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key) 
	{
		case(27):
			shut_down(1);    
			break;
//Reset Model View to Identity		
		case 'r':
			modelMatrix = glm::mat4(1.0);
			break;
//Rotations		
		case 'y':
			modelMatrix = glm::rotate(modelMatrix, 5.0f, glm::vec3(0.0f, 1.0f, 0.0f));
			break;

		case 'Y':
			modelMatrix = glm::rotate(modelMatrix, -5.0f, glm::vec3(0.0f, 1.0f, 0.0f));
			break;

		case 'x':
			modelMatrix = glm::rotate(modelMatrix, 5.0f, glm::vec3(1.0f, 0.0f, 0.0f));
			break;

		case 'X':
			modelMatrix = glm::rotate(modelMatrix, -5.0f, glm::vec3(1.0f, 0.0f, 0.0f));
			break;

		case 'z':
			modelMatrix = glm::rotate(modelMatrix, 5.0f, glm::vec3(0.0f, 0.0f, 1.0f));
			break;

		case 'Z':
			modelMatrix = glm::rotate(modelMatrix, -5.0f, glm::vec3(0.0f, 0.0f, 1.0f));
			break;
//Translations
		case 'a':
			modelMatrix = glm::translate(modelMatrix, glm::vec3(-0.5, 0.0, 0.0));
			break;

		case 'd':
			modelMatrix = glm::translate(modelMatrix, glm::vec3(+0.5, 0.0, 0.0));
			break;
		
		case 'w':
			modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0, +0.5, 0.0));
			break;
		
		case 's':
			modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0, -0.5, 0.0));
			break;
		
		case 'q':
			modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0, 0.0, +0.5));
			break;
		
		case 'e':
			modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0, 0.0, -0.5));
			break;	
//Scales
		case 'j':
			modelMatrix = glm::scale(modelMatrix, glm::vec3(2.0, 1.0, 1.0));
			break;

		case 'J':
			modelMatrix = glm::scale(modelMatrix, glm::vec3(0.5, 1.0, 2.0));
			break;
		
		case 'k':
			modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0, 2.0, 1.0));
			break;
		
		case 'K':
			modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0, 0.5, 1.0));
			break;
		
		case 'l':
			modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0, 1.0, 2.0));
			break;
		
		case 'L':
			modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0, 1.0, 0.5));
			break;

		case '1':
			UseFragmentShader = !UseFragmentShader;
			break;

		case'2':
			UseDiffuseShade = !UseDiffuseShade;
			break;

		case'3':
			UseSpecularShade = !UseSpecularShade;
			break;

		case'4':
			UseAmbientShade = !UseAmbientShade;
			break;

		case'5':
			UseDepthShade = !UseDepthShade;
			break;
	}
}

void onMouseMotionCb(int x, int y)
{
   int deltaX = lastX - x;
   int deltaY = lastY - y;
   bool moveLeftRight = abs(deltaX) > abs(deltaY);
   bool moveUpDown = !moveLeftRight;

   if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
   {
	  if (moveLeftRight && deltaX > 0) theCamera.orbitLeft(deltaX);
	  else if (moveLeftRight && deltaX < 0) theCamera.orbitRight(-deltaX);
	  else if (moveUpDown && deltaY > 0) theCamera.orbitUp(deltaY);
	  else if (moveUpDown && deltaY < 0) theCamera.orbitDown(-deltaY);
   }
   else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
   {
	   if (theModifierState & GLUT_ACTIVE_ALT) // camera move   
	   {
			if (moveLeftRight && deltaX > 0) theCamera.moveLeft(deltaX);
			else if (moveLeftRight && deltaX < 0) theCamera.moveRight(-deltaX);
			else if (moveUpDown && deltaY > 0) theCamera.moveUp(deltaY);
			else if (moveUpDown && deltaY < 0) theCamera.moveDown(-deltaY);
	   }
	   else
	   {
		   if (moveUpDown && deltaY > 0) theCamera.moveForward(deltaY);
		   else if (moveUpDown && deltaY < 0) theCamera.moveBack(-deltaY);
	   }

   }
   else if (theButtonState == GLUT_RIGHT_BUTTON) // Zoom
   {
	   if (theModifierState & GLUT_ACTIVE_CTRL || theModifierState & GLUT_ACTIVE_ALT) // camera move   
	   {
		   initCamera();
	   }
   }
 
   lastX = x;
   lastY = y;
   setMatrices();
   glutPostRedisplay();
}

void onMouseCb(int button, int state, int x, int y)
{
   theButtonState = button;
   theModifierState = glutGetModifiers();
   lastX = x;
   lastY = y;
   //glutSetMenu(theMenu);
}

 void DrawOverlay()
{
  // Draw Overlay
  glColor4f(1.0, 0.0, 0.0, 1.0);
  glPushAttrib(GL_LIGHTING_BIT);
	 glDisable(GL_LIGHTING);

	 glMatrixMode(GL_PROJECTION);
	 glLoadIdentity();
	 gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	 glMatrixMode(GL_MODELVIEW);
	 glLoadIdentity();
	 glRasterPos2f(0.1, 0.1);
	 char info[1024];
	 sprintf(info, "Test Run");
	 //sprintf(info, "Framerate: %3.1f  |  Frame: %u  |  %s", 
	 //    theFpsTracker.fpsAverage(), theSmokeSim.getTotalFrames(),
	 //    theSmokeSim.isRecording()? "Recording..." : "");
	 //FrameNum = theSmokeSim.getTotalFrames();
	 for (unsigned int i = 0; i < strlen(info); i++)
	 {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, info[i]);
	 }
  glPopAttrib();
  glutPostRedisplay();
}

void DrawAxes()
{
	glPushAttrib(GL_LIGHTING_BIT | GL_LINE_BIT);
		glDisable(GL_LIGHTING);

		glLineWidth(2.0); 
		glBegin(GL_LINES);
			glColor3f(1.0, 0.0, 0.0);
			glVertex3f(0.0, 0.0, 2.0);
			glVertex3f(2.0, 0.0, 2.0);

			glColor3f(1.0, 0.0, 0.0);
			glVertex3f(0.0, 0.0, 2.0);
			glVertex3f(0.0, 2.0, 2.0);

			glColor3f(1.0, 0.0, 0.0);
			glVertex3f(0.0, 0.0, 2.0);
			glVertex3f(0.0, 0.0, 4.0);
		glEnd();
	glPopAttrib();
	glutPostRedisplay();
}

void DrawGrid()
{
	glPushAttrib(GL_LIGHTING_BIT | GL_LINE_BIT);
		glDisable(GL_LIGHTING);

		glLineWidth(1.0); 
		glBegin(GL_LINES);
		for(int i = 0; i < width; i+=100)
		{
			//glColor4f(0.5, 0.5, 0.5, 0.2);
			glColor3f(1,1,1);
			glVertex3f(i, 0.0, 10);
			glVertex3f(i, height, 10.0);
		}
		for(int i = 0; i < height; i+=100)
		{
			//glColor4f(0.5, 0.5, 0.5, 0.2);
			glColor3f(1,1,1);
			glVertex3f(0.0, i, 0.0);
			glVertex3f(width, i, 0.0);
		}
		glEnd();
	glPopAttrib();
}

#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

#ifdef __APPLE__
  void init(){

	if (glfwInit() != GL_TRUE){
	  shut_down(1);      
	}

	// 16 bit color, no depth, alpha or stencil buffers, windowed
	if (glfwOpenWindow(width, height, 5, 6, 5, 0, 0, 0, GLFW_WINDOW) != GL_TRUE){
	  shut_down(1);
	}

	// Set up vertex array object, texture stuff
	initVAO();
	initTextures();
  }
#else
  void init(int argc, char* argv[]){
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("CIS565 Rasterizer");

	// Init GLEW
	glewInit();
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
	  /* Problem: glewInit failed, something is seriously wrong. */
	  std::cout << "glewInit failed, aborting." << std::endl;
	  exit (1);
	}

	initVAO();
	initTextures();
  }
#endif

void initPBO(GLuint* pbo){
  if (pbo) {
	// set up vertex data parameter
	int num_texels = width*height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
	
	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1,pbo);
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  runCuda();
}

void initTextures(){
	glGenTextures(1,&displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
	GLfloat vertices[] =
	{ 
		-1.0f, -1.0f, 
		 1.0f, -1.0f, 
		 1.0f,  1.0f, 
		-1.0f,  1.0f, 
	};

	GLfloat texcoords[] = 
	{ 
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);
	
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
	GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
	GLint location;

	glUseProgram(program);
	
	if ((location = glGetUniformLocation(program, "u_image")) != -1)
	{
		glUniform1i(location, 0);
	}

	return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
	// unregister this buffer object with CUDA
	cudaGLUnregisterBufferObject(*pbo);
	
	glBindBuffer(GL_ARRAY_BUFFER, *pbo);
	glDeleteBuffers(1, pbo);
	
	*pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  kernelCleanup();
  cudaDeviceReset();
  #ifdef __APPLE__
  glfwTerminate();
  #endif
  exit(return_code);
}
