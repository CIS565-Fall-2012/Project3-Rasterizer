// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){

  //Some Initializations
  currentEye = glm::vec3(0.0f, 0.0f, 5.0f);
  currentUp = glm::vec3(0, 1, 0);
  currentForward = glm::vec3(0, 0, -1);
  fovy = 90.0f;
  aspectRatio = 1.0f;
  currentNear = 1.0f;
  currentFar = 20.0f;
  lights = new Light[numberOfLights];
  lights[0].col = glm::vec3(1, 1, 1);
  lights[0].pos = glm::vec3(0, 3, 5);
  lights[1].col = glm::vec3(0.0f, 0.0f, 0.8f);
  lights[1].pos = glm::vec3(5, 3, -10);
  //Some Initializations


  bool loadedScene = false;
  glm::mat4 transform = glm::mat4(1);
  //Bovine
  transform = glm::scale(transform, glm::vec3(5, 5, 5));
  //Thor
  //transform = glm::scale(transform, glm::vec3(0.01f, 0.01f, 0.01f));
  //Bumble bee
  //transform = glm::scale(transform, glm::vec3(0.03f, 0.03f, 0.03f));
  //Tank
  //transform = glm::scale(transform, glm::vec3(0.00005f, 0.00005f, 0.00005f));
  //VEH
  //transform = glm::scale(transform, glm::vec3(0.005f, 0.005f, 0.005f));
  //Teddy Bear
  //transform = glm::scale(transform, glm::vec3(1, 1, 1);

  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "mesh")==0){
      //renderScene = new scene(data);
      mesh = new obj();
      objLoader* loader = new objLoader(data, mesh);
      mesh->buildVBOs();
	  mesh->setModelMatrix(transform);
      delete loader;
      loadedScene = true;
	  ++numberOfMeshes;
    }

	else if(strcmp(header.c_str(), "texture") == 0)
	{
		initializeTextureData(data);
	}
  }

  /*
  //For dealing with multiple meshes
  mesh = new obj[argc - 1];
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "mesh")==0){
      objLoader* loader = new objLoader(data, mesh);
      mesh[i - 1].buildVBOs();
	  mesh[i - 1].setModelMatrix(transform);
      delete loader;
      loadedScene = true;
	  ++numberOfMeshes;
    }
  }
  */

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
    return 0;
  }

  vto = mesh->getVTO();
  vtosize = mesh->getVTOsize();

  //Attempt 2
  //vector<glm::vec4>* texture = mesh->getTextureCoords();
  //ibo = mesh->getIBO();
  //ibosize = mesh->getIBOsize();
  //vtosize = 2 * ibosize;
  //vto = new float[vtosize];
  //char c;
  //for(unsigned int i = 0; i < ibosize; ++i)
  //{
	 // /*vto[2 * i] = texture[0][ibo[i]].x;
	 // vto[2 * i + 1] = texture[0][ibo[i]].y;*/
	 // std::cout << ibo[i] << "\n";
	 // system("pause");
  //}

  //Attempt 1
  //vector<glm::vec4>* texture = mesh->getTextureCoords();
  //vtosize = (int)(2 * texture->size());
  //vto = new float[vtosize];
  //for(unsigned int i = 0; i < texture->size(); ++i)
  //{
	 // vto[2 * i] = texture[0][i].x;
	 // vto[2 * i + 1] = texture[0][i].y;
	 // //std::cout << texture[0][i].x << "\t" << texture[0][i].y << "\t" << texture[0][i].z << "\n";
  //}

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
	//ADDED
	glutMouseFunc(onMouseCb);
    glutMotionFunc(onMouseMotionCb);
	//ADDED
    glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

void initializeTextureData(std::string s)
{
	BMP image1;
	if(image1.ReadFromFile(s.c_str()))
	{
		textureImageWidth = image1.TellWidth();
		textureImageHeight = image1.TellHeight();
		textureImage = new unsigned char[textureImageWidth * textureImageHeight * 3];
		for(int j = 0; j < textureImageHeight; ++j)
		{
			for(int i = 0; i < textureImageWidth; ++i)
			{
				RGBApixel p = image1.GetPixel(i, j);
				textureImage[3 * (i + j * textureImageWidth)] = p.Red;
				textureImage[3 * (i + j * textureImageWidth) + 1] = p.Green;
				textureImage[3 * (i + j * textureImageWidth) + 2] = p.Blue;
			}
		}
	}
	std::cout << "Texture Data Initialization Done!\n";
}


//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  dptr=NULL;
  
  vbo = mesh->getVBO();
  vbosize = mesh->getVBOsize();

  /*float newcbo[] = {0.0, 1.0, 0.0, 
                    0.0, 0.0, 1.0, 
                    1.0, 0.0, 0.0};*/
  /*float newcbo[] = {0.60f, 0.60f, 0.60f, 
                    0.60f, 0.60f, 0.60f, 
                    0.60f, 0.60f, 0.60f};
  cbo = newcbo;
  cbosize = 9;*/

  mesh->setColor(glm::vec3(0.8f, 0.8f, 0.8f));
  cbo = mesh->getCBO();
  cbosize = mesh->getCBOsize();

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  nbo = mesh->getNBO();
  nbosize = mesh->getNBOsize();


  //Computing the model-view-projection matrix right here

  theCamera.getProjection(&fovy, &aspectRatio, &currentNear, &currentFar);
  currentNear = -currentNear;
  currentFar = -currentFar;
  float range = 1.0f * tan((PI / 180.0f) * (fovy / 2.0f)) * (-currentNear);
  currentLeft = -range * aspectRatio;
  currentRight = range * aspectRatio;
  currentBottom = -range;
  currentTop = range;

  currentUp = theCamera.getUp();
  glm::vec3 cameraRight = theCamera.getRight();
  currentEye = theCamera.getPosition();
  currentForward = glm::cross(currentUp, cameraRight);


  //View Matrix
  glm::mat4 viewMatrix;
  glm::vec3 x = glm::cross(currentForward, currentUp);
  currentUp = glm::cross(x, currentForward);

  //Making the forward direction as (0, 0, -1)
  viewMatrix[0][0] = x.x;
  viewMatrix[1][0] = x.y;
  viewMatrix[2][0] = x.z;
  viewMatrix[0][1] = currentUp.x;
  viewMatrix[1][1] = currentUp.y;
  viewMatrix[2][1] = currentUp.z;
  viewMatrix[0][2] = -currentForward.x;
  viewMatrix[1][2] = -currentForward.y;
  viewMatrix[2][2] = -currentForward.z;

  viewMatrix[0][3] = 0.0f;
  viewMatrix[1][3] = 0.0f;
  viewMatrix[2][3] = 0.0f;
  viewMatrix[3][3] = 1.0f;

  viewMatrix[3][0] = 0.0f;
  viewMatrix[3][1] = 0.0f;
  viewMatrix[3][2] = 0.0f;
  
  //Translating eye to (0, 0, 0)
  viewMatrix = glm::translate(viewMatrix, -currentEye);

  //Perspective Projection Matrix
  glm::mat4 projectionMatrix;
  projectionMatrix[0][0] = 2.0f * currentNear / (currentRight - currentLeft);
  projectionMatrix[0][1] = 0.0;
  projectionMatrix[0][2] = -1.0f * (currentRight + currentLeft) / (currentRight - currentLeft);
  projectionMatrix[0][3] = 0.0f;

  projectionMatrix[1][0] = 0.0f;
  projectionMatrix[1][1] = 2.0f * currentNear / (currentTop - currentBottom);
  projectionMatrix[1][2] = -1.0f * (currentTop + currentBottom) / (currentTop - currentBottom);
  projectionMatrix[1][3] = 0.0f;

  projectionMatrix[2][0] = 0.0f;
  projectionMatrix[2][1] = 0.0f;
  projectionMatrix[2][2] = (currentFar + currentNear) / (currentFar - currentNear);
  projectionMatrix[2][3] = -1.0f * currentFar * currentNear / (currentFar - currentNear);

  projectionMatrix[3][0] = 0.0f;
  projectionMatrix[3][1] = 0.0f;
  projectionMatrix[3][2] = 1.0f;
  projectionMatrix[3][3] = 0.0f;

  //Computing model-view-projection matrix
  glm::mat4 modelViewProjection = projectionMatrix * viewMatrix * mesh->getModelMatrix();

  //Not Using glm
  /*float fov, aspect, zNear, zFar;
  theCamera.getProjection(&fov, &aspect, &zNear, &zFar);
  currentUp = theCamera.getUp();
  glm::vec3 cameraRight = theCamera.getRight();
  currentEye = theCamera.getPosition();

  glm::vec3 cameraForward = glm::cross(currentUp, cameraRight);
  glm::mat4 view = glm::lookAt(currentEye, currentEye + cameraForward, currentUp);
  glm::mat4 projection = glm::perspective(fov, aspect, zNear, zFar);
  glm::mat4 modelViewProjection = projection * view * mesh->getModelMatrix();*/

  //Converting to cudaMat4 before sending to GPU
  cudaMat4 cudaModelViewProjection = utilityCore::glmMat4ToCudaMat4(modelViewProjection);

  //For displaying the vertex co-ordinates before sending to GPU
  /*for(int i = 0; i < vbosize; ++i)
  {
	  std::cout << vbo[i] << "\t";
	  if((i + 1) % 3 == 0)
	  {
		  std::cout << "\n";
	  }
  }
  std::cout << "\n";*/

  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize, nbo, nbosize, vto, vtosize, cudaModelViewProjection,
	  currentEye, lights, numberOfLights, textureImage, textureImageWidth, textureImageHeight);
  cudaGLUnmapBufferObject(pbo);

  vbo = NULL;
  cbo = NULL;
  ibo = NULL;
  nbo = NULL;

  /*char ccc;
  std::cin >> ccc;*/
  

  //Needed when when more than 1 model is present
  /*
  float newcbo[] = {0.0, 1.0, 0.0, 
                    0.0, 0.0, 1.0, 
                    1.0, 0.0, 0.0};
  cbo = newcbo;
  cbosize = 9;

  vbosize = 0;
  ibosize = 0;

  for(unsigned int i = 0; i < numberOfMeshes; ++i)
  {
	vbosize += mesh[i].getVBOsize();
	ibosize += mesh[i].getIBOsize();
  }

  vbo = new float[vbosize];
  ibo = new int[ibosize];

  //Copying all vbo data from all meshes into vbo
  unsigned int currentIndexPos = 0;
  for(unsigned int meshIndex = 0; meshIndex < numberOfMeshes; ++meshIndex)
  {
	  float *tempVBO = mesh[meshIndex].getVBO();
	  unsigned int tempVBOSize = (unsigned int)(mesh[meshIndex].getVBOsize());
	  for(unsigned int i = 0; i < tempVBOSize; ++i)
	  {
		  vbo[i + currentIndexPos] = tempVBO[i];
	  }
	  currentIndexPos += tempVBOSize;
  }
  //Copying all vbo data from all meshes into vbo

  //Copying all ibo data from all meshes into ibo
  currentIndexPos = 0;
  for(unsigned int meshIndex = 0; meshIndex < numberOfMeshes; ++meshIndex)
  {
	  int *tempIBO = mesh[meshIndex].getIBO();
	  unsigned int tempIBOSize = (unsigned int)(mesh[meshIndex].getIBOsize());
	  for(unsigned int i = 0; i < tempIBOSize; ++i)
	  {
		  ibo[i + currentIndexPos] = tempIBO[i];
	  }
	  currentIndexPos += tempIBOSize;
  }
  //Copying all ibo data from all meshes into ibo

  cudaGLMapBufferObject((void**)&dptr, pbo);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, cbo, cbosize, ibo, ibosize);
  cudaGLUnmapBufferObject(pbo);
  
  delete [] vbo;
  delete [] cbo;
  delete [] ibo;
  */


  ++frame;
  ++fpstracker;

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

    string title = "Aparajith's GPU Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS";
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
    }
  }

  void onMouseCb(int button, int state, int x, int y)
  {
     theButtonState = button;
     theModifierState = glutGetModifiers();
     if (theButtonState == GLUT_RIGHT_BUTTON)
     {
  	   if (state == 0)
  	   {
  		   mouseStartX = x;
  		   mouseStartY = height - y;
  	   }
  	   else if (state == 1/* && addForce*/)
  	   {
  		   mouseEndX = x;
  		   mouseEndY = height - y;
  	   }
     }
  
     lastX = x;
     lastY = y;
     //glutSetMenu(theMenu);
  }

  void onMouseMotionCb(int x, int y)
  {
     int deltaX = lastX - x;
     int deltaY = lastY - y;
     bool moveLeftRight = abs(deltaX) > abs(deltaY);
     bool moveUpDown = !moveLeftRight;
  
     switch(theButtonState)
     {
     case GLUT_LEFT_BUTTON:
  	    // Move Camera
  		if (moveLeftRight && deltaX > 0) theCamera.orbitLeft(deltaX);
  		else if (moveLeftRight && deltaX < 0) theCamera.orbitRight(-deltaX);
  		else if (moveUpDown && deltaY > 0) theCamera.orbitUp(deltaY);
  		else if (moveUpDown && deltaY < 0) theCamera.orbitDown(-deltaY);
  		break;
     case GLUT_MIDDLE_BUTTON:
  	    // Zoom
  	   if (theModifierState & GLUT_ACTIVE_ALT) // camera move
         {
              if (moveLeftRight && deltaX > 0) theCamera.moveLeft(-deltaX);
              else if (moveLeftRight && deltaX < 0) theCamera.moveRight(deltaX);
              else if (moveUpDown && deltaY > 0) theCamera.moveUp(-deltaY);
              else if (moveUpDown && deltaY < 0) theCamera.moveDown(deltaY);
         }
         else
         {
             if (moveUpDown && deltaY > 0) theCamera.moveForward(deltaY);
             else if (moveUpDown && deltaY < 0) theCamera.moveBack(-deltaY);
         }
  	   break;
     case GLUT_RIGHT_BUTTON:
  	   break;
     }
  
     lastX = x;
     lastY = y;
     glutPostRedisplay();
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
    glutCreateWindow("Aparajith's GPU Rasterizer");

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
	initCamera();
  }
  void initCamera()
  {
      /*glEnable(GL_BLEND);
      glEnable(GL_ALPHA_TEST);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LEQUAL);
      glShadeModel(GL_SMOOTH);
  
      glEnable(GL_NORMALIZE);
      glDisable(GL_LIGHTING);
      glCullFace(GL_BACK);*/

	  Camera::dfltAspect = aspectRatio;
	  Camera::dfltEye = currentEye;
	  Camera::dfltFar = currentFar;
	  Camera::dfltLook = currentForward;
	  Camera::dfltNear = currentNear;
	  Camera::dfltUp = currentUp;
	  Camera::dfltVfov = fovy;

	  theCamera.reset();
  
  	  //double w = 0;
  	  //double h = 15;
  	  //double d = -50;
  	  //double angle = 0.5 * fovy * PI / 180.0;
  	  //double dist;
  	  //if (w > h) dist = w * 0.5 / tan(angle);  // aspect is 1, so this can be done
  	  //else dist = h*0.5/tan(angle);
  	  //theCamera.dfltEye = vec3(w *0.5, h, -(dist+d));
  	  //theCamera.dfltLook = vec3(0.0, 0.0, 0.0);
  	  //theCamera.reset();
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
