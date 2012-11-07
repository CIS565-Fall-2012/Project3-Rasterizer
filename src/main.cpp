// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include "main.h"
#include "stb_image/stb_image.h"

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
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
    return 0;
  }

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
	glutMouseFunc(mouse);
	 glutMotionFunc(motion); 
    glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	

  vbo = mesh->getVBO();

  vbosize = mesh->getVBOsize();
  nbo = mesh->getNBO();

  nbosize = mesh->getNBOsize();
  float newcbo[] = {0.0, 1.0, 0.0, 
                    0.0, 0.0, 1.0, 
                    1.0, 0.0, 0.0};
  cbo = newcbo;
  cbosize = 9;

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();
 
  calcuatetransformationMatrix( eye,glm::vec2(width, height), front,  back);
    dptr=NULL;
  cudaGLMapBufferObject((void**)&dptr, pbo);
  if(ReadBlendType() == ADD)
  {
	  drawTexture(dptr,width, height,texture);
  }
 
  //clearPBOpos(dptr,width,height);
  cudaRasterizeCore(dptr, glm::vec2(width, height), frame, vbo, vbosize, nbo, nbosize, cbo, cbosize, ibo, ibosize);
  cudaGLUnmapBufferObject(pbo);

 
  vbo = NULL;
  cbo = NULL;
  ibo = NULL;

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

	 glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, 'ç');
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
	   case('s'):
		   Toggle(SCISSOR_TEST);
		   break;
		   case('S'):
		   Toggle(SCISSOR_TEST);
		   break;
	   case ('+'):
		   break;
	   case ('b'):
		   SetBlendType(ADD);
		   break;
		   case ('B'):
		   SetBlendType(ADD);
		   break;

    }
  }
	void mouse(int button, int state, int x, int y)
	{
		if(!clipping && glutGetModifiers() == GLUT_ACTIVE_ALT && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN )
		{

			currentX = x;
			currentY = y;
			rotating = true;
			return;

			
		}



		if(!clipping && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		{
			windowSize.x = x;
			windowSize.y = y;
			clipping = true;
			return;
		}

		if(clipping && button == GLUT_LEFT_BUTTON &&state == GLUT_UP)
		{
			int newx = min(windowSize.x, windowSize.z);
			int newy = min(windowSize.y, windowSize.w);

			int newz = max(windowSize.x, windowSize.z);
			int neww = max(windowSize.y, windowSize.w);

			windowSize = glm::vec4(newx,newy, newz, neww);

			SetScissorWindow(windowSize);
			clipping = false;
			return;
		}

		if(button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
		{
			currentX = x;
			currentY = y;
			dragging = true;
		}
		if(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP)
		{
			dragging = false;
		}
		rotating = false;
	}

	void motion(int x, int y)
	{
		if(rotating)
		{
			float rotX = (x - currentX) * rotateSpeed;
			glm::clamp(rotX, minAngle,maxAngle);
			float rotY = (y - currentY) * rotateSpeed;
			glm::clamp(rotY, minAngle,maxAngle);
				glm::mat4 rotationX(1.0),rotationY(1.0);
				rotationY = glm::rotate(rotationY,rotY, eye.right);
				eye.up = glm::vec3(rotationY * glm::vec4(eye.up,0));
				eye.view = glm::vec3(rotationY * glm::vec4(eye.view,0));
				rotationX = glm::rotate(rotationX,rotX,eye.up);
				eye.view = glm::vec3(rotationX * glm::vec4(eye.view,0));
				eye.right = glm::vec3(rotationX * glm::vec4(eye.right,0));
		}

		if(clipping)
		{
			windowSize.z = x;
			windowSize.w = y;
		}

		if(dragging)
		{
			glm::mat4 dragMatrix(1.0);
			glm::vec3 dragVector = float(x - currentX) * draggingSpeed * eye.right + float(y - currentY) * draggingSpeed * eye.up;
			 dragMatrix = glm::translate(dragMatrix,dragVector);
			 eye.position = glm::vec3(dragMatrix * glm::vec4(eye.position, 1.0f));
			 
		}
		currentX = x;
		currentY = y;
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
  dptr=NULL;
  cudaGLMapBufferObject((void**)&dptr, pbo);
  clearPBOpos(dptr,width,height);
  cudaGLUnmapBufferObject(pbo);
  // Clean up on program exit
  atexit(cleanupCuda);
  SetScissorWindow(glm::vec4(300,300,500,500));
  texture.mapptr = stbi_load("yoyo.jpg",&texture.width, &texture.height,&texture.depth,0);
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

void calcuatetransformationMatrix(  Camera eye, glm::vec2 resolution, float front, float back)
{
	/*glm::vec4 normal = glm::vec4(glm::normalize(glm::vec3(center - eye.position)),0);
	
	glm::vec4 Y = glm::normalize(glm::vec4(eye.up,0) - glm::dot(glm::vec4(eye.up,0),normal ) * normal);
	glm::vec4 X = glm::cross(Y, normal);*/

	/*glm::vec3 Z = glm::normalize(glm::vec3(center - eye.position));
	glm::vec3 Y = glm::normalize(eye.up - glm::dot(eye.up, Z) * Z);
	glm::vec3 X = glm::cross(Y, Z);

	glm::mat4 m;
	
	// Look At
	trans_matrix.x.x = X.x; trans_matrix.y.x = X.y; trans_matrix.z.x = X.z; trans_matrix.w.x = -1 *  eye.position.x;
	trans_matrix.x.y = Y.x; trans_matrix.y.y = Y.y; trans_matrix.z.y = Y.z; trans_matrix.w.y = -1 *  eye.position.y;
	trans_matrix.x.z = Z.x; trans_matrix.y.z = Z.y; trans_matrix.z.z = Z.z; trans_matrix.w.z = -1 *  eye.position.z;
	trans_matrix.x.w = 0;   trans_matrix.y.w = 0;   trans_matrix.z.w = 0;   trans_matrix.w.w = 1;     

	//Perspective ViewPort Transform
	glm::vec2 w_start;
	glm::vec2 w_end;
	cudaMat4 viewport_trans;
	viewport_trans.x = glm::vec4(2.0f * view_plane / far / resolution.x, 0,0,0);
	viewport_trans.y = glm::vec4(0, 2.0f * view_plane / far / resolution.y,0,0);
	viewport_trans.z = glm::vec4((2.0f * glm::dot(X,Z) - (w_end.x + w_start.x)) / ((w_end.x - w_start.x)*far),
		                         (2.0f * glm::dot(Y,Z) - (w_end.y + w_start.y)) / ((w_end.x - w_start.x)*far),
		                          1.0f / far, 
								  0);

	viewport_trans.w = glm::vec4(0,0,0,1);

	trans_Matrix = multiplyMV(view_port_trans, trans_Matrix);*/

	glm::vec3 Z = -1.0f * eye.view;
	glm::vec3 Y = glm::normalize(eye.up - glm::dot(eye.up, Z) * Z);
	glm::vec3 X = glm::normalize(glm::cross(1.0f * Y, Z));

	//Look At
	glm::mat4 lookatMatrix;
	/*lookatMatrix[0][0] = X.x; lookatMatrix[0][1] = X.y; lookatMatrix[0][2] = X.z; lookatMatrix[0][3] = -1 * eye.position.x;
	lookatMatrix[1][0] = Y.x; lookatMatrix[1][1] = Y.y; lookatMatrix[ 1][2] = Y.z; lookatMatrix[1][3] = -1 * eye.position.y;
	lookatMatrix[2][0] = Z.x; lookatMatrix[2][1] = Z.y; lookatMatrix[2][2] = Z.z; lookatMatrix[2][3] = -1 * eye.position.z;
	lookatMatrix[3][0] = 0;   lookatMatrix[3][1] = 0;   lookatMatrix[3][2] = 0;   lookatMatrix[3][3] = 1;*/

	lookatMatrix[0][0] = X.x;                     lookatMatrix[0][1] = Y.x;       lookatMatrix[0][2] = Z.x;      lookatMatrix[0][3] = 0;
	lookatMatrix[1][0] = X.y;                     lookatMatrix[1][1] = Y.y;       lookatMatrix[1][2] = Z.y;      lookatMatrix[1][3] = 0;
	lookatMatrix[1][0] = X.z;                     lookatMatrix[2][1] = Y.z;       lookatMatrix[2][2] = Z.z;      lookatMatrix[2][3] = 0;
	lookatMatrix[1][0] = -1 * eye.position.x;     lookatMatrix[3][1] = -1 * eye.position.y;         lookatMatrix[3][2] = -1 * eye.position.z;        lookatMatrix[3][3] = 1;




	float aspectRatio = resolution.x / resolution.y;
	float inverseTanFov = 1.0f / tan((eye.fov * PI/ 180.0f));
	

	glm::mat4 viewTrans;

/*	viewTrans[0][0] = inverseTanFov / aspectRatio;    viewTrans[0][1] = 0;              viewTrans[0][2] = 0;                             viewTrans[0][3] = 0;
	viewTrans[1][0] = 0;                              viewTrans[1][1] = inverseTanFov;  viewTrans[1][2] = 0;                             viewTrans[1][3] = 0;
	viewTrans[2][0] = 0;                              viewTrans[2][1] = 0;              viewTrans[2][2] = (front + back)/(front - back); viewTrans[2][3] = 2 * front * back / (front - back);
	viewTrans[3][0] = 0;                              viewTrans[3][1] = 0;              viewTrans[3][2] = -1;                            viewTrans[3][3] = 0;*/

	viewTrans[0][0] = inverseTanFov / aspectRatio;    viewTrans[0][1] = 0;              viewTrans[0][2] = 0;                                           viewTrans[0][3] = 0;
	viewTrans[1][0] = 0;                              viewTrans[1][1] = inverseTanFov;  viewTrans[1][2] = 0;                                           viewTrans[1][3] = 0;
	viewTrans[2][0] = 0;                              viewTrans[2][1] = 0;              viewTrans[2][2] = (front + back)/(front - back);               viewTrans[2][3] = -1;
	viewTrans[3][0] = 0;                              viewTrans[3][1] = 0;              viewTrans[3][2] = 2 * front * back / (front - back);           viewTrans[3][3] = 0;
	

	setProjectionMatrix(viewTrans);
	setViewMatrix(lookatMatrix);

	//return viewTrans * lookatMatrix;
	//return glm::mat4(1.0);
	//Look At
	//return lookatMatrix; 

}