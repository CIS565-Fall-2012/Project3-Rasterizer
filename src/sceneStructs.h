#ifndef SCENESTRUCTS_H
#define SCENESTRUCTS_H

struct triangle {
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;
	glm::vec3 c0;
	glm::vec3 c1;
	glm::vec3 c2;
};

struct fragment {
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec3 position;
};

struct Light {
	glm::vec3 color;
	glm::vec3 position;
};

struct Eye {
	glm::vec3 position;
	glm::vec3 up;
	float fovy; // in degrees
};

#endif // end of SCENESTRUCTS_H