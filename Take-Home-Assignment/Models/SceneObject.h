#pragma once
#include "GLMesh.h"

struct SceneObject {
	GLMesh* mesh = nullptr;
	glm::vec3 position{ 0.0f };
	glm::vec3 rotation{ 0.0f }; // Euler angles in degrees
	float     scale = 1.0f;
	glm::vec3 color{ 1.0f, 1.0f, 1.0f }; // per-object color
};