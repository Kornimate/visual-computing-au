#pragma once
#include "../Models/Mesh.h"

class PredefinedShapeBuilderSrevice {
public:
	static Mesh createBoxMesh(float sx, float sy, float sz);
	static Mesh createPyramidMesh(float baseSize, float height);
	static Mesh createSphereMesh(float radius, int slices, int stacks);
	static Mesh createPentagonPrismMesh(float radius, float height);
};