#pragma once
#include "../Models/GLMesh.h"
#include "../Models/Mesh.h"

class GPUConvertService {
public:
	static GLMesh uploadMesh(const Mesh& m);
};