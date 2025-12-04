#pragma once
#include "Models/GLMesh.h"
#include "Models/SceneObject.h"
#include "Views/window2D.h"
#include "Views/window3D.h"

class App {
private:
	Window* _win2d;
	Window* _win3d;

public:
	App(Window* win2d, Window* win3d);
	~App();
	void initialize();
	void run();

	static AppState g_state;

	static GLMesh gCubeMesh;
	static GLMesh gCuboidMesh;
	static GLMesh gSphereMesh;
	static GLMesh gPyramidMesh;
	static GLMesh gPentagonPrismMesh;
	static std::vector<GLMesh> gDynamicMeshes;   // meshes created from extrusion / revolution
	static std::vector<SceneObject> gObjects;    // all objects in 3D scene
};