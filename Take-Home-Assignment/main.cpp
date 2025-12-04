#include "app.h"
#include "Views/window2D.h"
#include "Views/window3D.h"

int main() {
	srand((unsigned)time(nullptr));
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	App* app = new App(new Window2D(), new Window3D());

	app->initialize();
	app->run();


	delete app;
	return 0;
}