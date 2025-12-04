cmake_minimum_required(VERSION 3.16)
project(Take-Home-Assignment)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
CACHE STRING "Vcpkg toolchain file")
find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
add_executable(Take-Home-Assignment main.cpp "App.cpp" "Views/Window.h" "Views/Window2D.cpp" "Views/Window3D.cpp" "App.h" "Views/Window2D.h" "Views/window3D.h" "Models/Constants.h" "Models/Mesh.h" "Models/SceneObject.h" "Models/AppState.h" "Services/ShaderService.h" "Services/ShaderService.cpp" "Services/ShapeFormationService.h" "Services/ShapeFormationService.cpp" "Services/PredefinedShapeBuilderService.h" "Services/PredefinedShapeBuilderService.cpp" "Services/GPUConvertService.h" "Services/GPUConvertService.cpp" "Services/FileService.cpp" "Services/FileService.h")
target_link_libraries(Take-Home-Assignment PRIVATE
    OpenGL::GL
    glfw
    glad::glad
    glm::glm
    ${OpenCV_LIBS}
)
file(COPY "Resources/" DESTINATION ${CMAKE_BINARY_DIR}/Resources)

