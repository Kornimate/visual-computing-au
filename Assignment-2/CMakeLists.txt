cmake_minimum_required(VERSION 3.16)
project(Assignment2)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
CACHE STRING "Vcpkg toolchain file")
find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
add_executable(Assignment2 main.cpp  "Services/MatrixService.h" "Services/MatrixService.cpp" "Views/Window.h" "Views/Window.cpp" "Services/TransformService.h" "Services/TransformService.cpp" "Views/Shader.cpp" "Views/Shader.h"  "Models/Filters.h" "Models/Filters.cpp" "Services/LoggerService.h" "Services/LoggerService.cpp")
target_link_libraries(Assignment2 PRIVATE
    OpenGL::GL
    glfw
    glad::glad
    glm::glm
    ${OpenCV_LIBS}
)
