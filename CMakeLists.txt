cmake_minimum_required(VERSION 3.16)
project(MyProject)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_TOOLCHAIN_FILE "C:/Users/User/Documents/VCPKG/vcpkg/scripts/buildsystems/vcpkg.cmake"
CACHE STRING "Vcpkg toolchain file")
find_package(OpenCV REQUIRED)
add_executable(MyProject main.cpp)
target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS})
file(COPY ${CMAKE_SOURCE_DIR}/ImageSource DESTINATION ${CMAKE_BINARY_DIR})
