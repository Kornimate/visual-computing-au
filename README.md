# <p align=center>Visual Computing AU</p>
*<p align=center>Repository for Visual Computing Course in Aarhus University</p>*

### Assignment 2: Build the code for Release and Debug versions as well
0. If you have built it for running only (next paragraph after this) the delete the out folder, make sure cmake and externa libraries (`opencv:x64-windows glfw3:x64-windows glad:x64-windows glm:x64-windows`) are installed
1. Go to the Assignment 2 folder and fill out the missing parts of the command (`<your path to vcpkg>`) and run the following command: `cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE="<your path to vcpkg>/scripts/buildsystems/vcpkg.cmake"
`
2. Go to the created build folder and open the Assignment2.sln in Visual Studio
3. Set Assigment2 as Startup project
4. Select the build configuration as you wish (Release/Debug) and run the code


### Assignment 2: Build the code just for running: (Debug only)
0. Use Visual Studio, if you have built it for release as well then nothing to do, as that has debug too
1. Open the actual Assignment's folder (Assignment 2) in VS
2. In CMakeLists.txt change the `set(CMAKE_TOOLCHAIN_FILE "<your vcpkg location>")` to your actual location of vcpkg
3. Build and Run the project using VS

### Assignment 2 usage:
 - Usage:
   - Filters:
     - [Key 1] None 
     - [Key 2] Pixelate   
     - [Key 3] SinCity   
     - [Key 4] Comic  
   - Runtime:
     - [Key G] GPU path
     - [Key C] CPU path
   - Custom controls for filters:
     - [Key Down Arrow] decrease pixel block size
     - [Key Up Arrow] increase pixel block size
   - Transforms:
     - [Mouse Left-drag] translate (pan)
     - [Mouse Right-drag] OR [Key Shift+ Mouse Left-drag] rotate
     - [Mouse wheel] zoom
     - [Key R] reset transform
     - [Key Esc] Quit

 *this assignment as made using Windows as OS, and the following vcpkg command was used to download the external libraries: <br>
 `vcpkg install opencv:x64-windows glfw3:x64-windows glad:x64-windows glm:x64-windows`*

### Assignment 1: Build the code just for running: (Debug only)
0. Use Visual Studio
1. Open the actual Assignment's folder (Assignment 1) in VS
2. In CMakeLists.txt change the `set(CMAKE_TOOLCHAIN_FILE "<your vcpkg location>")` to your actual location of vcpkg
3. Build and Run the project using VS

