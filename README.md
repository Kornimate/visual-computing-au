# visual-computing-au
Repository for Visual Computing Course in Aarhus University

### Build the code:
0. Use Visual Studio
1. Open the actual Assignment's folder (Assignment 1/2) in VS
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
