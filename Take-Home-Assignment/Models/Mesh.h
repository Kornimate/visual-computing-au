#pragma once
#include <vector>

struct Mesh {
    std::vector<float> vertices;        // x, y, z
    std::vector<unsigned int> indices;  // triangle indices
};