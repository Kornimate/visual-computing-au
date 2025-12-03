#pragma once
#include <string>
#include <vector>

struct DetectedShape {
    std::string label;              // "circle", "square", "rectangle", "triangle", "pentagon", "polygon", "none"
    std::vector<cv::Point> polygon; // approximated contour
};