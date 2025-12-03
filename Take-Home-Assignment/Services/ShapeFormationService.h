#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "../Models/DetectedShape.h"
#include "../Models/Mesh.h"

class ShapeFormationService {
public:
	static std::vector<cv::Point> ShapeFormationService::getRawDrawnStroke(const cv::Mat& img);
	static DetectedShape detectShapeWithPolygon(const cv::Mat& img);
	static Mesh extrudeY(const std::vector<cv::Point>& poly, float height);
	static Mesh revolvePolygon(const std::vector<cv::Point>& raw, char axis, int segments);
};