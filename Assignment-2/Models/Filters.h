#pragma once
#include <opencv2/opencv.hpp>

enum FilterEnum { F_NONE = 0, F_PIXELATE = 1, F_SINCITY = 2, F_COMIC = 3, };

class FilterModel {
public:
	static void pixelateCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR, int block);
	static void sinCityCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR, int /*unused*/, int preserveHue = 0);
	static void comicCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR);
};