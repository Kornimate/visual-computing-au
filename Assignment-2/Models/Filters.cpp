#include "Filters.h"

void FilterModel::pixelateCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR, int block) {
	block = std::max(1, block);
	int smallW = std::max(1, srcBGR.cols / block);
	int smallH = std::max(1, srcBGR.rows / block);
	cv::resize(srcBGR, dstBGR, cv::Size(smallW, smallH), 0, 0, cv::INTER_NEAREST);
	cv::resize(dstBGR, dstBGR, srcBGR.size(), 0, 0, cv::INTER_NEAREST);
}


void FilterModel::sinCityCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR, int /*unused*/, int preserveHue) {
	(void)preserveHue;
	// Keep REDs, grayscale others.
	cv::Mat hsv, gray, mask1, mask2, mask;
	cv::cvtColor(srcBGR, hsv, cv::COLOR_BGR2HSV);
	// Red wraps around hue (0..179 in OpenCV)
	cv::inRange(hsv, cv::Scalar(0, 80, 60), cv::Scalar(10, 255, 255), mask1);
	cv::inRange(hsv, cv::Scalar(170, 80, 60), cv::Scalar(180, 255, 255), mask2);
	cv::bitwise_or(mask1, mask2, mask);

	cv::cvtColor(srcBGR, gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(gray, dstBGR, cv::COLOR_GRAY2BGR);
	srcBGR.copyTo(dstBGR, mask); // copy colored pixels where mask is true
}

void FilterModel::comicCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR) {
	cv::Mat gray, edges;
	cv::cvtColor(srcBGR, gray, cv::COLOR_BGR2GRAY);
	cv::medianBlur(gray, gray, 7);
	cv::Laplacian(gray, edges, CV_8U, 5);
	cv::threshold(edges, edges, 80, 255, cv::THRESH_BINARY_INV);
	cv::Mat quant;
	cv::pyrMeanShiftFiltering(srcBGR, quant, 8, 16);
	cv::bitwise_and(quant, quant, dstBGR, edges);
}