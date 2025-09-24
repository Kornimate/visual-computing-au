#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

static void runFeatureDetection(const Mat& img1, const Mat& img2, Ptr<Feature2D> detector, string name, NormTypes norm) {

	vector<KeyPoint> kp1, kp2;
	Mat desc1, desc2;

	detector->detectAndCompute(img1, noArray(), kp1, desc1);
	detector->detectAndCompute(img2, noArray(), kp2, desc2);

	cout << name << " Key points in Image1: " << kp1.size() << ", Image2: " << kp2.size() << endl;

	BFMatcher matcher(norm);

	vector<DMatch> matches;

	long t0 = getTickCount();

	matcher.match(desc1, desc2, matches);
	
	long t1 = getTickCount();

	double elapsed_ms = (t1 - t0) * 1000.0 / getTickFrequency();

	cout << name << " Matching time: " << elapsed_ms << " ms" << endl;

	vector<float> distances;

	for (auto& m : matches) distances.push_back(m.distance);

	Mat distMat(distances);
	int channels[] = { 0 };
	int histSize = 50;
	float rangeArr[] = { 0.0f, 256.0f };
	const float* ranges[] = { rangeArr };
	Mat hist;

	calcHist(&distMat, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);

	int hist_w = 400;
	int hist_h = 300;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX);

	for (int i = 1; i < histSize; i++) {
		line(histImage,
			 Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			 Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(0))),
			 Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow(name + " Match Distance Histogram", histImage);

	//matches
	Mat img_matches;
	drawMatches(img1, kp1, img2, kp2, matches, img_matches);
	imshow(name + " Matches", img_matches);
}

int main() {

	//import images
	Mat img1 = imread("./ImageSource/outdoor-s1-1.jpg", IMREAD_GRAYSCALE);
	Mat img2 = imread("./ImageSource/outdoor-s1-2.jpg", IMREAD_GRAYSCALE);

	//check if import is successful
	if (img1.empty() || img2.empty()) {
		cout << "Error: Could not load images!" << endl;
		return -1;
	}

	//smaller images for ability to show it and 
	Mat smallImg1;
	resize(img1, smallImg1, Size(), 0.15, 0.15);

	Mat smallImg2;
	resize(img2, smallImg2, Size(), 0.15, 0.15);

	// SIFT
	Ptr<Feature2D> sift = SIFT::create();
	runFeatureDetection(smallImg1, smallImg2, sift, "[SIFT]", NORM_L2);

	// AKAZE
	Ptr<Feature2D> akaze = AKAZE::create();
	runFeatureDetection(smallImg1, smallImg2, akaze, "[AKAZE]", NORM_HAMMING);

	waitKey(0);
	return 0;
}

