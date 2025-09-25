#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <tuple>

using namespace cv;
using namespace std;

static Mat overlayBlend(const Mat& warped, const Mat& img2) {
	Mat result = warped.clone();
	for (int y = 0; y < img2.rows; y++) {
		for (int x = 0; x < img2.cols; x++) {
			if (img2.at<Vec3b>(y, x) != Vec3b(0, 0, 0)) { // non-black pixels
				result.at<Vec3b>(y, x) = img2.at<Vec3b>(y, x);
			}
		}
	}
	return result;
}

static Mat featherBlend(const Mat& warped, const Mat& img2) {
	Mat result = warped.clone();
	for (int y = 0; y < img2.rows; y++) {
		for (int x = 0; x < img2.cols; x++) {
			Vec3b c1 = warped.at<Vec3b>(y, x);
			Vec3b c2 = img2.at<Vec3b>(y, x);
			if (c2 != Vec3b(0, 0, 0)) { // overlapping
				// weighted average
				for (int i = 0; i < 3; i++)
					result.at<Vec3b>(y, x)[i] = uchar(0.5 * c1[i] + 0.5 * c2[i]);
			}
		}
	}
	return result;
}

static tuple<vector<DMatch>, vector<KeyPoint>, vector<KeyPoint>> detectAndMatch(const Mat& img1, const Mat& img2, Ptr<Feature2D> detector, string name, NormTypes norm, bool drawMatchesAsWindow = false) {

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

	// Draw matches
	if (drawMatchesAsWindow)
	{
		Mat img_matches;
		drawMatches(img1, kp1, img2, kp2, matches, img_matches);
		imshow(name + " Matches", img_matches);
	}

	return make_tuple(matches, kp1, kp2);
}

static Mat stitchUsingRANSAC(const Mat& img1, const Mat& img2, vector<DMatch> matches, vector<KeyPoint> kp1, vector<KeyPoint> kp2, double ransacThresh) {

	vector<Point2f> pts1, pts2;
	for (auto& m : matches) {
		pts1.push_back(kp1[m.queryIdx].pt);
		pts2.push_back(kp2[m.trainIdx].pt);
	}

	Mat inlierMask;
	TickMeter tm;

	tm.start();

	Mat H = findHomography(pts1, pts2, RANSAC, ransacThresh, inlierMask);

	tm.stop();

	int inliers = countNonZero(inlierMask);
	cout << "RANSAC threshold = " << ransacThresh << ", inliers = " << inliers << ", time = " << tm.getTimeMilli() << " ms" << endl;

	Mat warped;
	warpPerspective(img1, warped, H, Size(img1.cols + img2.cols, max(img1.rows, img2.rows)));
	Mat result(warped, Rect(0, 0, img2.cols, img2.rows));
	img2.copyTo(result);

	imshow("Stitched Image", warped);

	return warped;
}

static void useFeatureDetection(vector<DMatch> matches, string name) {

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
}

int main() {

	// Import images
	Mat img1 = imread("./ImageSource/indoor-s1-1.jpg", IMREAD_COLOR_BGR);
	Mat img2 = imread("./ImageSource/indoor-s1-2.jpg", IMREAD_COLOR_BGR);

	// Check if import is successful
	if (img1.empty() || img2.empty()) {
		cout << "Error: Could not load images!" << endl;
		return -1;
	}

	// Smaller images for ability to show it and 
	Mat smallImg1;
	resize(img1, smallImg1, Size(), 0.15, 0.15);

	Mat smallImg2;
	resize(img2, smallImg2, Size(), 0.15, 0.15);

	// Matches: SIFT and AKAZE
	Ptr<Feature2D> sift = SIFT::create();
	string siftName = "[SIFT]";
	vector<DMatch> matchesSift;
	vector<KeyPoint> kp1;
	vector<KeyPoint> kp2;

	tie(matchesSift, kp1, kp2) = detectAndMatch(smallImg1, smallImg2, sift, siftName, NORM_L2); //add true param to show matches

	// SIFT
	useFeatureDetection(matchesSift, siftName);

	Ptr<Feature2D> akaze = AKAZE::create();
	string akazeName = "[AKAZE]";
	vector<DMatch> matchesAkaze;

	tie(matchesAkaze, ignore, ignore) = detectAndMatch(smallImg1, smallImg2, akaze, akazeName, NORM_HAMMING); //add true param to show matches

	// AKAZE
	useFeatureDetection(matchesAkaze, akazeName);

	// Homography and warping
	Mat resultOfWarping;
	vector<double> thresholds = { 1.0, 3.0, 5.0, 10.0 };

	for (double t : thresholds) {

		Mat resultOfWarping = stitchUsingRANSAC(smallImg1, smallImg2, matchesSift, kp1, kp2, 10.0);
		waitKey(0); // pause to observe visual quality
	}

	Mat overlayResult = overlayBlend(resultOfWarping, smallImg2);
	imshow("Overlay Blend", overlayResult);

	Mat featherResult = featherBlend(resultOfWarping, smallImg2);
	imshow("Feathering Blend", featherResult);

	waitKey(0);

	return 0;
}

