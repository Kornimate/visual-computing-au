#include "TransformService.h"
#include "MatrixService.h"
#include <opencv2/opencv.hpp>

Mat3 TransformService::buildUvForward(const AppState& st) { // in UV space
	Mat3 M = MatrixService::T(st.tx, st.ty);
	M = MatrixService::matMul(M, MatrixService::T(0.5f, 0.5f)); // Center about (0.5,0.5)
	M = MatrixService::matMul(M, MatrixService::R(st.rot)); // apply S
	M = MatrixService::matMul(M, MatrixService::S(st.scale)); // apply R
	M = MatrixService::matMul(M, MatrixService::T(-0.5f, -0.5f)); // apply T
	return M;
}

// Convert forward UV transform to OpenCV inverse 2x3 in pixel coords
void TransformService::uvForward_to_cvInverse2x3(const AppState& st, int camW, int camH, cv::Mat& cv2x3Inv)
{
	// UV<->PX scale matrices
	Mat3 Suv2px; Suv2px.a11 = (float)camW; Suv2px.a22 = (float)camH;
	Mat3 Spx2uv; Spx2uv.a11 = 1.0f / (float)camW; Spx2uv.a22 = 1.0f / (float)camH;

	Mat3 Fuv = TransformService::buildUvForward(st);               // src(UV) -> dst(UV)
	Mat3 Fpx = MatrixService::matMul(Suv2px, MatrixService::matMul(Fuv, Spx2uv)); // src(px) -> dst(px)
	Mat3 Finv = MatrixService::matInv(Fpx);                     // dst(px) -> src(px)

	cv2x3Inv = (cv::Mat_<float>(2, 3) <<
		Finv.a11, Finv.a12, Finv.a13,
		Finv.a21, Finv.a22, Finv.a23);
}