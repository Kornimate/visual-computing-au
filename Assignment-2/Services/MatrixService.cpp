#include "MatrixService.h"
#include <cmath>

Mat3 MatrixService::matMul(const Mat3& A, const Mat3& B) { // matrix multiplication 
	Mat3 M;
	M.a11 = A.a11 * B.a11 + A.a12 * B.a21 + A.a13 * B.a31;
	M.a12 = A.a11 * B.a12 + A.a12 * B.a22 + A.a13 * B.a32;
	M.a13 = A.a11 * B.a13 + A.a12 * B.a23 + A.a13 * B.a33;
	M.a21 = A.a21 * B.a11 + A.a22 * B.a21 + A.a23 * B.a31;
	M.a22 = A.a21 * B.a12 + A.a22 * B.a22 + A.a23 * B.a32;
	M.a23 = A.a21 * B.a13 + A.a22 * B.a23 + A.a23 * B.a33;
	M.a31 = A.a31 * B.a11 + A.a32 * B.a21 + A.a33 * B.a31;
	M.a32 = A.a31 * B.a12 + A.a32 * B.a22 + A.a33 * B.a32;
	M.a33 = A.a31 * B.a13 + A.a32 * B.a23 + A.a33 * B.a33;
	return M;
}

Mat3 MatrixService::matInv(const Mat3& m) {
	Mat3 r;
	float d =
		m.a11 * (m.a22 * m.a33 - m.a23 * m.a32) -
		m.a12 * (m.a21 * m.a33 - m.a23 * m.a31) +
		m.a13 * (m.a21 * m.a32 - m.a22 * m.a31);
	if (std::fabs(d) < 1e-8f) return Mat3(); // identity fallback
	float inv = 1.0f / d;
	r.a11 = (m.a22 * m.a33 - m.a23 * m.a32) * inv;
	r.a12 = -(m.a12 * m.a33 - m.a13 * m.a32) * inv;
	r.a13 = (m.a12 * m.a23 - m.a13 * m.a22) * inv;
	r.a21 = -(m.a21 * m.a33 - m.a23 * m.a31) * inv;
	r.a22 = (m.a11 * m.a33 - m.a13 * m.a31) * inv;
	r.a23 = -(m.a11 * m.a23 - m.a13 * m.a21) * inv;
	r.a31 = (m.a21 * m.a32 - m.a22 * m.a31) * inv;
	r.a32 = -(m.a11 * m.a32 - m.a12 * m.a31) * inv;
	r.a33 = (m.a11 * m.a22 - m.a12 * m.a21) * inv;
	return r;
}

Mat3 MatrixService::T(float tx, float ty) { Mat3 m; m.a13 = tx; m.a23 = ty; return m; }
Mat3 MatrixService::S(float s) { Mat3 m; m.a11 = s; m.a22 = s; return m; }
Mat3 MatrixService::R(float ang) {
	Mat3 m; float c = std::cos(ang), s = std::sin(ang);
	m.a11 = c; m.a12 = -s; m.a21 = s; m.a22 = c; return m;
}