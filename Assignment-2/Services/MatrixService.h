#pragma once

// === Small 3x3 matrix helper for UV-space affine ===
struct Mat3 {
	float a11 = 1, a12 = 0, a13 = 0;
	float a21 = 0, a22 = 1, a23 = 0;
	float a31 = 0, a32 = 0, a33 = 1;
};

class MatrixService {
public:
	static Mat3 matMul(const Mat3& A, const Mat3& B);
	static Mat3 matInv(const Mat3& m);
	static Mat3 T(float tx, float ty);
	static Mat3 S(float s);
	static Mat3 R(float ang);
};
