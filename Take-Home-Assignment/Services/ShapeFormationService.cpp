#include "ShapeFormationService.h"
#include "../Models/Constants.h"

std::vector<cv::Point> ShapeFormationService::getRawDrawnStroke(const cv::Mat& img) {
    cv::Mat gray, blurImg, thresh;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurImg, cv::Size(5, 5), 0);
    cv::threshold(blurImg, thresh, 200, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) return {};

    // largest contour
    double best = 0.0;
    int idx = 0;
    for (int i = 0; i < (int)contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > best) {
            best = a;
            idx = i;
        }
    }
    return contours[idx]; // raw points
}

DetectedShape ShapeFormationService::detectShapeWithPolygon(const cv::Mat& img) {
    cv::Mat gray, blurImg, thresh;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurImg, cv::Size(5, 5), 0);
    cv::threshold(blurImg, thresh, 200, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return { "none", {} };
    }

    // Largest contour
    double maxArea = 0.0;
    int maxIdx = -1;
    for (int i = 0; i < (int)contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea) {
            maxArea = a;
            maxIdx = i;
        }
    }

    auto& c = contours[maxIdx];

    std::vector<cv::Point> approx;
    cv::approxPolyDP(c, approx, 0.02 * cv::arcLength(c, true), true);

    int n = (int)approx.size();
    std::string label = "polygon";

    if (n == 3) {
        label = "triangle";
    }
    else if (n == 4) {
        // Distinguish square vs rectangle using bounding box aspect ratio
        cv::Rect r = cv::boundingRect(c);
        float ar = (float)r.width / (float)r.height;
        if (ar < 1.0f) ar = 1.0f / ar; // make >= 1
        if (ar < 1.1f) label = "square";      // roughly equal sides
        else           label = "rectangle";   // more rectangular
    }
    else if (n == 5) {
        label = "pentagon";
    }
    else {
        // check for circle-like
        double peri = cv::arcLength(c, true);
        double area = cv::contourArea(c);
        double circularity = 4 * CV_PI * (area / (peri * peri));
        if (circularity > 0.75) label = "circle";
        else label = "polygon";
    }

    return { label, approx };
}

Mesh ShapeFormationService::extrudeY(const std::vector<cv::Point>& poly, float height) {
    Mesh m;
    if (poly.size() < 3) return m;

    int n = (int)poly.size();
    float half = height * 0.5f;

    // Center in pixel coordinates
    float cx = 0.0f, cy = 0.0f;
    for (auto& p : poly) {
        cx += (float)p.x;
        cy += (float)p.y;
    }
    cx /= n;
    cy /= n;

    float scale = 0.0015f;  // pixel -> world scaling

    // Bottom ring (y = -half)
    for (auto& p : poly) {
        float x = (p.x - cx) * scale;
        float z = (cy - p.y) * scale; // flip Y
        m.vertices.push_back(x);
        m.vertices.push_back(-half);
        m.vertices.push_back(z);
    }

    // Top ring (y = +half)
    for (auto& p : poly) {
        float x = (p.x - cx) * scale;
        float z = (cy - p.y) * scale;
        m.vertices.push_back(x);
        m.vertices.push_back(+half);
        m.vertices.push_back(z);
    }

    // Bottom face (triangle fan)
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(0);
        m.indices.push_back(i + 1);
        m.indices.push_back(i);
    }

    // Top face (triangle fan)
    int off = n;
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(off);
        m.indices.push_back(off + i);
        m.indices.push_back(off + i + 1);
    }

    // Side walls
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;

        int bi = i;
        int bj = j;
        int ti = i + n;
        int tj = j + n;

        // First triangle
        m.indices.push_back(bi);
        m.indices.push_back(ti);
        m.indices.push_back(bj);

        // Second triangle
        m.indices.push_back(bj);
        m.indices.push_back(ti);
        m.indices.push_back(tj);
    }

    return m;
}

Mesh ShapeFormationService::revolvePolygon(const std::vector<cv::Point>& raw, char axis, int segments) {
    Mesh m;
    if (raw.size() < 2) return m;

    float cx = AppConstants::DRAW_W * 0.5f;
    float cy = AppConstants::DRAW_H * 0.5f;
    float scale = 0.0015f;

    // convert raw points into 3D profile: (x,y,0)
    std::vector<float> px, py, pz;
    px.reserve(raw.size());
    py.reserve(raw.size());
    pz.reserve(raw.size());

    for (auto& p : raw) {
        float x = (p.x - cx) * scale;
        float y = (cy - p.y) * scale; // flip Y so up is +Y
        px.push_back(x);
        py.push_back(y);
        pz.push_back(0.0f);
    }

    int R = (int)raw.size();

    // sweep around axis
    for (int i = 0; i <= segments; ++i) {
        float ang = (2.0f * CV_PI * i) / segments;
        float c = std::cos(ang);
        float s = std::sin(ang);

        for (int k = 0; k < R; ++k) {
            float x = px[k], y = py[k], z = pz[k];
            float rx, ry, rz;

            if (axis == 'y') {
                // rotate around Y axis
                rx = x * c + z * s;
                ry = y;
                rz = -x * s + z * c;
            }
            else {
                // rotate around X axis
                rx = x;
                ry = y * c - z * s;
                rz = y * s + z * c;
            }

            m.vertices.push_back(rx);
            m.vertices.push_back(ry);
            m.vertices.push_back(rz);
        }
    }

    // connect rings into quads -> triangles
    for (int i = 0; i < segments; ++i) {
        int r0 = i * R;
        int r1 = (i + 1) * R;

        for (int k = 0; k < R - 1; ++k) {
            int a = r0 + k;
            int b = r0 + k + 1;
            int c2 = r1 + k;
            int d = r1 + k + 1;

            m.indices.push_back(a);
            m.indices.push_back(c2);
            m.indices.push_back(b);

            m.indices.push_back(b);
            m.indices.push_back(c2);
            m.indices.push_back(d);
        }
    }

    return m;
}