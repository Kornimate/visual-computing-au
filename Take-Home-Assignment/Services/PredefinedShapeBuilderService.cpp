#include "PredefinedShapeBuilderService.h"
#include <cmath>
#include <opencv2/opencv.hpp>

Mesh PredefinedShapeBuilderService::createBoxMesh(float sx, float sy, float sz) {
    Mesh m;
    float x = sx * 0.5f;
    float y = sy * 0.5f;
    float z = sz * 0.5f;

    float v[] = {
        -x, -y, -z,
         x, -y, -z,
         x,  y, -z,
        -x,  y, -z,
        -x, -y,  z,
         x, -y,  z,
         x,  y,  z,
        -x,  y,  z
    };
    m.vertices.assign(v, v + 8 * 3);

    unsigned int idx[] = {
        // front (z+)
        4,5,6,  6,7,4,
        // back (z-)
        1,0,3,  3,2,1,
        // left (x-)
        0,4,7,  7,3,0,
        // right (x+)
        5,1,2,  2,6,5,
        // top (y+)
        3,7,6,  6,2,3,
        // bottom (y-)
        0,1,5,  5,4,0
    };
    m.indices.assign(idx, idx + 36);
    return m;
}

Mesh PredefinedShapeBuilderService::createPyramidMesh(float baseSize, float height) {
    Mesh m;
    float h = height;
    float b = baseSize * 0.5f;

    float yBase = -h * 0.5f;
    float yTop = +h * 0.5f;

    std::vector<float> v = {
        -b, yBase, -b,
         b, yBase, -b,
         0, yBase,  b,
         0, yTop,   0
    };
    m.vertices = v;

    // Base
    m.indices.push_back(0);
    m.indices.push_back(1);
    m.indices.push_back(2);

    // Sides
    m.indices.push_back(0);
    m.indices.push_back(1);
    m.indices.push_back(3);

    m.indices.push_back(1);
    m.indices.push_back(2);
    m.indices.push_back(3);

    m.indices.push_back(2);
    m.indices.push_back(0);
    m.indices.push_back(3);

    return m;
}

Mesh PredefinedShapeBuilderService::createSphereMesh(float radius, int slices, int stacks) {
    Mesh m;
    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * CV_PI;
        float y = std::cos(phi);
        float r = std::sin(phi);
        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * 2.0f * CV_PI;
            float x = r * std::cos(theta);
            float z = r * std::sin(theta);
            m.vertices.push_back(radius * x);
            m.vertices.push_back(radius * y);
            m.vertices.push_back(radius * z);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int i0 = i * (slices + 1) + j;
            int i1 = i0 + 1;
            int i2 = i0 + (slices + 1);
            int i3 = i2 + 1;
            m.indices.push_back(i0);
            m.indices.push_back(i2);
            m.indices.push_back(i1);
            m.indices.push_back(i1);
            m.indices.push_back(i2);
            m.indices.push_back(i3);
        }
    }
    return m;
}

Mesh PredefinedShapeBuilderService::createPentagonPrismMesh(float radius, float height) {
    Mesh m;
    int n = 5;
    float h = height * 0.5f;

    // Bottom ring
    for (int i = 0; i < n; ++i) {
        float a = (2.0f * CV_PI * i) / n;
        float x = radius * std::cos(a);
        float z = radius * std::sin(a);
        m.vertices.push_back(x);
        m.vertices.push_back(-h);
        m.vertices.push_back(z);
    }

    // Top ring
    for (int i = 0; i < n; ++i) {
        float a = (2.0f * CV_PI * i) / n;
        float x = radius * std::cos(a);
        float z = radius * std::sin(a);
        m.vertices.push_back(x);
        m.vertices.push_back(+h);
        m.vertices.push_back(z);
    }

    // Bottom face
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(0);
        m.indices.push_back(i + 1);
        m.indices.push_back(i);
    }

    // Top face
    int off = n;
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(off);
        m.indices.push_back(off + i);
        m.indices.push_back(off + i + 1);
    }

    // Walls
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        int bi = i;
        int bj = j;
        int ti = i + n;
        int tj = j + n;

        m.indices.push_back(bi);
        m.indices.push_back(ti);
        m.indices.push_back(bj);
        m.indices.push_back(bj);
        m.indices.push_back(ti);
        m.indices.push_back(tj);
    }

    return m;
}