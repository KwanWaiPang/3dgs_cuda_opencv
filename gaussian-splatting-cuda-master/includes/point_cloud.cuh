#pragma once

#include <vector>

struct Point {
    float x;
    float y;
    float z;
};

struct Normal {
    float x;
    float y;
    float z;
};

struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct PointCloud {
    std::vector<Point> _points;//点的位置
    std::vector<Normal> _normals;//点的法向
    std::vector<Color> _colors;//点的颜色
};