#pragma once

#include "camera.cuh"
#include "point_cloud.cuh"
#include <camera_info.cuh>
#pragma diag_suppress code_of_warning
#include <Eigen/Dense>
#pragma diag_default code_of_warning
#include <filesystem>

// Also here as in camera info. I guess this can be cleaned up and removed later on
// TODO: Check and remove this class if possible
struct SceneInfo {//场景信息
    std::vector<CameraInfo> _cameras;//读取的图像
    PointCloud _point_cloud;//点云（可以来自于lidar，xyz以及intensity）
    float _nerf_norm_radius;//相机分布的半径
    Eigen::Vector3f _nerf_norm_translation;//所有相机中心点的负值（为了归一化到00吧）
    std::filesystem::path _ply_path;//点云路径
};
