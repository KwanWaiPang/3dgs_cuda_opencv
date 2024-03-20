#pragma once

#pragma diag_suppress code_of_warning
#include <Eigen/Dense>
#pragma diag_default code_of_warning
#include <memory>
#include <string>
#include <torch/torch.h>
#include <vector>

class Camera : torch::nn::Module {
public:
    Camera(int imported_colmap_id,
           Eigen::Matrix3f R, Eigen::Vector3f T,//相机的旋转矩阵和平移向量（外参）
           float FoVx, float FoVy,//相机的内参
           torch::Tensor image,//转换为tensor格式的图像数据
           std::string image_name,
           int image_id,//图像id，读入的第几张图片
           float scale = 1.f);
    // Getters
    int Get_uid() const { return _uid; }
    int Get_colmap_id() const { return _colmap_id; }
    Eigen::Matrix3f& Get_R() { return _R; }
    Eigen::Vector3f& Get_T() { return _T; }
    float Get_FoVx() const { return static_cast<float>(_FoVx); }
    float Get_FoVy() const { return static_cast<float>(_FoVy); }
    std::string Get_image_name() const { return _image_name; }
    const torch::Tensor& Get_original_image() { return _original_image; }
    int Get_image_width() const { return _image_width; }
    int Get_image_height() const { return _image_height; }
    float Get_zfar() const { return _zfar; }
    float Get_znear() const { return _znear; }
    torch::Tensor& Get_world_view_transform() { return _world_view_transform; }
    torch::Tensor& Get_projection_matrix() { return _projection_matrix; }
    torch::Tensor& Get_full_proj_transform() { return _full_proj_transform; }
    torch::Tensor& Get_camera_center() { return _camera_center; }

private:
    int _uid;
    int _colmap_id;
    Eigen::Matrix3f _R; // rotation  matrix
    Eigen::Vector3f _T; // translation vector
    float _FoVx;
    float _FoVy;
    std::string _image_name;
    torch::Tensor _original_image;//原始的图像数据
    int _image_width;
    int _image_height;
    float _zfar;
    float _znear;
    torch::Tensor _trans;
    float _scale;
    torch::Tensor _world_view_transform;//世界坐标系到相机坐标系的变换矩阵（位姿）
    torch::Tensor _projection_matrix;//投影矩阵（相机坐标系到图像平面）
    torch::Tensor _full_proj_transform;//世界坐标系到图像平面的变换矩阵
    torch::Tensor _camera_center;//相机中心在世界坐标系下的坐标
};

struct CameraInfo;
namespace gs::param {
    struct ModelParameters;
}
struct ModelParameters;
Camera loadCam(const gs::param::ModelParameters& params, int id, CameraInfo& cam_info);
