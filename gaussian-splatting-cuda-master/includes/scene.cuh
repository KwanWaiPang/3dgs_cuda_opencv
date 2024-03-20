#pragma once

#include "scene_info.cuh"
#include <memory>
#include <vector>

class CameraInfo;
class GaussianModel;
class Camera;
namespace gs::param {
    struct ModelParameters;
}
struct ModelParameters;

class Scene {
public:
    Scene(GaussianModel& gaussians, const gs::param::ModelParameters& params);
    [[nodiscard]] int Get_camera_count() const { return _cameras.size(); }
    Camera& Get_training_camera(int i) { return _cameras[i]; }
    [[nodiscard]] float Get_cameras_extent() const { return static_cast<float>(_scene_infos->_nerf_norm_radius); }

private:
    GaussianModel& _gaussians;//高斯模型
    const gs::param::ModelParameters& _params;//模型参数
    std::vector<Camera> _cameras;//图像数据
    std::unique_ptr<SceneInfo> _scene_infos;//场景的信息
};