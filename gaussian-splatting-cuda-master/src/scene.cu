#include "camera.cuh"
#include "camera_utils.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "read_utils.cuh"
#include "scene.cuh"

// TODO: support start from later iterations. Compare original code
// We also have only training, no testing
// TODO: support also testing
/**
 * @brief 初始化场景，传入模型的参数，最终获取初始化后的高斯模型（进行高斯函数的初始化、获取图像数据）
 */
Scene::Scene(GaussianModel& gaussians, const gs::param::ModelParameters& params) : _gaussians(gaussians),//返回的初始化后的高斯模型
                                                                                   _params(params) //模型参数为输入参数
{
    // Right now there is only support for colmap
    if (std::filesystem::exists(_params.source_path)) {
        _scene_infos = read_colmap_scene_info(_params.source_path, _params.resolution);//获取场景的信息
    } else {
        std::cout << "Error: " << _params.source_path << " does not exist!" << std::endl;
        exit(-1);
    }

    _cameras.reserve(_scene_infos->_cameras.size());
    std::vector<nlohmann::json> json_cams;
    json_cams.reserve(_scene_infos->_cameras.size());
    int counter = 0;
    for (auto& cam_info : _scene_infos->_cameras) {
        _cameras.emplace_back(loadCam(_params, counter, cam_info));//加载相机数据，同时释放了cam_info中的图像数据（应该就是内存的管理吧）
        json_cams.push_back(Convert_camera_to_JSON(cam_info, counter, _cameras.back().Get_R(), _cameras.back().Get_T()));//将相机数据转换为json格式，获取相机的内参与外参
        ++counter;
    }
    dump_JSON(params.output_path / "cameras.json", json_cams);//再输出一次相机的数据（内外参）
    // TODO: json camera dumping for debugging purpose at least

    // get the parameterr self.cameras.extent（从点云中初始化高斯模型）
    _gaussians.Create_from_pcd(_scene_infos->_point_cloud, _scene_infos->_nerf_norm_radius);
}