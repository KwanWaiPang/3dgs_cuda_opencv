// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#include "parameters.cuh"
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace gs {
    namespace param {
        OptimizationParameters read_optim_params_from_json() {//从json文件中读取优化参数

            // automatically get the root path of the project
            std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");//存储了当前正在运行的可执行文件的绝对路径。
            std::filesystem::path parentDir = executablePath.parent_path().parent_path();//存储了执行可执行文件的父目录的父目录的路径
            std::filesystem::path json_path = parentDir / "parameter/optimization_params.json";//json文件路径
            // Check if the file exists before trying to open it
            if (!std::filesystem::exists(json_path)) {
                throw std::runtime_error("Error: " + json_path.string() + " does not exist!");
            }

            std::ifstream file(json_path);//打开json文件
            if (!file.is_open()) {
                throw std::runtime_error("OptimizationParameter file could not be opened.");
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string jsonString = buffer.str();
            file.close(); // Explicitly 显式 close the file

            // Parse the JSON string （解析JSON字符串）
            nlohmann::json json = nlohmann::json::parse(jsonString);

            OptimizationParameters params;
            params.iterations = json["iterations"];//迭代次数
            params.position_lr_init = json["position_lr_init"];
            params.position_lr_final = json["position_lr_final"];
            params.position_lr_delay_mult = json["position_lr_delay_mult"];
            params.position_lr_max_steps = json["position_lr_max_steps"];
            params.feature_lr = json["feature_lr"];
            params.percent_dense = json["percent_dense"];
            params.opacity_lr = json["opacity_lr"];
            params.scaling_lr = json["scaling_lr"];
            params.rotation_lr = json["rotation_lr"];
            params.lambda_dssim = json["lambda_dssim"];
            params.min_opacity = json["min_opacity"];
            params.densification_interval = json["densification_interval"];
            params.opacity_reset_interval = json["opacity_reset_interval"];
            params.densify_from_iter = json["densify_from_iter"];
            params.densify_until_iter = json["densify_until_iter"];
            params.densify_grad_threshold = json["densify_grad_threshold"];
            params.early_stopping = json["early_stopping"];
            params.convergence_threshold = json["convergence_threshold"];
            params.empty_gpu_cache = json["empty_gpu_cache"];

            return params;
        }
    } // namespace param
} // namespace gs
