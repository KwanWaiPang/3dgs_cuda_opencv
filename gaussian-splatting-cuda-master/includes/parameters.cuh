// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include <filesystem>

namespace gs {
    namespace param {
        struct OptimizationParameters {//优化参数
            size_t iterations = 30'000;//迭代次数
            float position_lr_init = 0.00016f;
            float position_lr_final = 0.0000016f;
            float position_lr_delay_mult = 0.01f;
            int64_t position_lr_max_steps = 30'000;
            float feature_lr = 0.0025f;
            float percent_dense = 0.01f;
            float opacity_lr = 0.05f;
            float scaling_lr = 0.001f;
            float rotation_lr = 0.001f;
            float lambda_dssim = 0.2f;
            float min_opacity = 0.005f;
            uint64_t densification_interval = 100;//致密化以及剪纸的频率
            uint64_t opacity_reset_interval = 3'000;//重新设置透明度的频率
            uint64_t densify_from_iter = 500;//这个代数之后执行致密化以及剪枝处理
            uint64_t densify_until_iter = 15'000;//在这个迭代次数以内都进行致密化
            float densify_grad_threshold = 0.0002f;
            bool early_stopping = false;
            float convergence_threshold = 0.007f;
            bool empty_gpu_cache = false;//是否清空GPU的缓存，如果是，每100代清空一次
        };

        struct ModelParameters {//模型的参数
            int sh_degree = 3;//（颜色）球谐函数的阶数
            std::filesystem::path source_path = "";//源路径（输入数据）
            std::filesystem::path output_path = "output";//输出路径
            std::string images = "images";//图片字符
            int resolution = -1;//分辨率
            bool white_background = false;//是否白色背景
            bool eval = false;//是否验证集
        };

        OptimizationParameters read_optim_params_from_json();
    } // namespace param
} // namespace gs