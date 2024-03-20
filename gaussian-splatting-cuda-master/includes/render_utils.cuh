// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "camera.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "rasterizer.cuh"
#include "sh_utils.cuh"
#include <cmath>
#include <torch/torch.h>

/**
 * @brief 进行渲染，主要是通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
 * @param viewpoint_camera 当前相机对象
 * @param gaussianModel 高斯模型
 * @param bg_color 背景颜色
 * @return 渲染后的图片、高斯模型的2D坐标、高斯模型的可见性、高斯模型的半径
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render(Camera& viewpoint_camera,//当前相机对象
                                                                                     GaussianModel& gaussianModel,//高斯模型
                                                                                     torch::Tensor& bg_color,//背景颜色
                                                                                     float scaling_modifier = 1.0,
                                                                                     torch::Tensor override_color = torch::empty({})) {
    // Ensure background tensor (bg_color) is on GPU!
    bg_color = bg_color.to(torch::kCUDA);

    // Set up rasterization configuration（设置光栅化的参数）
    GaussianRasterizationSettings raster_settings = {
        .image_height = static_cast<int>(viewpoint_camera.Get_image_height()),
        .image_width = static_cast<int>(viewpoint_camera.Get_image_width()),
        .tanfovx = std::tan(viewpoint_camera.Get_FoVx() * 0.5f),
        .tanfovy = std::tan(viewpoint_camera.Get_FoVy() * 0.5f),
        .bg = bg_color,
        .scale_modifier = scaling_modifier,
        .viewmatrix = viewpoint_camera.Get_world_view_transform(),
        .projmatrix = viewpoint_camera.Get_full_proj_transform(),
        .sh_degree = gaussianModel.Get_active_sh_degree(),////sh的阶数，初始化为0，最大是3（_max_sh_degree/sh_degree）
        .camera_center = viewpoint_camera.Get_camera_center(),////相机中心在世界坐标系下的坐标
        .prefiltered = false};

    //利用上面的setting来初始化一个光栅化器
    GaussianRasterizer rasterizer = GaussianRasterizer(raster_settings);

    auto means3D = gaussianModel.Get_xyz();//获取所有高斯模型的坐标
    auto means2D = torch::zeros_like(gaussianModel.Get_xyz()).requires_grad_(true);//与高斯模型的坐标张量相同形状的张量，初始化为0，需要梯度
    means2D.retain_grad();//张量标记为保留梯度。这意味着在计算梯度时，PyTorch 将保留并计算与 means2D 相关的梯度信息，即使在反向传播过程中被使用了多次也会保留梯度。
    auto opacity = gaussianModel.Get_opacity();//获取所有高斯模型的不透明度

    auto scales = torch::Tensor();
    auto rotations = torch::Tensor();
    auto cov3D_precomp = torch::Tensor();

    scales = gaussianModel.Get_scaling();//获取所有高斯模型的缩放
    rotations = gaussianModel.Get_rotation();//获取所有高斯模型的旋转

    auto shs = torch::Tensor();
    torch::Tensor colors_precomp = torch::Tensor();
    // This is nonsense. Background color not used? See orginal file colors_precomp=None line 70
    shs = gaussianModel.Get_features();//获取所有高斯模型的sh系数（颜色）

    torch::cuda::synchronize();//调用了CUDA的同步函数，用于确保所有在CUDA设备上的计算都已经完成。

    // Rasterize visible Gaussians to image, obtain their radii (on screen).
    auto [rendererd_image, radii] = rasterizer.forward(
        means3D,
        means2D,//这个是要获得的？
        opacity,
        shs,
        colors_precomp,//定义了shs就不用定义colors_precomp了（如果不是RGB颜色的话，就是sh没定义的话，就需要预先提供高斯球的颜色）
        scales,
        rotations,
        cov3D_precomp);//如果 scales 或 rotations 中有一个定义了 cov3D_precomp 就不需要定义了（cov3D_precomp为预先计算的协方差矩阵）

    // Apply visibility filter to remove occluded Gaussians.
    // TODO: I think there is no real use for means2D, isn't it?
    // render, viewspace_points, visibility_filter, radii
    return {rendererd_image, means2D, radii > 0, radii};
}
