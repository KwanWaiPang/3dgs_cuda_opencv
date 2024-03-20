// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#include "rasterizer.cuh"

torch::autograd::tensor_list rasterize_gaussians(torch::Tensor means3D,
                                                 torch::Tensor means2D,
                                                 torch::Tensor sh,
                                                 torch::Tensor colors_precomp,
                                                 torch::Tensor opacities,
                                                 torch::Tensor scales,
                                                 torch::Tensor rotations,
                                                 torch::Tensor cov3Ds_precomp,
                                                 GaussianRasterizationSettings raster_settings) {
    //获取相机的内参
    torch::Device device = torch::kCUDA;
    auto image_height = torch::tensor(raster_settings.image_height, device);
    auto image_width = torch::tensor(raster_settings.image_width, device);
    auto tanfovx = torch::tensor(raster_settings.tanfovx, device);
    auto tanfovy = torch::tensor(raster_settings.tanfovy, device);

    //检查背景颜色是否在GPU上
    if (!raster_settings.bg.is_cuda()) {
        raster_settings.bg = raster_settings.bg.to(device);
    }
    auto scale_modifier = torch::tensor(raster_settings.scale_modifier, device);

    if (!raster_settings.viewmatrix.is_cuda()) {
        raster_settings.viewmatrix = raster_settings.viewmatrix.to(device);
    }
    if (!raster_settings.projmatrix.is_cuda()) {
        raster_settings.projmatrix = raster_settings.projmatrix.to(device);
    }
    auto sh_degree = torch::tensor(raster_settings.sh_degree, device);

    if (!raster_settings.camera_center.is_cuda()) {
        raster_settings.camera_center = raster_settings.camera_center.to(device);
    }
    auto prefiltered = torch::tensor(raster_settings.prefiltered, device);

    means2D = means2D.to(device);
    means3D = means3D.to(device);
    sh = sh.to(device);
    colors_precomp = colors_precomp.to(device);
    opacities = opacities.to(device);
    scales = scales.to(device);
    rotations = rotations.to(device);
    cov3Ds_precomp = cov3Ds_precomp.to(device);

    //将所有的参数都放在gpu后,调用自动求导的函数的前向传播过程，跳进_RasterizeGaussians的forward
    return _RasterizeGaussians::apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        raster_settings.bg,
        scale_modifier,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        sh_degree,
        raster_settings.camera_center,
        prefiltered);
}
