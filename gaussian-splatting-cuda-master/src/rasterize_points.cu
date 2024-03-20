/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "config.h"
#include "rasterize_points.cuh"
#include "rasterizer.h"
#include <torch/extension.h>
#include <tuple>

/**
 * @brief 调整传入的张量的大小，并返回其连续内存的指针。
 * @param t 接受一个torch::Tensor的引用
 * @return lambda 返回一个std::function对象，该对象接受一个size_t类型的参数，并返回一个char*类型的指针
 */
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        t.zero_();
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,//所有高斯的中心点
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug) 
{
    //检查输入的 means3D 张量是否为二维，并且第二维的大小是否为 3，如果不是则抛出错误。
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);//高斯球的个数
    const int H = image_height;//图像的高
    const int W = image_width;//图像的宽

    auto int_opts = means3D.options().dtype(torch::kInt32);//32位整数类型的张量选项。
    auto float_opts = means3D.options().dtype(torch::kFloat32);//32位浮点数类型的张量选项。

    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);//输出的颜色
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));//输出的半径

    torch::Device device(torch::kCUDA);// 创建一个 CUDA 设备对象
    torch::TensorOptions options(torch::kByte);//创建一个字节类型的张量选项。
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));//创建一个空的几何缓冲区张量，并将其放在 CUDA 设备上。
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);//创建一个函数对象，用于调整几何缓冲区的大小。
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;//渲染点的数量
    if (P != 0) {
        int M = 0;
        if (sh.size(0) != 0) {
            M = sh.size(1);
        }

        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,//几何缓冲区的调整大小函数
            binningFunc,//binning缓冲区的调整大小函数
            imgFunc,//图像缓冲区的调整大小函数
            P, degree, M,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            sh.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),//输出的颜色
            radii.contiguous().data_ptr<int>(),//输出的半径
            debug);
    }
    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug) {
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    int M = 0;
    if (sh.size(0) != 0) {
        M = sh.size(1);
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

    if (P != 0) {

        CudaRasterizer::Rasterizer::backward(P, degree, M, R,
                                             background.contiguous().data_ptr<float>(),
                                             W, H,
                                             means3D.contiguous().data_ptr<float>(),
                                             sh.contiguous().data_ptr<float>(),
                                             colors.contiguous().data_ptr<float>(),
                                             scales.data_ptr<float>(),
                                             scale_modifier,
                                             rotations.data_ptr<float>(),
                                             cov3D_precomp.contiguous().data_ptr<float>(),
                                             viewmatrix.contiguous().data_ptr<float>(),
                                             projmatrix.contiguous().data_ptr<float>(),
                                             campos.contiguous().data_ptr<float>(),
                                             tan_fovx,
                                             tan_fovy,
                                             radii.contiguous().data_ptr<int>(),
                                             reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
                                             reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
                                             reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
                                             dL_dout_color.contiguous().data_ptr<float>(),
                                             dL_dmeans2D.contiguous().data_ptr<float>(),
                                             dL_dconic.contiguous().data_ptr<float>(),
                                             dL_dopacity.contiguous().data_ptr<float>(),
                                             dL_dcolors.contiguous().data_ptr<float>(),
                                             dL_dmeans3D.contiguous().data_ptr<float>(),
                                             dL_dcov3D.contiguous().data_ptr<float>(),
                                             dL_dsh.contiguous().data_ptr<float>(),
                                             dL_dscales.contiguous().data_ptr<float>(),
                                             dL_drotations.contiguous().data_ptr<float>(),
                                             debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix) {
    const int P = means3D.size(0);

    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

    if (P != 0) {
        CudaRasterizer::Rasterizer::markVisible(P,
                                                means3D.contiguous().data_ptr<float>(),
                                                viewmatrix.contiguous().data_ptr<float>(),
                                                projmatrix.contiguous().data_ptr<float>(),
                                                present.contiguous().data_ptr<bool>());
    }

    return present;
}