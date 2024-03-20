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

#pragma once

#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

namespace CudaRasterizer {
    template <typename T>
    static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment) {
        std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        chunk = reinterpret_cast<char*>(ptr + count);
    }

    struct GeometryState {
        size_t scan_size;
        float* depths;//视图空间中的z坐标
        char* scanning_space;
        bool* clamped;
        int* internal_radii;
        float2* means2D;//像素坐标
        float* cov3D;//3D协方差
        float4* conic_opacity;//逆2D协方差（确定椭圆函数的形状和大小，进而影响纹理采样时各个样本的权重。）和不透明度
        float* rgb;//颜色信息
        uint32_t* point_offsets;
        uint32_t* tiles_touched;//2D高斯函数覆盖的矩形范围

        static GeometryState fromChunk(char*& chunk, size_t P);
    };

    struct ImageState {
        uint2* ranges;
        uint32_t* n_contrib;
        float* accum_alpha;

        static ImageState fromChunk(char*& chunk, size_t N);
    };

    struct BinningState {
        size_t sorting_size;
        uint64_t* point_list_keys_unsorted;
        uint64_t* point_list_keys;
        uint32_t* point_list_unsorted;
        uint32_t* point_list;
        char* list_sorting_space;

        static BinningState fromChunk(char*& chunk, size_t P);
    };

    template <typename T>
    size_t required(size_t P) {
        char* size = nullptr;
        T::fromChunk(size, P);
        return ((size_t)size) + 128;
    }
}; // namespace CudaRasterizer