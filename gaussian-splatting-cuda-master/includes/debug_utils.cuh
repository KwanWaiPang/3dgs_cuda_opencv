// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <vector>

namespace ts {
    inline void save_my_tensor(const torch::Tensor& tensor, std::string filename) {
        std::cout << filename << ": Expected dims: " << tensor.dim() << " expected shape: " << tensor.sizes() << "Expected type: " << tensor.dtype() << std::endl;
        auto cpu_tensor = tensor.to(torch::kCPU); // Move tensor to CPU
        int64_t numel = cpu_tensor.numel();
        std::vector<int64_t> sizes = cpu_tensor.sizes().vec();
        int dims = cpu_tensor.dim();

        std::ofstream outfile(filename, std::ios::binary);

        // Write dimensions
        outfile.write(reinterpret_cast<char*>(&dims), sizeof(int));

        // Write sizes
        outfile.write(reinterpret_cast<char*>(sizes.data()), dims * sizeof(int64_t));

        // Write tensor data based on its type
        if (cpu_tensor.dtype() == torch::kFloat32) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<float>()), numel * sizeof(float));
        } else if (cpu_tensor.dtype() == torch::kInt64) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<int64_t>()), numel * sizeof(int64_t));
        } else if (cpu_tensor.dtype() == torch::kBool) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<bool>()), numel * sizeof(bool));
        } else if (cpu_tensor.dtype() == torch::kInt32) {
            outfile.write(reinterpret_cast<char*>(cpu_tensor.data_ptr<int32_t>()), numel * sizeof(int32_t));
        } else {
            throw std::runtime_error("Unsupported tensor type");
        }
        // Add more data types as needed...

        outfile.close();
    }

    inline torch::Tensor load_my_tensor(const std::string& filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (!infile.is_open()) {
            throw std::runtime_error("Failed to open file " + filename);
        }

        // Read tensor dimensions
        int dims;
        infile.read(reinterpret_cast<char*>(&dims), sizeof(int));

        // Read tensor sizes
        std::vector<int64_t> sizes(dims);
        infile.read(reinterpret_cast<char*>(sizes.data()), dims * sizeof(int64_t));

        // Determine the size of the tensor data
        int64_t numel = 1;
        for (int i = 0; i < dims; ++i) {
            numel *= sizes[i];
        }

        torch::Tensor tensor;

        // We assume here float
        std::vector<float> data(numel);
        infile.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));
        tensor = torch::tensor(data).reshape(sizes);

        infile.close();
        return tensor;
    }

    inline void print_debug_info(const torch::Tensor& tensor, const std::string& tensor_name, bool save_tensor = false) {
        std::cout << tensor_name << " Size: " << tensor.sizes() << ", Data type: " << tensor.dtype() << "\n";
        if (save_tensor) {
            save_my_tensor(tensor, "libtorch_" + tensor_name + ".pt");
        }
    }

#undef DEBUG_ERRORS
    //#define DEBUG_ERRORS

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
    template <typename T>
    inline void check(T err, const char* const func, const char* const file,
                      const int line) {
#ifdef DEBUG_ERRORS
        if (err != cudaSuccess) {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                      << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            std::exit(EXIT_FAILURE);
        }
#endif // DEBUG_ERRORS
    }

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
    inline void checkLast(const char* const file, const int line) {
#ifdef DEBUG_ERRORS
        cudaDeviceSynchronize();
        cudaError_t err{cudaGetLastError()};
        if (err != cudaSuccess) {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                      << std::endl;
            std::cerr << cudaGetErrorString(err) << std::endl;
            std::exit(EXIT_FAILURE);
        }
#endif // DEBUG_ERRORS
    }
} // namespace ts