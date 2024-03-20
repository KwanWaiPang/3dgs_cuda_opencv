#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "loss_monitor.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "render_utils.cuh"
#include "scene.cuh"
#include <args.hxx>
#include <c10/cuda/CUDACachingAllocator.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


void Write_model_parameters_to_file(const gs::param::ModelParameters& params) {
    std::filesystem::path outputPath = params.output_path;//获取输出路径
    std::filesystem::create_directories(outputPath); // Make sure the directory exists

    std::ofstream cfg_log_f(outputPath / "cfg_args");
    if (!cfg_log_f.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    // Write the parameters in the desired format（写入基本的模型参数）
    cfg_log_f << "Namespace(";
    cfg_log_f << "eval=" << (params.eval ? "True" : "False") << ", ";
    cfg_log_f << "images='" << params.images << "', ";
    cfg_log_f << "model_path='" << params.output_path.string() << "', ";
    cfg_log_f << "resolution=" << params.resolution << ", ";
    cfg_log_f << "sh_degree=" << params.sh_degree << ", ";
    cfg_log_f << "source_path='" << params.source_path.string() << "', ";
    cfg_log_f << "white_background=" << (params.white_background ? "True" : "False") << ")";
    cfg_log_f.close();

    std::cout << "Output folder: " << params.output_path.string() << std::endl;
}

/**
 * 用于生成一个从 0 到 max_index - 1 的随机索引向量（打乱再反转）
 * 
 * @param max_index 最大的索引
 * @return 索引数组
 */
std::vector<int> get_random_indices(int max_index) {
    std::vector<int> indices(max_index);//创建了一个容量为 max_index 的整数向量
    std::iota(indices.begin(), indices.end(), 0);//使用 std::iota 函数，将索引向量 indices 初始化为从 0 开始递增的整数序列。
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());//使用 std::shuffle 函数，对索引向量进行随机打乱，以生成随机索引序列。
    std::reverse(indices.begin(), indices.end());//将索引向量反转，这一步的目的是为了将原本的随机序列转换为降序排列。
    return indices;
}

/**
 * 读入命令行的参数
 * 
 * @param args 命令行参数
 * @param modelParams 模型参数
 * @param optimParams 优化参数
 * @return 1或0，是否成功读入
 */
int parse_cmd_line_args(const std::vector<std::string>& args,//命令行参数
                        gs::param::ModelParameters& modelParams,//模型参数
                        gs::param::OptimizationParameters& optimParams) //优化参数
{
    if (args.empty()) {
        std::cerr << "No command line arguments provided!" << std::endl;
        return -1;
    }
    args::ArgumentParser parser("3D Gaussian Splatting CUDA Implementation\n",
                                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");
    //当命令行中出现了下面的参数时，就会调用相应的函数
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
    args::ValueFlag<int> resolution(parser, "resolution", "Set resolutino", {'r', "resolution"});
    args::Flag enable_cr_monitoring(parser, "enable_cr_monitoring", "Enable convergence rate monitoring", {"enable-cr-monitoring"});
    args::Flag force_overwrite_output_path(parser, "force", "Forces to overwrite output folder", {'f', "force"});//强制覆盖输出文件夹
    args::Flag empty_gpu_memory(parser, "empty_gpu_cache", "Forces to reset GPU Cache. Should be lighter on VRAM", {"empty-gpu-cache"});
    args::ValueFlag<std::string> data_path(parser, "data_path", "Path to the training data", {'d', "data-path"});//训练数据的路径
    args::ValueFlag<std::string> output_path(parser, "output_path", "Path to the training output", {'o', "output-path"});//训练输出的路径
    args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations to train the model", {'i', "iter"});//迭代次数
    args::CompletionFlag completion(parser, {"complete"});

    try {
        parser.Prog(args.front());
        parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    } catch (const args::Help&) {
        std::cout << parser;
        return -1;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }

    //存在什么参数就进行什么样的处理
    if (data_path) {
        modelParams.source_path = args::get(data_path);//训练数据的路径
    } else {
        std::cerr << "No data path provided!" << std::endl;
        return -1;
    }
    if (output_path) {
        modelParams.output_path = args::get(output_path);//训练输出的路径
        // std::cout << "Input the Output directory: " << modelParams.output_path << std::endl;
    } else {
        std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
        std::filesystem::path parentDir = executablePath.parent_path().parent_path();
        std::filesystem::path outputDir = parentDir / "output";
        try {

            bool isCreated = std::filesystem::create_directory(outputDir);
            if (!isCreated) {
                if (!force_overwrite_output_path) {//如果不强制覆盖输出文件夹
                    // std::cerr << "Directory already exists! Not overwriting it" << std::endl;
                    std::cout << "Directory already exists! Not overwriting it" << std::endl;
                    return -1;
                } else {
                    std::cout<< "Output directory already exists! Overwriting it" << std::endl;
                    std::filesystem::create_directory(outputDir);
                    std::filesystem::remove_all(outputDir);
                }
            }
        } catch (...) {
            std::cerr << "Failed to create output directory!" << std::endl;
            return -1;
        }
        modelParams.output_path = outputDir;
    }

    if (iterations) {
        optimParams.iterations = args::get(iterations);
    }
    optimParams.early_stopping = args::get(enable_cr_monitoring);
    if (optimParams.early_stopping && convergence_rate) {
        optimParams.convergence_threshold = args::get(convergence_rate);
    }

    if (resolution) {
        modelParams.resolution = args::get(resolution);
    }

    optimParams.empty_gpu_cache = args::get(empty_gpu_memory);
    return 0;
}

/**
 * 计算图像的峰值信噪比（PSNR）
 * 
 * @param rendered_img 渲染的图像
 * @param gt_img 真实的图像
 * @return 峰值信噪比值
 */
float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) {

    torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
    torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
    return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

// 将 torch::Tensor 转换为 cv::Mat
cv::Mat tensor_to_mat(const torch::Tensor& tensor) {
    // 克隆 tensor，确保其数据在 CPU 内存上连续存储
    torch::Tensor tensor_cpu = tensor.to(torch::kCPU).clone();

    // 获取 tensor 的尺寸和数据指针
    // tensor创建的时候为 C × H × W，同时归一化到 [0, 1] 之间
    int height = tensor_cpu.size(1);
    int width = tensor_cpu.size(2);
    int channels = tensor_cpu.size(0);
    const float* data_ptr = tensor_cpu.data_ptr<float>();

    // // 创建对应尺寸的 cv::Mat
    // cv::Mat mat(height, width, CV_MAKETYPE(CV_32F, channels), const_cast<float*>(data_ptr));
    // 创建对应尺寸的 cv::Mat，并进行深拷贝
    cv::Mat mat(height, width, CV_MAKETYPE(CV_32F, channels));
    memcpy(mat.data, data_ptr, sizeof(float) * height * width * channels);

    // 将图像数据类型转换为 8-bit 无符号整数类型（范围 [0,255]）
    mat.convertTo(mat, CV_8U, 255.0);

    return mat;
}

int main(int argc, char* argv[]) {//argc参数表示命令行参数的数量，argv参数是一个指向字符串数组的指针，其中存储了命令行参数的值。
    std::vector<std::string> args;//用来存储命令行参数
    args.reserve(argc);//用来预留args向量的空间以容纳argc个元素

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);//将命令行参数存储到args向量中
    }
    // TODO: read parameters from JSON file or command line
    auto modelParams = gs::param::ModelParameters();//初始化模型参数
    auto optimParams = gs::param::read_optim_params_from_json();//初始化优化参数（从json文件中读入参数）
    //读入命令行参数
    if (parse_cmd_line_args(args, modelParams, optimParams) < 0) {
        return -1;
    };
    //创建输出的文件，并且将模型参数写入文件
    Write_model_parameters_to_file(modelParams);

    auto gaussians = GaussianModel(modelParams.sh_degree);//初始化高斯模型（仅仅是声明以及存放sh_degree，并没有真正的对高斯函数进行初始化）
    auto scene = Scene(gaussians, modelParams);//初始化场景，传入模型的参数，最终获取初始化后的高斯模型（进行高斯函数的初始化、获取图像数据）
    //根据给定的优化参数初始化模型的优化器，并设置学习率、参数组和其他优化器参数。
    gaussians.Training_setup(optimParams);//根据优化参数进行训练设置

    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        // At the moment, I want to make sure that my GPU is utilized.
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        exit(-1);
    }
    auto pointType = torch::TensorOptions().dtype(torch::kFloat32);//设置张量的数据类型
    auto background = modelParams.white_background ? torch::tensor({1.f, 1.f, 1.f}) : torch::tensor({0.f, 0.f, 0.f}, pointType).to(torch::kCUDA);//设置背景颜色（根据white_background来选择。然后将背景颜色转化为 PyTorch Tensor，并移到 GPU 上。）

    //创建高斯卷积窗口（这是计算ssim loss的时候需要用到的）
    const int window_size = 11;//定义了卷积窗口的大小，即窗口的边长为 11 个像素。
    const int channel = 3;//定义了卷积窗口的通道数，即颜色通道为 3。
    const auto conv_window = gaussian_splatting::create_window(window_size, channel).to(torch::kFloat32).to(torch::kCUDA, true);//创建一个高斯卷积窗口，并将其转换为 float32 数据类型，并移动到 CUDA 设备上。

    const int camera_count = scene.Get_camera_count();//获取场景中的相机数量

    std::vector<int> indices;
    int last_status_len = 0;
    auto start_time = std::chrono::steady_clock::now();//获取当前时间
    float loss_add = 0.f;

    LossMonitor loss_monitor(200);//初始化损失监视器,buffersize=200
    float avg_converging_rate = 0.f;

    float psnr_value = 0.f;//初始化psnr值
    for (int iter = 1; iter < optimParams.iterations + 1; ++iter) {
        if (indices.empty()) {
            indices = get_random_indices(camera_count);
        }
        const int camera_index = indices.back();//获取最后一个索引（此时应该是乱序的）
        auto& cam = scene.Get_training_camera(camera_index);//获取当前索引对应的相机对象
        auto gt_image = cam.Get_original_image().to(torch::kCUDA, true);//获取当前索引对应的相机对象的图像tensor值（torch::Tensor）
        indices.pop_back(); //（删掉最后一个，也就是刚刚拿了的） remove last element to iterate over all cameras randomly

        // 每 1000 iterations将sh_degree增加1，最高不超过_max_sh_degree
        if (iter % 1000 == 0) {
            gaussians.One_up_sh_degree();
        }

        // Render进行渲染处理
        auto [image, viewspace_point_tensor, visibility_filter, radii] = render(cam, gaussians, background);

        // Loss Computations（计算loss）
        auto l1l = gaussian_splatting::l1_loss(image, gt_image);//计算l1损失
        auto ssim_loss = gaussian_splatting::ssim(image, gt_image, conv_window, window_size, channel);//计算ssim损失
        auto loss = (1.f - optimParams.lambda_dssim) * l1l + optimParams.lambda_dssim * (1.f - ssim_loss);//计算总的损失

        // Update status line
        //每100次迭代更新一次状态行，用于在控制台输出当前训练的进度信息。
        if (iter % 100 == 0) {
            auto cur_time = std::chrono::steady_clock::now();//获取当前时间点
            std::chrono::duration<double> time_elapsed = cur_time - start_time;//计算从开始训练到当前时间经过的时间间隔。
            // XXX shouldn't have to create a new stringstream, but resetting takes multiple calls
            std::stringstream status_line;//创建一个字符串流，用于构建状态行的文本。
            // XXX Use thousand separators, but doesn't work for some reason
            status_line.imbue(std::locale(""));//确保输出的数字格式在当前地域设置下是正确的，比如使用适当的小数点符号、千位分隔符等。
            status_line
                << "\rIter: " << std::setw(6) << iter
                << "  Loss: " << std::fixed << std::setw(9) << std::setprecision(6) << loss.item<float>();
            if (optimParams.early_stopping) {
                status_line
                    << "  ACR: " << std::fixed << std::setw(9) << std::setprecision(6) << avg_converging_rate;
            }
            status_line
                << "  Splats: " << std::setw(10) << (int)gaussians.Get_xyz().size(0)
                << "  Time: " << std::fixed << std::setw(8) << std::setprecision(3) << time_elapsed.count() << "s"
                << "  Avg iter/s: " << std::fixed << std::setw(5) << std::setprecision(1) << 1.0 * iter / time_elapsed.count()
                << "  " // Some extra whitespace, in case a "Pruning ... points" message gets printed after
                ;
            const int curlen = status_line.str().length();
            const int ws = last_status_len - curlen;
            if (ws > 0)
                status_line << std::string(ws, ' ');
            std::cout << status_line.str() << std::flush;
            last_status_len = curlen;
        }

        if (optimParams.early_stopping) {
            avg_converging_rate = loss_monitor.Update(loss.item<float>());
        }
        loss_add += loss.item<float>();
        loss.backward();//更新loss，进行反向传播

        {//用于参数的更新
            torch::NoGradGuard no_grad;//确保在此代码块中关闭梯度的计算
            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);//根据可见性过滤器 visibility_filter 从 _max_radii2D 中选择可见的最大半径。
            auto visible_radii = radii.masked_select(visibility_filter);//根据可见性过滤器 visibility_filter 从 radii 中选择可见的半径。
            auto max_radii = torch::max(visible_max_radii, visible_radii);//计算可见最大半径和可见半径的最大值。
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);//使用最大半径更新 _max_radii2D，仅更新可见的半径。

            //如果达到最大的迭代次数，那么久保存结果并且计算psnr
            if (iter == optimParams.iterations) {
                std::cout << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                psnr_value = psnr_metric(image, gt_image);
                //保存渲染的图像
                cv::Mat render_image = tensor_to_mat(image);
                cv::Mat gt_image_mat = tensor_to_mat(gt_image);
                cv::imshow("img for 3dgs", render_image);
                cv::imshow("gt img", gt_image_mat);
                // cv::waitKey(1); // 等待1毫秒，以便处理窗口事件
                cv::waitKey(0); // 等待键盘输入，直到有按键被按下
                break;
            }

            //如果每7k保存一次结果
            if (iter % 7'000 == 0) {
                gaussians.Save_ply(modelParams.output_path, iter, false);
            }

            // Densification（进行稠密化）
            if (iter < optimParams.densify_until_iter) {//达到一定的代数才进行致密化
                gaussians.Add_densification_stats(viewspace_point_tensor, visibility_filter);//向模型中添加稠密化统计信息
                if (iter > optimParams.densify_from_iter && iter % optimParams.densification_interval == 0) //达到指定的迭代次数后，以一定的频率执行稠密化操作
                {
                    // @TODO: Not sure about type
                    float size_threshold = iter > optimParams.opacity_reset_interval ? 20.f : -1.f;
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, optimParams.min_opacity, scene.Get_cameras_extent(), size_threshold);//进行剪枝以及稠密化的操作
                }

                if (iter % optimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                    gaussians.Reset_opacity();
                }
            }

            if (iter >= optimParams.densify_until_iter && loss_monitor.IsConverging(optimParams.convergence_threshold)) {
                std::cout << "Converged after " << iter << " iterations!" << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                break;
            }

            //  Optimizer step
            if (iter < optimParams.iterations) {
                gaussians._optimizer->step();//执行优化器的一步操作，即更新模型参数。
                gaussians._optimizer->zero_grad(true);//将模型参数的梯度清零。参数 true 表示同时清空张量的持久化缓存，以释放内存。
                // @TODO: Not sure about type
                gaussians.Update_learning_rate(iter);//根据当前迭代次数更新学习率。这可能是根据迭代次数动态调整学习率的策略。
            }

            //如果要清空GPU的缓存，每隔100代清空一次，以避免内存溢出或性能下降。
            if (optimParams.empty_gpu_cache && iter % 100) {
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
    }

    auto cur_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = cur_time - start_time;//计算一共的时间

    std::cout << std::endl
              << "The training of the 3DGS is done in "
              << std::fixed << std::setw(7) << std::setprecision(3) << time_elapsed.count() << "sec, avg "
              << std::fixed << std::setw(4) << std::setprecision(1) << 1.0 * optimParams.iterations / time_elapsed.count() << " iter/sec, "
              << gaussians.Get_xyz().size(0) << " splats, "
              << std::fixed << std::setw(7) << std::setprecision(6) << " psrn: " << psnr_value << std::endl
              << std::endl
              << std::endl;

    return 0;
}
