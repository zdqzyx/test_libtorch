#include <torch/torch.h>
#include <torch/script.h> // 如果你是 TorchScript 模型

#include <iostream>

int main(int argc, char** argv) {
    // 解析命令行参数
//    std::string model_path = "../model/alstm_f56_cpu.pt";
    std::string model_path = argv[1];
    int batch_size = std::atoi(argv[2]);
    int seq_len = std::atoi(argv[3]);
    int input_size = std::atoi(argv[4]);

    std::cout << "Loading model from: " << model_path << std::endl;
    std::cout << "Input dimensions: batch_size=" << batch_size 
            << ", seq_len=" << seq_len 
            << ", input_size=" << input_size << std::endl;
    
    std::cout << "CUDA device count: " << torch::cuda::device_count() << "个" << std::endl;
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "cuDNN enabled: " << torch::cuda::cudnn_is_available() << std::endl;

    // 1. 检查是否有 CUDA
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Falling back to CPU." << std::endl;
    }else{
        std::cout << "CUDA is available! Using GPU for inference." << std::endl;
    }

    // 2. 设备选择
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        device = torch::Device(torch::kCPU);
    }
    // device = torch::Device(torch::kCPU);


    try {
        // 3. 加载 TorchScript 模型（假设是 .pt 文件）
        torch::jit::script::Module module = torch::jit::load(model_path, device);
        module.to(device);   // 放到 GPU

        module.eval();



        for(int i=0; i<10; i++){

            // 4. 构造输入张量（例：你的 (500,240,56)）
            // torch::Tensor input = torch::randn({500, 240, 56}, torch::kFloat32);
            torch::Tensor input = torch::randn({batch_size, seq_len, input_size}, torch::kFloat32);
            // 记录开始时间
            auto start = std::chrono::high_resolution_clock::now();
            input = input.to(device);  // 放到 GPU

            // 5. 推理
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            // 如果是 GPU，要同步确保测时准确
            auto output = module.forward(inputs).toTensor();


            // 记录结束时间
            auto end = std::chrono::high_resolution_clock::now();

            // 6. 如果需要回到 CPU：
            auto output_cpu = output.to(torch::kCPU);

            // ==========================================
            // 计算耗时
            // ==========================================
            // 1. 获取秒 (double类型，保留小数)
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "推理耗时 (秒): " << elapsed_seconds.count() << "s" << std::endl;

            // 2. 获取毫秒 (常用于性能指标)
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "推理耗时 (毫秒): " << elapsed_ms.count() << "ms" << std::endl;

            
            std::cout << "推理完成。\n输出形状: " << output_cpu.sizes() << std::endl;
            std::cout << "前5个输出值: " << output_cpu.slice(0, 0, 5) << std::endl;
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "推理执行错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
