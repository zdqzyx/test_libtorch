#include <torch/script.h> // LibTorch 头文件
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <chrono> // 必须包含这个头文件


int main(int argc, char** argv) {
    // 1. 设置模型路径
    // std::string model_path = "/home/nescadmin/Code/NESC_MD_API/test/alstm_f56_cpu.pt";

    // 解析命令行参数
    std::string model_path = argv[1];
    int batch_size = std::atoi(argv[2]);
    int seq_len = std::atoi(argv[3]);
    int input_size = std::atoi(argv[4]);

    std::cout << "Loading model from: " << model_path << std::endl;
    std::cout << "Input dimensions: batch_size=" << batch_size 
            << ", seq_len=" << seq_len 
            << ", input_size=" << input_size << std::endl;

    // 1. 检查是否有 CUDA
    if (torch::cuda::is_available()) {
        std::cerr << "CUDA is available! " << std::endl;
    }else{
        std::cerr << "CUDA is NOT available! Falling back to CPU." << std::endl;
    }
    
    // 2. 加载模型
    torch::jit::script::Module module;
    try {
        // 强制映射到 CPU
        module = torch::jit::load(model_path, torch::kCPU);
        module.eval(); // 极其重要：禁用 Dropout/BatchNormalization 更新
    }
    catch (const c10::Error& e) {
        std::cerr << "加载模型错误: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "模型加载成功！" << std::endl;

    // 3. 构造输入数据 (Batch, Seq_Len, Features)
    // 这里构造与 Python 导出时相同维度的数据
    // int batch_size = 5200;
    // int seq_len = 240;
    // int input_size = 56;
    


    try {

        for(int i=0; i<5; i++){
            // 创建输入 Tensor (随机数)
            torch::Tensor input_tensor = torch::randn({batch_size, seq_len, input_size});
            // 确保输入也在 CPU 上
            input_tensor = input_tensor.to(torch::kCPU);
            // 4. 执行推理
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // 记录开始时间
            auto start = std::chrono::high_resolution_clock::now();

            // forward() 返回 IValue，需要转换为 Tensor
            at::Tensor output = module.forward(inputs).toTensor();

            // 记录结束时间
            auto end = std::chrono::high_resolution_clock::now();
            // ==========================================
            // 计算耗时
            // ==========================================
            // 1. 获取秒 (double类型，保留小数)
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "推理耗时 (秒): " << elapsed_seconds.count() << "s" << std::endl;

            // 2. 获取毫秒 (常用于性能指标)
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "推理耗时 (毫秒): " << elapsed_ms.count() << "ms" << std::endl;

            
            std::cout << "推理完成。\n输出形状: " << output.sizes() << std::endl;
            std::cout << "前5个输出值: " << output.slice(0, 0, 5) << std::endl;
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "推理执行错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}