#include <torch/torch.h>
#include <torch/script.h> // 如果你是 TorchScript 模型

#include <iostream>

int main(int argc, char** argv) {
    std::cout << "CUDA device count: " << torch::cuda::device_count() << "个" << std::endl;
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "cuDNN enabled: " << torch::cuda::cudnn_is_available() << std::endl;
    // 1. 检查是否有 CUDA
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Falling back to CPU." << std::endl;
    }else{
        std::cout << "CUDA is available! Using GPU for inference." << std::endl;
    }
    return 0;
}
