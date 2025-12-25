wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.4.0%2Bcpu.zip
unzip libtorch-shared-with-deps-2.4.0+cpu.zip


wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.1%2Bcu124.zip


wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.1%2Bcu126.zip
unzip libtorch-shared-with-deps-2.9.1+cu126.zip -d libtorch_gpu

unzip  libtorch-shared-with-deps-2.9.1+cu130.zip -d libtorch_gpu_cu130


cd /mnt/c/Users/zdq/Documents/Code/Cplus/NESC_MD_API


sudo dnf install libcudnn9-cuda-12.x86_64 libcudnn8-devel -y


# 关键：指定 LibTorch 的路径
# CMAKE_PREFIX_PATH 指向解压后的 libtorch 文件夹

cd ..
rm -rf build
mkdir build
cd build

rm -rf ./*
cmake -DCMAKE_PREFIX_PATH=/mnt/c/Users/zdq/Documents/Code/Cplus/NESC_MD_API/test/libtorch/libtorch_gpu_cu128/libtorch ..
make
./app /mnt/c/Users/zdq/Documents/Code/Cplus/NESC_MD_API/test/libtorch_test/model/alstm_f56_cpu.pt 500 240 56

./app /mnt/c/Users/zdq/Documents/Code/Cplus/NESC_MD_API/test/libtorch_test/model/alstm_f56_gpu.pt 500 240 56


# 再次强调，你需要传入正确的路径。
# 这里的 TORCH_CUDA_ARCH_LIST 参数可以省略，因为它已经在 CMakeLists.txt 中设置了。
rm -rf ../build/*
cmake -DCMAKE_PREFIX_PATH=/mnt/c/Users/zdq/Documents/Code/Cplus/NESC_MD_API/test/libtorch_test/libdir/libtorch_gpu_cu128/libtorch \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      ..


cmake .. \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DUSE_CUDNN=ON


export LD_LIBRARY_PATH=/mnt/c/Users/zdq/Documents/Code/Cplus/NESC_MD_API/test/libtorch_test/libdir/libtorch_gpu_cu128/libtorch/lib:$LD_LIBRARY_PATH














# 安装 CUDA Toolkit 12.4
sudo dnf install cuda-toolkit-12-4
# 安装 cudnn 9.3
wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm
sudo rpm -i cudnn-local-repo-rhel9-9.3.0-1.0-1.x86_64.rpm


