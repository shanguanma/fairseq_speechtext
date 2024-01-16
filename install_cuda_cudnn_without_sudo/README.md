
# Why provide such an installation tutorial?
 * Some clusters may not provide such tools and only provide cuda drivers.
 * For cpp projects that require direct operation of cuda, such as flash_attn and k2, they cannot be compiled, even if they are installed using anaconda cudatookit

# reference: https://k2-fsa.github.io/k2/installation/cuda-cudnn.html

# Install cuDNN and CUDA 11.8

```
You can use the following commands to install CUDA 11.8.0. We install it into /home/maduo/installed/cuda-11.8.0. You can replace it if needed.
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

chmod +x cuda_11.8.0_520.61.05_linux.run

./cuda_11.8.0_520.61.05_linux.run \  
--silent \  
--toolkit \
  --installpath=/home/maduo/installed/cuda-11.8.0 \
    --no-opengl-libs \  --no-drm \  --no-man-page

wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz

tar xvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz --strip-components=1 -C /home/maduo/installed/cuda-11.8.0

You can save the following code to activate-cuda-11.8.sh
export CUDA_HOME=/home/maduo/installed/cuda-11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDA_TOOLKIT_ROOT=$CUDA_HOME
export CUDA_BIN_PATH=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

To check that you have installed CUDA 11.8 successfully, please run:
which nvcc

nvcc --version

The output should look like the following:

/home/maduo/installed/cuda-11.8.0/bin/nvcc

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0

```

# Install cuDNN and CUDA 12.1


```
You can use the following commands to install CUDA 12.1. We install it into /home/maduo/installed/cuda-12.1.0. You can replace it if needed.
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

chmod +x cuda_12.1.0_530.30.02_linux.run

./cuda_12.1.0_530.30.02_linux.run \
  --silent \
  --toolkit \
  --installpath=/home/maduo/installed/cuda-12.1.0 \
  --no-opengl-libs \
  --no-drm \
  --no-man-page
Install cuDNN for CUDA 12.1
Now, install cuDNN for CUDA 12.1.
wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-linux-x86_64-8.9.5.29_cuda12-archive.tar.xz

tar xvf cudnn-linux-x86_64-8.9.5.29_cuda12-archive.tar.xz --strip-components=1 -C /home/maduo/installed/cuda-12.1.0
Set environment variables for CUDA 12.1
Note that we have to set the following environment variables after installing CUDA 11.8. You can save the following code to activate-cuda-12.1.sh and use source activate-cuda-12.1.sh if you want to activate CUDA 12.1.
export CUDA_HOME=/home/maduo/installed/cuda-12.1.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDAToolkit_ROOT_DIR=$CUDA_HOME
export CUDAToolkit_ROOT=$CUDA_HOME
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDA_TOOLKIT_ROOT=$CUDA_HOME
export CUDA_BIN_PATH=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS
export CUDAToolkit_TARGET_DIR=$CUDA_HOME/targets/x86_64-linux
To check that you have installed CUDA 12.1 successfully, please run:
which nvcc

nvcc --version
The output should look like the following:
/home/maduo/installed/cuda-12.1.0/bin/nvcc

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```
