# ML Development Environment

As of 2018, most modern machine learning libraries offer hardware accelerated computation though NVIDIA graphics cards. Unlike most programs that you are probably accustomed to writing, much of the code that you will be writing will be run on a computer's GPU instead of CPU. This process is nicely abstracted from you as the software programming, and you probably won't have to write any low-level parallel processing operations in GPU-specific APIs, but you will need to have the necessary graphics hardware, drivers, and low-level libraries installed on your development machines so that high-level libraries like Tensorflow can work their magic.

This is a short, non-comprehensive guide to setting up a machine learning development environment on an Ubuntu 16.04 desktop with an NVIDIA GeForce graphics card. Both OSX and Windows are supported by NVIDIA's graphics libraries and some of the ML libraries commonly used, but Ubuntu is by far the preferred OS for machine learning development. 

Unless you are building, training, and running your ML models exclusively in WebGL with [Tensorflow.js](https://js.tensorflow.org/), an NVIDIA GPU is non-optional. AMD, Intel, and other competitors do not provide industry-standard APIs and tooling for machine learning, at least not anything that is comparable to NVIDIAs support. 

I recommend a GTX 1080Ti or GTX 1080 if you can afford it. If not any of the GTX 10 series should work. If you don't have access to physical GPU hardware, you can rent cloud-based GPU servers and VPSes from AWS and their competitors. This solution is the cheapest for quick experiments, but cloud-based GPU instances are very expensive in the long run.

## NVIDIA CUDA

CUDA is NVIDIA's proprietary parallel computing API. It enables developers to write general purpose parallel programs for NVIDIA GPUs, or [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units). While it is possible to write your own CUDA C and C++ programs from scratch, you will most likely be interfacing with CUDA indirectly though a popular deep learning library. These libraries abstract away the custom CUDA code and provide high-level APIs in Python, etc.

Here is a short list of some of the most popular machine learning / deep learning libraries that use NVIDIA's CUDA underneath:

- Tensorflow
- Keras
- Pytorch
- Torch
- Caffe
- Theano (deprecated)

## Install CUDA & CuDNN

https://developer.nvidia.com/cuda-90-download-archive

Do you accept the previously read EULA?
accept/decline/quit: accept      

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?
(y)es/(n)o/(q)uit: n

Install the CUDA 9.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
 [ default is /usr/local/cuda-9.0 ]: 

 Do you want to install a symbolic link at /usr/local/cuda?

 -   PATH includes /usr/local/cuda-9.0/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-9.0/lib64, or, add /usr/local/cuda-9.0/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run the uninstall script in /usr/local/cuda-9.0/bin

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-9.0/doc/pdf for detailed information on setting up CUDA.

sudo dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.0_amd64.deb libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb

## Tensorflow GPU