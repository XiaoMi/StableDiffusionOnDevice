# XiaoMi Stable Diffusion

如果你熟悉中文，你可以查看[中文版本](README.md)

## Introduction

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](LICENSE)

You can try to runn the Stable Diffusion model using Xiaomi's on device deployment framework.

<img src="assets/dog.jpg" alt="drawing" width="200"/> <img src="assets/universe.jpg" alt="drawing" width="200"/> <img src="assets/girl.jpg" alt="drawing" width="200"/>

This project includes two branches:

### NPU Branch
The NPU branch primarily utilizes the embedded Neural Processing Unit (NPU) for computation， and is compatible with Xiaomi 13 series and Xiaomi 14 series devices. By default, the branch is set to work on the Xiaomi 13 series. If you need to switch to Xiaomi 14 series, simply modify the `DL_MODE to 2` in the `CMakeLists.txt` file.

### CPU Branch
The CPU branch uses the Central Processing Unit (CPU) for computation, and is compatible with any Xiaomi smartphone that meets the system requirements. You can freely choose the number of model inference steps, seed values, and the size of the generated images to adjust the image generation results.


## Installation Instructions:
1. Navigate to $ROOT/app/src/main/cpp and run the [opencv.sh](app\src\main\cpp\opencv.sh) script to install the OpenCV library.
2. Go to $ROOT/app/src/main/cpp/boost, replace $NDK_ROOT in the [boost.sh](app\src\main\cpp\boost\boost.sh) script with the appropriate NDK path (e.g., android-ndk-r25c), and then run the boost.sh script to install the Boost library.
3. Follow the instructions in the [README.md](app/src/main/assets/stable_diffusion/README.md) to install the stable diffusion library files in the $ROOT/app/src/main/assets/stable_diffusion directory.
4. Download the models as required for your specific platform according to the model download instructions provided.


### Model Download
Please download the model from [Huggingface](https://huggingface.co/billlight/XiaoMiStableDiffusionV1.0) and place it in the directory described below.

QNN Branch:
* Xiaomi 13 series: root_folder/app/src/main/assets/stable_diffusion/qnn_model_8550/
* Xiaomi 14 series: root_folder/app/src/main/assets/stable_diffusion/qnn_model_8650/

ONNX Branch:
* Xiaomi series: root_folder/app/src/main/assets/stable_diffusion/onnx/

Uncompressed models website: 
* [runwayml/stable-diffusion-v1-5](https://huggingface.co/apple/coreml-stable-diffusion-v1-5)


## System Requirements

|SDK                 | NDK               | CMAKE             | DEVICE PLATFORM      |
|:------------------:|:-----------------:|:-----------------:|:--------------------:|
|33                  |26.0.10792818      | 3.18.1            |Xiaomi 8 Gen 2/8 Gen 3|


## Performance Benchmarks


|      Device        |    Platform       | Processing Unit   |  Units of Memory  | Model Size(GB)     | Memory Request(GB) | Image Resolution  |Inference Time(s)  |
|:------------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:-----------------: |:-----------------:|:-----------------:|
|  Xiaomi 13/Pro     |  8Gen2            |  NPU              | INT8              | 1.10               | 1.5                | 512               | 9.4               |
|  XiaoMi 14/Pro     |  8Gen3            |  NPU              | INT8              | 1.10               | 1.5                | 512               | 4.6               |
|  Xiaomi 13/Pro     |  8Gen2            |  CPU              | FLOAT16           | 1.93               | 12                 | 256               | 134               |
|  XiaoMi 14/Pro     |  8Gen3            |  CPU              | FLOAT16           | 1.93               | 12                 | 256               | 103               |



Notes: 
* These benchmarks were conducted by Xiaomi Mobile Phone Team, and were tested on Xiaomi 13 series and Xiaomi 14 series in October 2023. 
* The specific explanations of the above parameters are as follows:
    * "Device" represents the test mobile phone model.
    * "Platform" represents the corresponding Qualcomm chip model of the test mobile phone.
    * "Processing Unit" represents the selected computing unit.
    * "Model Size" is the size of the SD model.
    * "Memory Request" is the runtime memory consumption.
    * "Inference Time" is the time taken to generate an image. 
* The image generation process follows the standard SD image generation process: 20 steps, 512*512 resolution, and 77 text token length. 
* If the actual prompt length exceeds 77, it will be cropped to 77 characters, including the start and end tokens. 
* Unet runs with a batch size of 1, so Unet needs to be executed twice for positive and negative prompts separately, and then combined the result together. 
* The model parameters for running on NPU are W8A16, and on CPU, they are W16A16. 
* The CPU version requires a large amount of phone memory to run properly. 
* This project is developed based on the Stable Diffusion V1.5 architecture, and does not support other architectures of SD models for now. 
* Performance heavily depends on the hardware, system loading, and thermal status. 

## Model Quantization


XiaoMi SD adopts Post-Training Quantization (PTQ) to convert the Float32 model into an INT8 quantized model. PTQ process reduces the model size to approximately $\frac{1}{4}$ of its original size, making the model size about 1.1GB.
The model is quantized by collecting high-quality calibration data and optimizing quantization methods: 
* Calibration Data Collection: Collect high-quality prompt datasets from various domains as calibration datasets. 
* Quantization Calibration Method: Use the AdaRound quantization algorithm and high-precision per-channel quantization methods to pick the optimal 8-bit fixed-point values for floating-point weight parameters. 
PTQ significantly reduces the storage and computation requirements of the model without sacrificing the performance of the large SD model, thereby improving the model's usability in resource-constrained environments. 


## Model Deployment 


Compared to the CPU solution, XiaoMi SD deploys the quantized model on Qualcomm's NPU through the QNN framework, fully utilizing the parallel computing capabilities of the NPU. This allows high-density operations such as matrix multiplication and convolution operations in deep learning to leverage hardware acceleration at the chip level, reducing power consumption and improving computational efficiency.


* Model Size reduces 75%+
* Memory Usage reduces 75%+
* Inference Speed increases 95%+


## License
[MIT-License](LICENSE.md)
