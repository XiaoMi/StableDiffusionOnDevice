#include "models/UNet.h"

UNet::UNet() {
    latent_h = 32;
    latent_w = 32;

    model_sub_path_ = "/stable_diffusion/onnx/unet_fp16.ort";

    model_name_ = "UNet==";

    input_node_names = {"sample",
                        "timestep",
                        "encoder_hidden_states"};
    input_node_dims = {
            {1, latent_c, latent_h, latent_w},
            {1,},
            {1, 77, 768}
    };

    output_node_names = {"out_sample",};
    output_node_dims = {
            {1, latent_c, latent_h, latent_w}
    };
    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
}

UNet::~UNet() = default;

int UNet::pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) {
    ONNXModel::pre_process(inputs, outputs);
    auto sample = inputs[0].clone();
    convert_hwc_to_chw(reinterpret_cast<const float *>(sample.data),
                       input_node_values[0].data(),
                       latent_h,
                       latent_w,
                       latent_c);
    return 0;
}

int UNet::post_process(std::vector<cv::Mat> &outputs) {
    auto sample = outputs[0].clone();
    convert_chw_to_hwc(reinterpret_cast<const float *>(sample.data),
                       reinterpret_cast<float *>(outputs[0].data),
                       latent_h,
                       latent_w,
                       latent_c);
    return 0;
}

void UNet::set_latent_size(int latent_size_h, int latent_size_w) {
    LOGI("%s set latent size %d %d", model_name_.c_str(), latent_size_h, latent_size_w);

    latent_h = latent_size_h;
    latent_w = latent_size_w;

    input_node_dims[0][2] = latent_h;
    input_node_dims[0][3] = latent_w;
    output_node_dims[0][2] = latent_h;
    output_node_dims[0][3] = latent_w;
    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
}

cv::Mat UNet::decode(const cv::Mat &input, float t, const cv::Mat &cond) {
    cv::Mat t_mat(cv::Size(1, 1), CV_32FC1, cv::Scalar_<float>(t));
    cv::Mat denoised_cond(cv::Size(latent_w, latent_h), CV_32FC4);

    std::vector<cv::Mat> inputs = {input, t_mat, cond}, outputs{denoised_cond};
    inference(inputs, outputs);
    return outputs[0];
}


