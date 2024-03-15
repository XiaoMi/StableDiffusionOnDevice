#include "models/UNet.h"
#include "qcom/time_all.h"

UNet::UNet() {
    latent_h = 32;
    latent_w = 32;
    model_sub_path_ = std::string("/stable_diffusion/qnn_model_") + QCOM_VERSION +
                      "/unet_quantized.serialized.bin";
    model_name_ = "UNet==";
    input_node_dims = {
            {1, latent_h, latent_w, latent_c},
            {1280,},
            {1, 77, 768}
    };

    output_node_names = {"out_sample",};
    output_node_dims = { {1, latent_h, latent_w, latent_c} };
    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
}

UNet::~UNet() = default;

int UNet::pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) {
    QCOMModel::pre_process(inputs, outputs);
    return 0;
}

int UNet::post_process(std::vector<cv::Mat> &outputs) {
    return 0;
}

void UNet::set_latent_size(int latent_size_h, int latent_size_w) {
    LOGI("%s set latent size %d %d", model_name_.c_str(), latent_size_h, latent_size_w);

    latent_h = latent_size_h;
    latent_w = latent_size_w;

    input_node_dims[0][1] = latent_h;
    input_node_dims[0][2] = latent_w;
    output_node_dims[0][1] = latent_h;
    output_node_dims[0][2] = latent_w;
    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
}

int UNet::denoise(const cv::Mat &input, float t, const cv::Mat &cond, cv::Mat &denoised) {
    cv::Mat t_mat(cv::Size(1280, 1), CV_32FC1, time_embedding_input_map[(int) t]);
    std::vector<cv::Mat> inputs = {input, t_mat, cond}, outputs{denoised};

    if (inference(inputs, outputs) < 0)
        return -1;
    return 0;
}


