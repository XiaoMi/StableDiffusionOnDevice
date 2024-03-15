#include "models/DecoderModel.h"

DecoderModel::DecoderModel() {
    model_sub_path_ = std::string("/stable_diffusion/qnn_model_") + QCOM_VERSION +
                      "/vae_decoder_quantized.serialized.bin";
    model_name_ = "DecoderModel===";
    latent_h = 32;
    latent_w = 32;
    image_h = latent_h * 8;
    image_w = latent_w * 8;

    input_node_dims = {{1, latent_h, latent_w, latent_c}};
    output_node_dims = {{1, image_h, image_w, image_c}};

    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);

    input_norm_ = 1 / 0.18215f;
    input_mean_ = 0.0f;
}

DecoderModel::~DecoderModel() = default;


int DecoderModel::post_process(std::vector <cv::Mat> &outputs) {
    auto res = outputs[0].clone();
    for (int i = 0; i < res.rows * res.cols * 3; ++i) {
        float pix_f = *((float *) res.data + i) * 255.0f;
        *((float *) outputs[0].data + i) = std::max(std::min(pix_f, 255.0f), 0.0f);
    }
    return 0;
}

void DecoderModel::set_latent_size(int latent_size_h, int latent_size_w) {
    LOGI("%s set latent size %d %d", model_name_.c_str(), latent_size_h, latent_size_w);

    latent_h = latent_size_h;
    latent_w = latent_size_w;

    image_h = latent_h * 8;
    image_w = latent_w * 8;

    input_node_dims[0][1] = latent_h;
    input_node_dims[0][2] = latent_w;
    output_node_dims[0][1] = image_h;
    output_node_dims[0][2] = image_w;
    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
}

int DecoderModel::decode(cv::Mat &sample, cv::Mat &res_img) {
    before_run();

    cv::Mat img(cv::Size(image_w, image_h), CV_32FC3);
    std::vector<cv::Mat> inputs{sample}, outputs{img};
    if (inference(inputs, outputs) < 0) {
        return -1;
    }
    outputs[0].convertTo(outputs[0], CV_8UC3);
    res_img = outputs[0].clone();
    after_run();
    return 0;
}
