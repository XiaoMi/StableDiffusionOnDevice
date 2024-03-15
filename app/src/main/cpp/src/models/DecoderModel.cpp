#include "models/DecoderModel.h"

DecoderModel::DecoderModel() {
    model_sub_path_ = "/stable_diffusion/onnx/vae_decoder_fp16.ort";
    model_name_ = "DecoderModel===";

    latent_h = 32;
    latent_w = 32;

    image_h = latent_h * 8;
    image_w = latent_w * 8;

    input_node_names = {"latent_sample"};
    input_node_dims = {{1, latent_c, latent_h, latent_w}};
    output_node_names = {"sample"};
    output_node_dims = {{1, image_c, image_h, image_w}};

    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);

    input_norm_ = 1 / 0.18215f;
    input_mean_ = 0.0f;
}

DecoderModel::~DecoderModel() = default;

int DecoderModel::pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) {
    ONNXModel::pre_process(inputs, outputs);
    auto sample = inputs[0].clone();
    sample.convertTo(sample, CV_32FC4, input_norm_, input_mean_);
    convert_hwc_to_chw(reinterpret_cast<float *>(sample.data),
                       input_node_values[0].data(),
                       latent_h,
                       latent_w,
                       latent_c);
    return 0;
}

int DecoderModel::post_process(std::vector<cv::Mat> &outputs) {
    auto res = outputs[0].clone();
    for (int h = 0; h < res.rows; ++h) {
        for (int w = 0; w < res.cols; ++w) {
            for (int c = 0; c < 3; ++c) {
                int dst_idx = (h * res.cols + w) * 3 + c;
                int src_idx = (c * res.rows + h) * res.cols + w;
                auto *src_ptr = (float *) res.data;
                auto *dst_ptr = (float *) outputs[0].data;
                float pix_f = (src_ptr[src_idx] / 2.0f + 0.5f) * 255.0f;
                // todo why opencv 4.6.0 use fc3 range 0-255
                dst_ptr[dst_idx] = std::max(std::min(pix_f, 255.0f), 0.0f);
            }
        }
    }
    return 0;
}

void DecoderModel::set_latent_size(int latent_size_h, int latent_size_w) {
    LOGI("%s set latent size %d %d", model_name_.c_str(), latent_size_h, latent_size_w);

    latent_h = latent_size_h;
    latent_w = latent_size_w;

    image_h = latent_h * 8;
    image_w = latent_w * 8;

    input_node_dims[0][2] = latent_h;
    input_node_dims[0][3] = latent_w;
    output_node_dims[0][2] = image_h;
    output_node_dims[0][3] = image_w;
    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
}

cv::Mat DecoderModel::decode(const cv::Mat &sample) {
    before_run();

    cv::Mat img(cv::Size(image_w, image_h), CV_32FC3);
    std::vector<cv::Mat> inputs{sample}, outputs{img};
    inference(inputs, outputs);
    outputs[0].convertTo(outputs[0], CV_8UC3);

    after_run();
    return outputs[0];
}
