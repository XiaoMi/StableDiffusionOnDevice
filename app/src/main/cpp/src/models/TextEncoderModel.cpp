#include "models/TextEncoderModel.h"




TextEncoderModel::TextEncoderModel() {
    model_name_ = "TextEncoderModel===";

    input_node_names = {"input_ids",};
    input_node_dims = {{1, 77}};

    output_node_names = {"last_hidden_state", "pooler_output"};
    output_node_dims = {{1, 77, 768},
                        {1, 77, 768}};

    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
}

TextEncoderModel::~TextEncoderModel() = default;

int TextEncoderModel::post_process(std::vector<cv::Mat> &outputs) {
    return 0;
}

void TextEncoderModel::set_latent_size(int latent_size_h, int latent_size_w) {

}

cv::Mat TextEncoderModel::decode(const std::vector<int> &input_ids) {
    cv::Mat output_cond((int) output_node_dims[0][1],
                        (int) output_node_dims[0][2],
                        CV_32FC1), output_tmp;

    std::vector<std::vector<int>> inputs{input_ids};
    std::vector<cv::Mat> outputs{output_cond, output_tmp};
    inference(inputs, outputs);
    return outputs[0];
}

int TextEncoderModel::set_language(int language_mode) {
    model_sub_path_ = "/stable_diffusion/onnx/text_encoder_fp16.ort";
    return 0;
}
