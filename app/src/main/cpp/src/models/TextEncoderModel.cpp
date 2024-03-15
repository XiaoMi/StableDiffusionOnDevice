#include "models/TextEncoderModel.h"


TextEncoderModel::TextEncoderModel() {
    model_name_ = "TextEncoderModel===";

    model_sub_path_ = std::string("/stable_diffusion/qnn_model_") + QCOM_VERSION +
                      "/text_encoder_quantized.serialized.bin";
    input_node_dims = {{1, 77}};
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

int TextEncoderModel::encode(const std::vector<int> &input_ids, cv::Mat &cond) {
    cv::Mat output_cond((int) output_node_dims[0][1],
                        (int) output_node_dims[0][2],
                        CV_32FC1), output_tmp;
    cv::Mat input_prompt((int) input_node_dims[0][1], 1, CV_32FC1);
    for (int i = 0; i < input_node_sizes[0]; ++i) {
        *((float *) input_prompt.data + i) = (float) input_ids[i];
    }
    std::vector<cv::Mat> inputs{input_prompt};
    std::vector<cv::Mat> outputs{output_cond, output_tmp};
    if (inference(inputs, outputs) < 0)
        return -1;
    cond = outputs[0].clone();
    return 0;
}
