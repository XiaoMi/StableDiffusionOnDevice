#ifndef STABLEDIFFUSION_TEXTENCODERMODEL_H
#define STABLEDIFFUSION_TEXTENCODERMODEL_H

#include "onnx/ONNXModel.h"

class TextEncoderModel : public ONNXModel {

public:
    TextEncoderModel();

    ~TextEncoderModel();

    int set_language(int language_mode);

    int post_process(std::vector<cv::Mat> &outputs) override;

    void set_latent_size(int latent_size_h, int latent_size_w) override;

    cv::Mat decode(const std::vector<int> &input_ids);

private:
    int is_ch_;
};


#endif //STABLEDIFFUSION_TEXTENCODERMODEL_H
