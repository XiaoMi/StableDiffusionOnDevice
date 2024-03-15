#ifndef STABLEDIFFUSION_DECODERMODEL_H
#define STABLEDIFFUSION_DECODERMODEL_H

#include "onnx/ONNXModel.h"


class DecoderModel : public ONNXModel{

public:
    DecoderModel();

    ~DecoderModel();

    int pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) override;

    int post_process(std::vector<cv::Mat> &outputs) override;

    void set_latent_size(int latent_size_h, int latent_size_w) override;

    cv::Mat decode(const cv::Mat &sample);

private:

};


#endif //STABLEDIFFUSION_DECODERMODEL_H
