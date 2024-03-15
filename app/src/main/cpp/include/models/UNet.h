#ifndef STABLEDIFFUSION_UNET_H
#define STABLEDIFFUSION_UNET_H
#include "onnx/ONNXModel.h"

class UNet : public ONNXModel{

public:
    UNet();

    ~UNet();

    int pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) override;

    int post_process(std::vector<cv::Mat> &outputs) override;

    void set_latent_size(int latent_size_h, int latent_size_w) override;

    cv::Mat decode(const cv::Mat &input, float t, const cv::Mat &cond);

private:

};


#endif //STABLEDIFFUSION_UNET_H
