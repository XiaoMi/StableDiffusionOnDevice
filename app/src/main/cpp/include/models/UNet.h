//
// Created by ykfan on 2023/8/8.
//

#ifndef STABLEDIFFUSION_UNET_H
#define STABLEDIFFUSION_UNET_H
#include "qcom/QCOMModel.h"

class UNet : public QCOMModel{
public:
    UNet();

    ~UNet();

    int pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) override;

    int post_process(std::vector<cv::Mat> &outputs) override;

    void set_latent_size(int latent_size_h, int latent_size_w) override;

    int denoise(const cv::Mat &input, float t, const cv::Mat &cond,cv::Mat &denoised);

private:

};


#endif //STABLEDIFFUSION_UNET_H
