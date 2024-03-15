//
// on 2023/8/8.
//

#ifndef STABLEDIFFUSION_TEXTENCODERMODEL_H
#define STABLEDIFFUSION_TEXTENCODERMODEL_H
#include "qcom/QCOMModel.h"

#define USE_DL_QCOM_TEXT_ENCODER

class TextEncoderModel : public QCOMModel {

public:
    TextEncoderModel();

    ~TextEncoderModel();

    int post_process(std::vector<cv::Mat> &outputs) override;

    void set_latent_size(int latent_size_h, int latent_size_w) override;

    int encode(const std::vector<int> &input_ids,cv::Mat &cond);

private:
};


#endif //STABLEDIFFUSION_TEXTENCODERMODEL_H
