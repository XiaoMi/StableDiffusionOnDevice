//
// on 2023/8/7.
//

#ifndef STABLEDIFFUSION_DECODERMODEL_H
#define STABLEDIFFUSION_DECODERMODEL_H

#include "qcom/QCOMModel.h"

class DecoderModel : public QCOMModel{

public:
    DecoderModel();

    ~DecoderModel();


    int post_process(std::vector<cv::Mat> &outputs) override;

    void set_latent_size(int latent_size_h, int latent_size_w) override;

    int decode(cv::Mat &sample,cv::Mat &res_img);

private:

};


#endif //STABLEDIFFUSION_DECODERMODEL_H
