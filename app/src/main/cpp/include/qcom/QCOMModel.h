//
// on 2023/8/14.
//

#ifndef STABLEDIFFUSION_QCOMMODEL_H
#define STABLEDIFFUSION_QCOMMODEL_H

#include "utils/utils.h"

#include "BuildId.hpp"
#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "PAL/GetOpt.hpp"
#include "QnnSampleApp.hpp"
#include "QnnSampleAppUtils.hpp"

using namespace qnn;
using namespace tools;
using namespace sample_app;

std::vector<size_t> cal_dims_size(const std::vector<std::vector<int64_t>> &dims);

class QCOMModel {
public:
    QCOMModel();

    ~QCOMModel();

    int load(const std::string &path, int device_id = 0);

    int unload();

    // call create session before run because of memory limit
    int before_run();

    int after_run();

    int inference(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs);

    virtual int pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs);

    virtual int post_process(std::vector<cv::Mat> &outputs) = 0;

    virtual void set_latent_size(int latent_size_h, int latent_size_w) = 0;

    std::vector<std::vector<int64_t>> get_output_node_dims();

    std::vector<size_t> get_output_node_sizes();

protected:

    void* sg_backendHandle;
    void* sg_modelHandle;
    bool loadFromCachedBinary = true;
    std::unique_ptr<qnn::tools::sample_app::QnnSampleApp> qnn_net;

    std::string model_path_;
    std::string model_name_;
    std::string model_sub_path_;
    float input_norm_ = 1.0f;
    float input_mean_ = 0.0f;

    int latent_h = -1;
    int latent_w = -1;
    int latent_c = 4;

    int image_h = -1;
    int image_w = -1;
    int image_c = 3;

    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<size_t> input_node_sizes;

    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims;
    std::vector<size_t> output_node_sizes;

    std::vector<uint8_t *> input_node_values;
    std::vector<float *> output_node_values;
};

#endif //STABLEDIFFUSION_QCOMMODEL_H
