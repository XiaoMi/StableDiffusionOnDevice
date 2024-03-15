#pragma once

#include "utils/utils.h"
#include "scheduler/scheduler_euler_a.h"
#include "scheduler/scheduler_dpmpp_2m.h"
#include "scheduler/scheduler_dpmpp_2m_karras.h"
#include "models//UNet.h"

using namespace std;

class DiffusionSolver {
public:
    int load(const string &path, int diffusion_mode = 1);

    void set_latent_size(int latent_size_h, int latent_size_w);

    cv::Mat sampler_txt2img(int seed, int step, const cv::Mat &c, const cv::Mat &uc);

private:
    cv::Mat CFGDenoiser_CompVisDenoiser(const cv::Mat &input, float t, const cv::Mat &cond,
                                        const cv::Mat &uncond);

private:
    const float guidance_scale = 7.5;
    float strength = 0.5;

    int latent_h = 32;
    int latent_w = 32;
    int latent_c = 4;

    int image_w = latent_w * 8;
    int image_h = latent_h * 8;
    int input_sample_size = 1 * latent_c * latent_h * latent_w;

    // 0 not init, 1 txt2img, 2 controlNet;
    int diffusion_mode_ = 0;
    string path_;
    scheduler_base *scheduler = nullptr;

    UNet uNet;
};