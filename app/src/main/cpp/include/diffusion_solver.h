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
    int sampler_txt2img(int seed, int step, cv::Mat &c, cv::Mat &uc, cv::Mat &x_mat);
    int unload();


private:
    int CFGDenoiser_CompVisDenoiser(cv::Mat &input, float t,
                                    const cv::Mat &cond, const cv::Mat &uncond, cv::Mat &denoised);


private:
    const float guidance_scale = 7.5;
    int latent_h = 32;
    int latent_w = 32;
    int latent_c = 4;

    int input_sample_size = 1 * latent_c * latent_h * latent_w;

    // 0 not init, 1 txt2img;
    int diffusion_mode_ = 0;
    string path_;
    scheduler_base *scheduler = nullptr;
    UNet uNet;
};