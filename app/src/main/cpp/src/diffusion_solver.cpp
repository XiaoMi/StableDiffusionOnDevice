#include "diffusion_solver.h"

int DiffusionSolver::load(const string &path, int diffusion_mode) {
    int res = 0;

    if (diffusion_mode_ == diffusion_mode) {
        LOGI("DiffusionSolver keep mode %d!", diffusion_mode);
        return res;
    }

    if (diffusion_mode_ == 0) {
        scheduler = new scheduler_dpmpp_2m();
        scheduler->set_timesteps(20);
        path_ = path;
    }

    if (diffusion_mode == 1) {
        res = uNet.load(path);
        if (res < 0) {
            LOGE("uNet load fail!");
            return res;
        }
        diffusion_mode_ = 1;
    }

    LOGI("DiffusionSolver load success with mode %d!", diffusion_mode);
    return res;
}

void DiffusionSolver::set_latent_size(int latent_size_h, int latent_size_w) {
    LOGI("DiffusionSolver set latent size %d(h) x %d(w)", latent_size_h, latent_size_w);

    latent_h = latent_size_h;
    latent_w = latent_size_w;

    image_w = latent_w * 8;
    image_h = latent_h * 8;
    input_sample_size = 1 * latent_c * latent_h * latent_w;

    uNet.set_latent_size(latent_size_h, latent_size_w);
}


cv::Mat DiffusionSolver::CFGDenoiser_CompVisDenoiser(const cv::Mat &input, float t,
                                                     const cv::Mat &cond, const cv::Mat &uncond) {

    auto denoised_cond = uNet.decode(input, t, cond);
    auto denoised_uncond = uNet.decode(input, t, uncond);

    auto *u_ptr = reinterpret_cast<float *>(denoised_uncond.data);
    auto *c_ptr = reinterpret_cast<float *>(denoised_cond.data);
    for (int hwc = 0; hwc < input_sample_size; hwc++) {
        (*u_ptr) = (*u_ptr) + guidance_scale * ((*c_ptr) - (*u_ptr));
        u_ptr++;
        c_ptr++;
    }

    return denoised_uncond;
}

cv::Mat DiffusionSolver::sampler_txt2img(int seed, int step, const cv::Mat &c, const cv::Mat &uc) {
    load(path_, 1);

    uNet.before_run();
    // init
    auto timesteps = scheduler->set_timesteps(step);
    cv::Mat x_mat = scheduler->randn_mat(seed % 100, latent_h, latent_w, 1); //generateLatentSample
    cv::Mat old_noised(cv::Size(latent_h, latent_w), CV_32FC4);

    for (int i = 0; i < timesteps.size(); i++) {
        cv::Mat latent_input = scheduler->scale_model_input(x_mat, i);  //latentModelInput
        cv::Mat denoised = CFGDenoiser_CompVisDenoiser(latent_input, timesteps[i], c, uc);
        x_mat = scheduler->step(i, x_mat, denoised, old_noised);
    }
    uNet.after_run();
    return x_mat;
}