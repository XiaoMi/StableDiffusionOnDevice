#ifndef STABLEDIFFUSION_SCHEDULER_DPMPP_2M_H
#define STABLEDIFFUSION_SCHEDULER_DPMPP_2M_H

#include "scheduler_base.h"


class scheduler_dpmpp_2m: public scheduler_base{
public:
    scheduler_dpmpp_2m();

    cv::Mat dpm_solver_first_order_update(cv::Mat model_output, int timestep,int prev_timestep,cv::Mat sample,cv::Mat noise);
    cv::Mat multistep_dpm_solver_second_order_update(std::array<cv::Mat , 2>model_output_list, std::array<int,2> timestep_list,int prev_timestep,cv::Mat sample,cv::Mat noise);
    virtual std::vector<float> set_timesteps(int steps) override;

    virtual cv::Mat scale_model_input(cv::Mat &sample, int step_index) override;

    virtual cv::Mat step(int step_index, cv::Mat &sample_mat, cv::Mat &denoised, cv::Mat &old_noised) override;

    virtual cv::Mat randn_mat(int seed, int height, int width, int is_latent_sample) override;

    virtual float getInitNoiseSigma() override;

    virtual std::vector<float> get_timesteps() override;

    virtual std::vector<float> get_sigmas() override;

    void set_init_sigma(float sigma) override;

private:
    float beta_start = 0.00085f;
    float beta_end = 0.012f;
    std::string beta_schedule = "scaled_linear";
    std::vector<float> trained_betas;
    const int solver_order = 1;
    bool thresholding = false;
    float dynamic_thresholding_ratio = 0.995f;
    float sample_max_value = 1.0f;
    std::string algorithm_type = "sde-dpmsolver++";
    std::string solver_type = "midpoint";
    bool lower_order_final = true;
    bool clip_sample = false;
    float clip_sample_range = 1.0f;
    std::array<cv::Mat,2> model_outputs={};
    int num_train_timesteps = 1000;
    std::vector<float> alphas, betas, alphas_cumprod, timesteps, sigmas, sigmas_total, alpha_ts, sigma_ts, lambda_ts;
    int rand_seed;
    int num_inference_steps = 8;
    float init_noise_sigma = 1.0f;
};


#endif //STABLEDIFFUSION_SCHEDULER_DPMPP_2M_H
