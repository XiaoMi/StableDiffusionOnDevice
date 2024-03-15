#ifndef STABLEDIFFUSION_SCHEDULER_DPMPP_2M_KARRAS_H
#define STABLEDIFFUSION_SCHEDULER_DPMPP_2M_KARRAS_H

#include "scheduler_base.h"


class scheduler_dpmpp_2m_karras: public scheduler_base {
public:
    scheduler_dpmpp_2m_karras();

    virtual std::vector<float> set_timesteps(int steps) override;

    virtual cv::Mat scale_model_input(cv::Mat &sample, int step_index) override;

    virtual cv::Mat step(int step_index, cv::Mat &sample_mat, cv::Mat &denoised, cv::Mat &old_noise) override;

    virtual cv::Mat randn_mat(int seed, int height, int width, int is_latent_sample) override;

    virtual std::vector<float> get_sigmas() override;

    void set_init_sigma(float sigma) override;

private:
    std::string prediction_type = "epsilon";

    int num_train_timesteps = 1000;
    std::vector<float> timesteps, sigmas;

    int num_inference_steps = -1;
    float init_noise_sigma = 14.6f;

};


#endif //STABLEDIFFUSION_SCHEDULER_DPMPP_2M_KARRAS_H
