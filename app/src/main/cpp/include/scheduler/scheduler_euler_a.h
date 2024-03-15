#ifndef STABLEDIFFUSION_SCHEDULER_EULER_A_H
#define STABLEDIFFUSION_SCHEDULER_EULER_A_H

#include "scheduler_base.h"

class scheduler_euler_a : public scheduler_base {
public:
    scheduler_euler_a();

    virtual std::vector<float> set_timesteps(int steps) override;

    virtual cv::Mat scale_model_input(cv::Mat &sample, int step_index) override;

    virtual cv::Mat step(int step_index, cv::Mat &sample_mat, cv::Mat &denoised, cv::Mat &old_noised) override;

    virtual float getInitNoiseSigma() override;

    virtual cv::Mat randn_mat(int seed, int height, int width, int is_latent_sample) override;

    virtual std::vector<float> get_timesteps() override;

    virtual std::vector<float> get_sigmas() override;

    virtual void set_init_sigma(float sigma)override;

private:
    float beta_start = 0.00085f;
    float beta_end = 0.012f;
    std::string beta_schedule = "scaled_linear";
    std::vector<float> trained_betas;
    int solver_order = 2;
    std::string prediction_type = "epsilon";
    bool thresholding = false;
    float dynamic_thresholding_ratio = 0.995f;
    float sample_max_value = 1.0f;
    std::string algorithm_type = "euler_a";
    std::string solver_type = "midpoint";
    bool lower_order_final = true;
    bool clip_sample = false;
    float clip_sample_range = 1.0f;

    int num_train_timesteps = 1000;
    std::vector<float> alphas, betas, alphas_cumprod, timesteps, sigmas, sigmas_total;

    int num_inference_steps = 8;
    float init_noise_sigma = 1.0f;
//    std::vector<float> sigmas{
//            14.6146555,
//            6.6780214,
//            3.5220659,
//            2.0606437,
//            1.2768335,
//            0.7912614,
//            0.4396673,
//            0.029167533,
//            0.0f
//    };
//
//
//    std::vector<float> timestep{
//            999.0000,
//            856.2857,
//            713.5714,
//            570.8571,
//            428.1429,
//            285.4286,
//            142.7143,
//            0,
//    };

};


#endif //STABLEDIFFUSION_SCHEDULER_EULER_A_H
