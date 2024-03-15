#include "scheduler/scheduler_dpmpp_2m.h"
#include "random"

std::vector<float> linspace_pp(float start, float end, int num_points) {
    std::vector<float> result(num_points);

    for (int i = 0; i < num_points; i++) {
        result[i] = start + (end - start) * i / (num_points - 1);
    }

    return result;
}

scheduler_dpmpp_2m::scheduler_dpmpp_2m() {
    if (beta_schedule == "linear") {
        auto array = linspace_pp(beta_start, beta_end, num_train_timesteps);
        betas.swap(array);
    } else if (beta_schedule == "scaled_linear") {
        auto array = linspace_pp(pow(beta_start, 0.5), pow(beta_end, 0.5), num_train_timesteps);
        for (float i: array) {
            betas.push_back(pow(i, 2));
        }
    }

    for (float beta: betas) {
        float alpha = 1.0f - beta;
        alphas.push_back(alpha);
        alphas_cumprod.push_back(alpha * (alphas_cumprod.empty() ? 1.0f : alphas_cumprod.back()));
    }

    for (float value: alphas_cumprod) {
        alpha_ts.push_back(sqrt(value));
        sigma_ts.push_back(sqrt(1. - value));
    }

    for (int i = 0; i < alphas_cumprod.size(); i++) {
        lambda_ts.push_back(log(alpha_ts[i]) - log(sigma_ts[i]));
    }
    vec_print("alpha_ts", alpha_ts);
    LOGI("%d", alpha_ts.size());
    for (float alpha: alphas_cumprod) {
        sigmas_total.push_back(sqrt((1 - alpha) / alpha));
    }
    timesteps = linspace_pp(0, num_train_timesteps - 1, num_train_timesteps);

}

std::vector<float> scheduler_dpmpp_2m::set_timesteps(int steps) {
    sigmas.clear();
    timesteps.clear();

    num_inference_steps = steps;
    timesteps = linspace_pp(0, num_train_timesteps - 1, num_inference_steps + 1);
    std::reverse(timesteps.begin(), timesteps.end());
    timesteps.pop_back();
    for (float &timestep: timesteps) {
        timestep = round(timestep);
    }

    vec_print("timesteps", timesteps);
    return timesteps;
}

cv::Mat scheduler_dpmpp_2m::scale_model_input(cv::Mat &sample, int step_index) {
    auto input_latent = sample.clone();
    return input_latent;
}


cv::Mat scheduler_dpmpp_2m::dpm_solver_first_order_update(cv::Mat model_output, int timestep,
                                                          int prev_timestep, cv::Mat sample,
                                                          cv::Mat noise) {
    float sigma_t = sigma_ts[prev_timestep];
    float alpha_t = alpha_ts[prev_timestep];
    float lambda_t = lambda_ts[prev_timestep];
    float sigma_s = sigma_ts[timestep];
    float alpha_s = alpha_ts[timestep];
    float lambda_s = lambda_ts[timestep];
    float h = lambda_t - lambda_s;
    cv::Mat x_t = model_output.clone();
    if (algorithm_type == "dpmsolver++") {
        for (int i = 0; i < model_output.rows * model_output.cols * model_output.channels(); i++) {
            *((float *) x_t.data + i) = *((float *) sample.data + i) * (sigma_t / sigma_s) -
                                        *((float *) model_output.data + i) * alpha_t *
                                        (exp(-h) - 1.0);

        }
    } else if (algorithm_type == "sde-dpmsolver++") {
        for (int i = 0; i < model_output.rows * model_output.cols * model_output.channels(); i++) {
            *((float *) x_t.data + i) =
                    *((float *) sample.data + i) * (sigma_t / sigma_s * exp(-h)) +
                    *((float *) model_output.data + i) * alpha_t * (1 - exp(-2 * h)) +
                    *((float *) noise.data + i) * sigma_t * sqrt(1 - exp(-2 * h));

        }

    }
    return x_t;
}

cv::Mat scheduler_dpmpp_2m::multistep_dpm_solver_second_order_update(
        std::array<cv::Mat, 2> model_output_list,
        std::array<int, 2> timestep_list,
        int prev_timestep,
        cv::Mat sample,
        cv::Mat noise) {
    int t = prev_timestep;
    int s0 = timestep_list[1];
    int s1 = timestep_list[0];
    cv::Mat m0 = model_output_list[1];
    cv::Mat m1 = model_output_list[0];
    float lambda_t = lambda_ts[t];
    float lambda_s0 = lambda_ts[s0];
    float lambda_s1 = lambda_ts[s1];
    float alpha_t = alpha_ts[t];
    float alpha_s0 = alpha_ts[s0];
    float sigma_t = sigma_ts[t];
    float sigma_s0 = sigma_ts[s0];
    float h = lambda_t - lambda_s0;
    float h_0 = lambda_s0 - lambda_s1;
    float r0 = r0 = h_0 / h;
    cv::Mat D0 = m0;
    cv::Mat D1 = m1.clone();
    for (int i = 0; i < m0.rows * m0.cols * m0.channels(); i++) {
        *((float *) D1.data + i) =
                1 / r0 * (*((float *) m0.data + i) - *((float *) m1.data + i));

    }
    cv::Mat x_t = m0.clone();
    if (algorithm_type == "dpmsolver++" && solver_type == "midpoint") {
        for (int i = 0; i < m0.rows * m0.cols * m0.channels(); i++) {
            *((float *) x_t.data + i) = *((float *) sample.data + i) * (sigma_t / sigma_s0) -
                                        *((float *) D0.data + i) * alpha_t *
                                        (exp(-h) - 1.0) -
                                        *((float *) D1.data + i) * 0.5 * alpha_t *
                                        (exp(-h) - 1.0);
        }
    } else if (algorithm_type == "sde-dpmsolver++" && solver_type == "midpoint") {
        for (int i = 0; i < m0.rows * m0.cols * m0.channels(); i++) {
            *((float *) x_t.data + i) =
                    *((float *) sample.data + i) * (sigma_t / sigma_s0 * exp(-h)) +
                    *((float *) D0.data + i) * alpha_t * (1 - exp(-2 * h)) +
                    *((float *) D1.data + i) * alpha_t * (1 - exp(-2 * h)) / (1 - 2 * h) +
                    *((float *) noise.data + i) * sigma_t * sqrt(1 - exp(-2 * h));
        }
    }
    return x_t;
}

cv::Mat scheduler_dpmpp_2m::step(int step_index, cv::Mat &sample_mat, cv::Mat &denoised,
                                 cv::Mat &old_denoised) {
    int timestep = timesteps[step_index];
    int prev_timestep = step_index == timesteps.size() - 1 ? 0 : timesteps[step_index + 1];

    auto *x_ptr = reinterpret_cast<float *>(sample_mat.data);
    auto *d_ptr = reinterpret_cast<float *>(denoised.data);


    int sample_length = sample_mat.cols * sample_mat.rows * sample_mat.channels();
    assert(sample_mat.channels() == 4);


    float sigma_t = sigma_ts[timestep];
    float alpha_t = alpha_ts[timestep];
    cv::Mat output = sample_mat.clone();
    for (int hwc = 0; hwc < sample_length; hwc++) {
        float sample = *(x_ptr + hwc);
        float model_output = *(d_ptr + hwc);
        *((float *) output.data + hwc) = (sample - sigma_t * model_output) / alpha_t;
    }
    if (solver_order > 1) {
        model_outputs[1] = model_outputs[0];
    }
    model_outputs[0] = output;
    cv::Mat noise = randn_mat(rand_seed - step_index - 1, sample_mat.rows, sample_mat.cols, 0);
    if (solver_order == 1 || step_index == 0) {
        output = dpm_solver_first_order_update(output, timestep, prev_timestep, sample_mat,
                                               noise);
    }
    if (solver_order == 2 && step_index > 0) {
        std::array<int, 2> timestep_list{(int) timesteps[step_index - 1], timestep};
        output = multistep_dpm_solver_second_order_update(model_outputs, timestep_list,
                                                          prev_timestep, sample_mat,
                                                          noise);
    }
    return output;
}

float scheduler_dpmpp_2m::getInitNoiseSigma() {
    return init_noise_sigma;
}

cv::Mat scheduler_dpmpp_2m::randn_mat(int seed, int height, int width, int is_latent_sample) {
    cv::Mat cv_x(cv::Size(width, height), CV_32FC4);
    rand_seed = seed;
    cv::RNG rng(seed);
    rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);

    if (is_latent_sample) {
        cv_x.convertTo(cv_x, CV_32FC4, init_noise_sigma, 0.0f);
        std::ranlux48 engine(seed);
        std::uniform_real_distribution<double> distrib(0.0, 1.0);
        for (int i = 0; i < cv_x.cols * cv_x.rows * cv_x.channels(); i++) {
            double u1 = distrib(engine);
            double u2 = distrib(engine);
            double radius = sqrt(-2.0f * log(u1));
            double theta = 2.0 * 3.141592653589793 * u2;
            double standardNormalRand = radius * cos(theta);
            *((float *) cv_x.data + i) = (float) (standardNormalRand * init_noise_sigma);
        }
    }
    return cv_x;
}

void scheduler_dpmpp_2m::set_init_sigma(float sigma) {
    init_noise_sigma = sigma;
}

std::vector<float> scheduler_dpmpp_2m::get_timesteps() {
    return timesteps;
}

std::vector<float> scheduler_dpmpp_2m::get_sigmas() {
    return sigmas;
}



