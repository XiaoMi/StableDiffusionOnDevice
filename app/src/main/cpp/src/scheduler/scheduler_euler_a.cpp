#include "scheduler/scheduler_euler_a.h"

std::vector<float> linspace(float start, float end, int num_points) {
    std::vector<float> result(num_points);
    float step = (end - start) / (float) (num_points - 1);

    for (int i = 0; i < num_points; ++i) {
        result[i] = start + i * step;
    }

    return result;
}

std::vector<float> arange(float start, float stop, float step = 1) {
    std::vector<float> result;
    for (float i = start; i < stop; i += step) {
        result.push_back(i);
    }
    return result;
}

std::vector<float>
interp(const std::vector<float> &input, std::vector<float> &x, std::vector<float> &y) {
    std::vector<float> output;

    for (float input_value: input) {
        if (input_value <= x[0]) {
            output.push_back(y[0]);
            continue;
        }
        if (input_value >= x[x.size() - 1]) {
            output.push_back(y[x.size() - 1]);
            continue;
        }

        for (int i = 0; i < x.size() - 1; ++i) {
            if (x[i] <= input_value && input_value <= x[i + 1]) {
                float slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
                float res_value = y[i] + slope * (input_value - x[i]);
                output.push_back(res_value);
                break;
            }
        }
    }

    return output;
}


scheduler_euler_a::scheduler_euler_a() {
    if (beta_schedule == "linear") {
        auto array = linspace(beta_start, beta_end, num_train_timesteps);
        betas.swap(array);
    } else if (beta_schedule == "scaled_linear") {
        auto array = linspace(pow(beta_start, 0.5), pow(beta_end, 0.5), num_train_timesteps);
        for (float i: array) {
            betas.push_back(pow(i, 2));
        }
    }

    for (float beta: betas) {
        float alpha = 1.0f - beta;
        alphas.push_back(alpha);
        alphas_cumprod.push_back(alpha * (alphas_cumprod.empty() ? 1.0f : alphas_cumprod.back()));
    }

    for (float alpha: alphas_cumprod) {
        sigmas_total.push_back(sqrt((1 - alpha) / alpha));
    }

    init_noise_sigma = *std::max_element(sigmas_total.begin(), sigmas_total.end());
}

std::vector<float> scheduler_euler_a::set_timesteps(int steps) {
    sigmas.clear();
    timesteps.clear();

    num_inference_steps = steps;
    timesteps = linspace(0, num_train_timesteps - 1, num_inference_steps);
    std::reverse(timesteps.begin(), timesteps.end());
    std::vector<float> sigmas_arange = arange(0, sigmas_total.size(), 1);
    sigmas = interp(timesteps, sigmas_arange, sigmas_total);
    sigmas.push_back(0.0f);

    LOGI("init_noise_sigma %f", init_noise_sigma);
    vec_print("timesteps", timesteps);
    vec_print("sigmas", sigmas_total);
    vec_print("sigmas", sigmas);
    return timesteps;
}

cv::Mat scheduler_euler_a::scale_model_input(cv::Mat &sample, int step_index) {
    float sigma = sigmas[step_index];
    auto input_latent = sample.clone();
    input_latent.convertTo(input_latent, CV_32FC4, 1 / pow(pow(sigma, 2) + 1, 0.5), 0);
    return input_latent;
}

cv::Mat scheduler_euler_a::step(int step_index, cv::Mat &sample_mat, cv::Mat &denoised, cv::Mat &old_noised) {
    srand(time(nullptr) + 1);
    cv::Mat randn = randn_mat(rand() % 1000, sample_mat.rows, sample_mat.cols, 0);
    float sigma = sigmas[step_index];
    float sigma_from = sigmas[step_index];
    float sigma_to = sigmas[step_index + 1];
    float sigma_up = sqrt(pow(sigma_to, 2) *
                          (pow(sigma_from, 2) - pow(sigma_to, 2))
                          / pow(sigma_from, 2));
    float sigma_down = sqrt(pow(sigma_to, 2) - pow(sigma_up, 2));
    float dt = sigma_down - sigma;
    int sample_length = denoised.rows * denoised.cols * denoised.channels();
    assert(sample_mat.channels() == 4);
    cv::Mat prev(cv::Size(denoised.cols, denoised.rows), CV_32FC4);
    auto *x_ptr = reinterpret_cast<float *>(sample_mat.data);
    auto *d_ptr = reinterpret_cast<float *>(denoised.data);
    auto *p_ptr = reinterpret_cast<float *>(prev.data);
    auto *r_ptr = reinterpret_cast<float *>(randn.data);
    for (int hwc = 0; hwc < sample_length; hwc++) {
        float sample = *x_ptr;
        float model_output = *d_ptr;
        float noise = *r_ptr;
        float pred_original_sample = 0.0f;
        if (prediction_type == "epsilon") {
            pred_original_sample = sample - sigma * model_output;
        } else if (prediction_type == "v_prediction") {
            pred_original_sample = model_output * (-sigma / pow((pow(sigma, 2) + 1), 0.5)) +
                                   (sample / (pow(sigma, 2) + 1));
        }
        float derivative = (sample - pred_original_sample) / sigma;
        float prev_sample = sample + derivative * dt;
        prev_sample = prev_sample + noise * sigma_up;
        *p_ptr = prev_sample;
        x_ptr++;
        d_ptr++;
        p_ptr++;
        r_ptr++;
    }

    return prev;
}

float scheduler_euler_a::getInitNoiseSigma() {
    return init_noise_sigma;
}

cv::Mat scheduler_euler_a::randn_mat(int seed, int height, int width, int is_latent_sample) {
    cv::Mat cv_x(cv::Size(width, height), CV_32FC4);

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

std::vector<float> scheduler_euler_a::get_timesteps() {
    return timesteps;
}

std::vector<float> scheduler_euler_a::get_sigmas() {
    return sigmas;
}

void scheduler_euler_a::set_init_sigma(float sigma){
    init_noise_sigma = sigma;
}

