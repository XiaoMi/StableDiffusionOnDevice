#include "scheduler/scheduler_dpmpp_2m_karras.h"
#include "random"


std::vector<float> linspace_k(float start, float end, int num){

    std::vector<float> linspaced;

    float delta = (end - start) / ((float)(num - 1));
    for(int i=0; i < num-1; ++i){
        linspaced.push_back(start + delta * (float)i);
    }
    linspaced.push_back(end);
    return linspaced;
}

std::vector<float> get_sigmas(std::vector<float> ramp, float min_inv_rho, float max_inv_rho, float rho){

    std::vector<float> sigmas_f;
    for (float r : ramp){
        r = pow(max_inv_rho + r * (min_inv_rho - max_inv_rho),  rho);
        sigmas_f.push_back(r);
    }
    sigmas_f.push_back(0.f);

    return sigmas_f;
}

std::vector<float> get_sigmas_karras(int n, float sigma_min, float sigma_max, float rho=7.){

    std::vector<float> sigmas;
    std::vector<float> ramp;

    ramp = linspace_k(0, 1, n);  //返回一个一维的tensor,线性间距向量

    float min_inv_rho = pow(sigma_min, (1 / rho));
    float max_inv_rho = pow(sigma_max, (1 / rho));

    sigmas =  get_sigmas(ramp, min_inv_rho, max_inv_rho, rho);
    return sigmas;
}

scheduler_dpmpp_2m_karras::scheduler_dpmpp_2m_karras() = default;

std::vector<float> scheduler_dpmpp_2m_karras::set_timesteps(int steps) {
    if (num_inference_steps == steps){
        return timesteps;
    }

    sigmas.clear();
    timesteps.clear();

    num_inference_steps = steps;
    timesteps = linspace_k(0, num_train_timesteps - 1, num_inference_steps);
    std::reverse(timesteps.begin(), timesteps.end());

    sigmas = get_sigmas_karras(num_inference_steps, 0.2, 14.6, 7.);
    init_noise_sigma = *std::max_element(sigmas.begin(), sigmas.end());
    vec_print("timesteps", timesteps);
    vec_print("sigmas", sigmas);

    LOGI("init_noise_sigma %f", init_noise_sigma);

    return timesteps;
}

cv::Mat scheduler_dpmpp_2m_karras::scale_model_input(cv::Mat &sample, int step_index) {
    float sigma = sigmas[step_index];
    auto input_latent = sample.clone();

    input_latent.convertTo(input_latent, CV_32FC4, 1 / pow(pow(sigma, 2) + 1, 0.5), 0);  //c_in

    return input_latent;
}


float sigma_fn(float s){
    return exp(-s);
}

float t_fn(float s){
    return -log(s);
}


cv::Mat scheduler_dpmpp_2m_karras::step(int step_index, cv::Mat &sample_mat, cv::Mat &denoised, cv::Mat &old_denoised) {

    float sigma = sigmas[step_index];
    float t = t_fn(sigmas[step_index]);
    float t_next = t_fn(sigmas[step_index + 1]);
    float h = t_next - t;

    float t_min = fmin(sigma_fn(t_next), sigma_fn(t));
    float t_max = fmax(sigma_fn(t_next), sigma_fn(t));

    auto *x_ptr = reinterpret_cast<float *>(sample_mat.data);
    auto *d_ptr = reinterpret_cast<float *>(denoised.data);
    auto *l_ptr = reinterpret_cast<float *>(old_denoised.data);

    int sample_length = sample_mat.rows * sample_mat.cols * sample_mat.channels();
    for (int hwc = 0; hwc < sample_length; hwc++) {
        float sample = *x_ptr;
        float model_output = *d_ptr;
        float old_noise = *l_ptr;
        float pred_original_sample = 0.0f;

        if (prediction_type == "epsilon") {
            pred_original_sample = sample - sigma * model_output;
        } else if (prediction_type == "v_prediction") {
            pred_original_sample = model_output * (-sigma / pow((pow(sigma, 2) + 1), 0.5)) +
                                   (sample / (pow(sigma, 2) + 1));
        }

        if (step_index==0 || sigmas[step_index + 1] == 0){
            sample = (t_min / t_max) * sample - expm1(-h) * pred_original_sample;
        }else{
            float h_last = t - t_fn(sigmas[step_index - 1]);
            float h_min = fmin(h_last, h);
            float h_max = fmax(h_last, h);
            float r = h_max / h_min;
            float h_d = (h_max + h_min) / 2;
            float denoised_d = (1 + 1 / (2 * r)) * pred_original_sample - (1 / (2 * r)) * old_noise;
            sample = (t_min / t_max) * sample - expm1(-h_d) * denoised_d;

        }

        *l_ptr = pred_original_sample;
        *x_ptr = sample;
        x_ptr++;
        d_ptr++;
        l_ptr++;
    }

    return sample_mat;
}

cv::Mat scheduler_dpmpp_2m_karras::randn_mat(int seed, int height, int width, int is_latent_sample) {
    cv::Mat cv_x(cv::Size(width, height), CV_32FC4);
    cv::RNG rng(seed);
    rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);

    if (is_latent_sample) {
        cv_x.convertTo(cv_x, CV_32FC4, init_noise_sigma, 0.0f);
    }

    return cv_x;
}

std::vector<float> scheduler_dpmpp_2m_karras::get_sigmas() {
    return sigmas;
}

void scheduler_dpmpp_2m_karras::set_init_sigma(float sigma){
    init_noise_sigma = sigma;
}