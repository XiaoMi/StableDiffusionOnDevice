#include "prompt_solver.h"


int PromptSolver::load(const std::string &path) {
    tokenizer_en.set_language(0);
    int res = textEncoder_en.load(path);
    if (res < 0) {
        LOGE("textEncoder_en load failed!");
        return res;
    }
    LOGI("textEncoder_en load success");

    res = tokenizer_en.load(path);
    if (res < 0) {
        LOGE("CLIPTokenizer load failed!");
        return res;
    }
    LOGI("CLIPTokenizer load success");
    LOGI("PromptSolver init success");
    return 0;
}

int PromptSolver::get_conditioning(const string &prompt_en, const string &default_prompt_en,
                                   cv::Mat &res_cond) {
    return get_conditioning_v2_en(prompt_en, default_prompt_en, res_cond);
}

int PromptSolver::get_conditioning_v2_en(const string &prompt, const string &default_prompt,
                                         cv::Mat &res_cond) {
    textEncoder_en.before_run();

    auto tokens_and_weights = tokenizer_en.tokenize(prompt, 77, true);
    auto tokenized = tokens_and_weights.first;
    auto weights = tokens_and_weights.second;

    vector<int> empty_tokenized_en(77, 49407);
    empty_tokenized_en[0] = 49406;

    if (tokenized == empty_tokenized_en && !default_prompt.empty()) {
        tokens_and_weights = tokenizer_en.tokenize(default_prompt, 77, true);
        tokenized = tokens_and_weights.first;
        weights = tokens_and_weights.second;
    }

    cv::Mat tokens_cond(77, 768, CV_32FC1);
    if (textEncoder_en.encode(tokenized, tokens_cond) < 0)
        return -1;

    res_cond = tokens_cond.clone();

    for (int i = 0; i < res_cond.rows; ++i) {
        for (int j = 0; j < res_cond.cols; ++j) {
            res_cond.at<float>(i, j) = res_cond.at<float>(i, j) * weights[i];
        }
    }

    cv::Scalar original_mean = cv::mean(tokens_cond);
    cv::Scalar res_mean = cv::mean(res_cond);
    LOGI("PromptSolver Scalar len %lu %lu",
         sizeof(original_mean.val) / sizeof(original_mean.val[0]),
         sizeof(res_mean.val) / sizeof(res_mean.val[0]));

    res_cond.convertTo(res_cond, CV_32FC1, original_mean[0] / res_mean[0]);

    textEncoder_en.after_run();
    return 0;
}

int PromptSolver::unload() {
    return textEncoder_en.unload();
}
