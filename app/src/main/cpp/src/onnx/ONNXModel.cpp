#include "../../include/onnx/ONNXModel.h"

std::vector<size_t> cal_dims_size(const std::vector<std::vector<int64_t>> &dims) {
    std::vector<size_t> res;
    for (auto now_dim: dims) {
        size_t now_size = std::accumulate(now_dim.begin(), now_dim.end(), 1,
                                          std::multiplies<int64_t>());
        res.push_back(now_size);
    }
    return res;
}

ONNXModel::ONNXModel() = default;

ONNXModel::~ONNXModel() {
    LOGI("%s model delete!", model_name_.c_str());

    if (env_) {
        delete env_;
        env_ = nullptr;
    }

    if (session_) {
        delete session_;
        session_ = nullptr;
    }
}

int ONNXModel::load(const std::string &path, int device_id) {
    model_path_ = path + model_sub_path_;

    env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    LOGI("%s load success", model_name_.c_str());

    input_node_sizes = cal_dims_size(input_node_dims);
    output_node_sizes = cal_dims_size(output_node_dims);
    return 0;
}


int ONNXModel::before_run() {
    if (!session_) {
        LOGI("%s load model %s", model_name_.c_str(), model_path_.c_str());
        if (!file_exists(model_path_)){
            LOGE("%s not exist!", model_path_.c_str());
            return -1;
        }

        Ort::SessionOptions session_options;
        session_ = new Ort::Session(*env_, model_path_.c_str(), session_options);
    }
    return 0;
}

int ONNXModel::after_run() {
    delete session_;
    session_ = nullptr;
    return 0;
}

int ONNXModel::inference(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) {
    pre_process(inputs, outputs);
    assert(input_node_values.size() == input_node_dims.size());
    assert(output_node_values.size() ==  output_node_dims.size());

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // create input tensor object from data values
    std::vector<Ort::Value> ort_inputs;
    for (int i = 0; i < input_node_values.size(); ++i) {
        auto input_node_dim = input_node_dims[i];
        Ort::Value input_sample_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                         input_node_values[i].data(),
                                                                         input_node_values[i].size(),
                                                                         input_node_dim.data(),
                                                                         input_node_dim.size());
        assert(input_sample_tensor.IsTensor());
        ort_inputs.push_back(std::move(input_sample_tensor));
    }
    LOGI("%s create input tensor success!", model_name_.c_str());

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_node_names.data(),
                                        ort_inputs.data(),
                                        ort_inputs.size(),
                                        output_node_names.data(),
                                        output_node_names.size());
    LOGI("%s run success!", model_name_.c_str());

    for (int i = 0; i < output_node_values.size(); ++i) {
        if (nullptr == output_node_values.at(i)) {
            continue;
        }
        auto *output_tmp = output_tensors[i].GetTensorMutableData<float>();
        memcpy(output_node_values.at(i), output_tmp, output_node_sizes[i] * sizeof(float));
    }

    post_process(outputs);
    input_node_values.clear();
    output_node_values.clear();
    return 0;
}

int ONNXModel::inference(const std::vector<std::vector<int>> &inputs_const,
                         std::vector<cv::Mat> &outputs) {
    auto inputs = inputs_const;
    output_node_values.push_back((float *)outputs[0].data);
    output_node_values.push_back((float *)outputs[1].data);
    assert(inputs.size() == input_node_dims.size());
    assert(output_node_values.size() == output_node_dims.size());

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // create input tensor object from data values
    std::vector<Ort::Value> ort_inputs;
    for (int i = 0; i < inputs.size(); ++i) {
        auto input_node_dim = input_node_dims[i];
        Ort::Value input_sample_tensor = Ort::Value::CreateTensor<int>(memory_info,
                                                                       inputs[i].data(),
                                                                       inputs[i].size(),
                                                                       input_node_dim.data(),
                                                                       input_node_dim.size());
        assert(input_sample_tensor.IsTensor());
        ort_inputs.push_back(std::move(input_sample_tensor));
    }
    LOGI("%s create input tensor success!", model_name_.c_str());

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_node_names.data(),
                                        ort_inputs.data(),
                                        ort_inputs.size(),
                                        output_node_names.data(),
                                        output_node_names.size());
    LOGI("%s run success!", model_name_.c_str());

    // Get pointer to output tensor float values
    for (int i = 0; i < output_node_values.size(); ++i) {
        if (nullptr == output_node_values.at(i)) {
            continue;
        }
        auto *output_tmp = output_tensors[i].GetTensorMutableData<float>();
        memcpy(output_node_values.at(i), output_tmp, output_node_sizes[i] * sizeof(float));
    }

    post_process(outputs);
    input_node_values.clear();
    output_node_values.clear();
    return 0;
}

int ONNXModel::pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) {
    for (int i = 0; i < inputs.size(); ++i) {
        auto input_node_size = input_node_sizes[i];
        std::vector<float> input_sample_values(input_node_size);
        input_node_values.push_back(std::move(input_sample_values));
        memcpy(input_node_values[i].data(), inputs[i].data, input_node_size * sizeof(float));
    }

    assert(outputs.size() == output_node_names.size());
    for (auto &output_node: outputs) {
        output_node_values.push_back((float *) output_node.data);
    }
    return 0;
}

