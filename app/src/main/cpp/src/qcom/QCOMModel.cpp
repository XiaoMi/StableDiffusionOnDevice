#include "qcom/QCOMModel.h"

std::vector<size_t> cal_dims_size(const std::vector<std::vector<int64_t>> &dims) {
    std::vector<size_t> res;
    for (auto now_dim: dims) {
        size_t now_size = std::accumulate(now_dim.begin(), now_dim.end(), 1,
                                          std::multiplies<int64_t>());
        res.push_back(now_size);
    }
    return res;
}

QCOMModel::QCOMModel() {

}

QCOMModel::~QCOMModel() {
    unload();
}

int QCOMModel::load(const std::string &path, int device_id) {
    unload();

    model_path_ = path + model_sub_path_;
    LOGI("%s load model %s", model_name_.c_str(), model_path_.c_str());
    if (!file_exists(model_path_)) {
        LOGE("%s not exist!", model_path_.c_str());
        return -1;
    }

    auto beginTime = std::chrono::high_resolution_clock::now();
    if (!qnn::log::initializeLogging()) {
        LOGE("ERROR: Unable to initialize logging!");
        return -1;
    }
    std::string modelPath;
    std::string backEndPath;
    std::string inputListPaths;
    bool debug = false;
    std::string outputPath;
    std::string opPackagePaths;

    iotensor::OutputDataType parsedOutputDataType = iotensor::OutputDataType::FLOAT_ONLY;
    iotensor::InputDataType parsedInputDataType = iotensor::InputDataType::FLOAT;
    sample_app::ProfilingLevel parsedProfilingLevel = ProfilingLevel::OFF;
    bool dumpOutputs = false;

    std::string saveBinaryName;

    QnnLog_Level_t logLevel{QNN_LOG_LEVEL_INFO};
    if (logLevel != QNN_LOG_LEVEL_MAX) {
        if (!log::setLogLevel(logLevel)) {
            LOGE("Unable to set log level.%d", ::qnn::log::Logger::isValid());
        }
    }

    std::string systemLibraryPath;

    modelPath = "";
    backEndPath = path + std::string("/stable_diffusion/qnn_lib_") + QCOM_VERSION +
                  "/libQnnHtp.so";
    inputListPaths = "";
    outputPath = "";
    opPackagePaths = "";
    systemLibraryPath = path + std::string("/stable_diffusion/qnn_lib_") + QCOM_VERSION +
                        "/libQnnSystem.so";

    LOGI("Model: %s", modelPath.c_str());
    LOGI("Backend: %s", backEndPath.c_str());

    QnnFunctionPointers qnnFunctionPointers;
    auto statusCode = dynamicloadutil::getQnnFunctionPointers(backEndPath,
                                                              modelPath,
                                                              &qnnFunctionPointers,
                                                              &sg_backendHandle,
                                                              !loadFromCachedBinary,
                                                              &sg_modelHandle);
    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
        if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
            LOGE("Error initializing QNN Function Pointers: could not load backend:%s ",
                 backEndPath.c_str());
        } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
            LOGE("Error initializing QNN Function Pointers: could not load model:%s ",
                 modelPath.c_str());
        } else {
            LOGE("Error initializing QNN Function Pointers");
        }
        return -1;
    }

    if (loadFromCachedBinary) {
        statusCode =
                dynamicloadutil::getQnnSystemFunctionPointers(systemLibraryPath,
                                                              &qnnFunctionPointers);
        if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
            LOGI("ZZY THIS STATUSCODE IS %d", statusCode);
            LOGE("Error initializing QNN System Function Pointers");
            return -1;
        }
    }

    qnn_net = std::make_unique<sample_app::QnnSampleApp>(qnnFunctionPointers,
                                                         inputListPaths,
                                                         opPackagePaths,
                                                         sg_backendHandle,
                                                         outputPath,
                                                         debug,
                                                         parsedOutputDataType,
                                                         parsedInputDataType,
                                                         parsedProfilingLevel,
                                                         dumpOutputs,
                                                         model_path_,
                                                         saveBinaryName);

    LOGI("qnn-sample-app build version: %s", qnn::tools::getBuildId().c_str());
    LOGI("Backend        build version: %s", qnn_net->getBackendBuildId().c_str());
    if (sample_app::StatusCode::SUCCESS != qnn_net->initialize()) {
        LOGE("Initialization failure");
        return -1;
    }
    if (sample_app::StatusCode::SUCCESS != qnn_net->initializeBackend()) {
        LOGE("Backend Initialization failure");
        return -1;
    }
    auto devicePropertySupportStatus = qnn_net->isDevicePropertySupported();
    if (sample_app::StatusCode::FAILURE != devicePropertySupportStatus) {
        auto createDeviceStatus = qnn_net->createDevice();
        if (sample_app::StatusCode::SUCCESS != createDeviceStatus) {
            LOGE("Initialization failure");
            return -1;
        }
    }
    if (sample_app::StatusCode::SUCCESS != qnn_net->initializeProfiling()) {
        LOGE("Profiling Initialization failure");
        return -1;
    }

    if (!loadFromCachedBinary) {
        if (sample_app::StatusCode::SUCCESS != qnn_net->createContext()) {
            LOGE("Context Creation failure");
            return -1;
        }
        if (sample_app::StatusCode::SUCCESS != qnn_net->composeGraphs()) {
            LOGE("Graph Prepare failure");
            return -1;
        }
        if (sample_app::StatusCode::SUCCESS != qnn_net->finalizeGraphs()) {
            LOGE("Graph Finalize failure");
            return -1;
        }
    } else {
        if (sample_app::StatusCode::SUCCESS != qnn_net->createFromBinary()) {
            LOGE("Create From Binary failure");
            return -1;
        }
    }

    if (sample_app::StatusCode::SUCCESS != qnn_net->initializePerformance()) {
        LOGE("initializePerformance() failure");
        return -1;
    }
    if (sample_app::StatusCode::SUCCESS != qnn_net->setRpcPolling()) {
        LOGE("setRpcPolling() failure");
        return -1;
    }
    if (sample_app::StatusCode::SUCCESS != qnn_net->setHighPerformanceMode()) {
        LOGE("setHighPerformanceMode failure");
        return -1;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    LOGI("%s init cost: %f ms", model_name_.c_str(), (float) (elapsedTime.count() / 1000));
    return 0;
}

int QCOMModel::before_run() {
    return 0;
}

int QCOMModel::after_run() {
    return 0;
}

int QCOMModel::inference(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) {
    pre_process(inputs, outputs);
    assert(input_node_values.size() == input_node_dims.size());
    assert(output_node_values.size() == output_node_dims.size());
    if (sample_app::StatusCode::SUCCESS !=qnn_net->execute_common_Graphs(input_node_values, output_node_values, output_node_sizes))
        return -1;
    post_process(outputs);
    input_node_values.clear();
    output_node_values.clear();
    return 0;
}


int QCOMModel::pre_process(const std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs) {
    for (auto &input_node: inputs) {
        input_node_values.push_back((uint8_t *) input_node.data);
    }
    for (auto &output_node: outputs) {
        output_node_values.push_back((float *) output_node.data);
    }

    return 0;
}

int QCOMModel::unload() {
    if (qnn_net) {
        if(qnn_wrapper_api::freeGraphsInfo(&qnn_net->m_graphsInfo, qnn_net->m_graphsCount))
            return -1;
        qnn_net->m_graphsInfo = nullptr;
        if (sample_app::StatusCode::SUCCESS != qnn_net->freeContext()) {
            LOGE("Context Free failure");
            return -1;
        }

        auto freeDeviceStatus = qnn_net->freeDevice();
        if (sample_app::StatusCode::SUCCESS != freeDeviceStatus) {
            LOGE("Device Free failure");
            return -1;
        }
        qnn_net.reset();
    }
    return 0;
}
