#include <inttypes.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <QnnDevice.h>
#include <QnnInterface.h>
#include <cstring>
#include <fstream>
#include <iostream>

#include "Utils/DataUtil.hpp"
#include "Log/Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#include "PAL/StringOp.hpp"
#include "QnnSampleApp.hpp"
#include "Utils/QnnSampleAppUtils.hpp"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "Utils/IOTensor.hpp"
#include "QnnTypeMacros.hpp"
#include "qcom/time_all.h"

using namespace qnn;
using namespace qnn::tools;

const std::string sample_app::QnnSampleApp::s_defaultOutputPath = "./output/";

sample_app::QnnSampleApp::QnnSampleApp(QnnFunctionPointers qnnFunctionPointers,
                                       std::string inputListPaths,
                                       std::string opPackagePaths,
                                       void *backendLibraryHandle,
                                       std::string outputPath,
                                       bool debug,
                                       iotensor::OutputDataType outputDataType,
                                       iotensor::InputDataType inputDataType,
                                       sample_app::ProfilingLevel profilingLevel,
                                       bool dumpOutputs,
                                       std::string cachedBinaryPath,
                                       std::string saveBinaryName)
        : m_qnnFunctionPointers(qnnFunctionPointers),
          m_outputPath(outputPath),
          m_saveBinaryName(saveBinaryName),
          m_cachedBinaryPath(cachedBinaryPath),
          m_debug(debug),
          m_outputDataType(outputDataType),
          m_inputDataType(inputDataType),
          m_profilingLevel(profilingLevel),
          m_dumpOutputs(dumpOutputs),
          m_backendLibraryHandle(backendLibraryHandle),
          m_isBackendInitialized(false),
          m_isContextCreated(false) {
    split(m_inputListPaths, inputListPaths, ',');
    split(m_opPackagePaths, opPackagePaths, ',');
    if (m_outputPath.empty()) {
        m_outputPath = s_defaultOutputPath;
    }
    return;
}

sample_app::QnnSampleApp::~QnnSampleApp() {
    if (nullptr != m_profileBackendHandle) {
        LOGD("Freeing backend profile object.");
        if (QNN_PROFILE_NO_ERROR !=
            m_qnnFunctionPointers.qnnInterface.profileFree(m_profileBackendHandle)) {
            LOGE("Could not free backend profile handle.");
        }
    }
    if (m_isContextCreated) {
        LOGD("Freeing context");
        if (QNN_CONTEXT_NO_ERROR !=
            m_qnnFunctionPointers.qnnInterface.contextFree(m_context, nullptr)) {
            LOGE("Could not free context");
        }
    }
    m_isContextCreated = false;
    if (m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) {
        LOGD("Freeing backend");
        if (QNN_BACKEND_NO_ERROR !=
            m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
            LOGE("Could not free backend");
        }
    }
    m_isBackendInitialized = false;
    if (nullptr != m_qnnFunctionPointers.qnnInterface.logFree && nullptr != m_logHandle) {
        if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.logFree(m_logHandle)) {
            LOGI("Unable to terminate logging in the backend.");
        }
    }
    LOGI("QNN has finished");
    return;
}

std::string sample_app::QnnSampleApp::getBackendBuildId() {
    char *backendBuildId{nullptr};
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char **) &backendBuildId)) {
        LOGE("Unable to get build Id from the backend.");
    }
    return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
}

sample_app::StatusCode sample_app::QnnSampleApp::initialize() {
    if (m_dumpOutputs && !::pal::FileOp::checkFileExists(m_outputPath) &&
        !pal::Directory::makePath(m_outputPath)) {
        LOGE("Could not create output directory:%s ", m_outputPath.c_str());
    }

    if (log::isLogInitialized()) {
        auto logCallback = log::getLogCallback();
        auto logLevel = log::getLogLevel();
        LOGI("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
             logCallback,
             logLevel);
        if (QNN_SUCCESS !=
            m_qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
            LOGI("Unable to initialize logging in the backend.");
        }
    } else {
        LOGI("Logging not available in the backend.");
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::initializeProfiling() {
    if (ProfilingLevel::OFF != m_profilingLevel) {
        LOGI("Profiling turned on; level = %d", m_profilingLevel);
        if (ProfilingLevel::BASIC == m_profilingLevel) {
            LOGI("Basic profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR !=
                m_qnnFunctionPointers.qnnInterface.profileCreate(
                        m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
                LOGI("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
            LOGI("Detailed profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR !=
                m_qnnFunctionPointers.qnnInterface.profileCreate(
                        m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
                LOGE("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        }
    }
    return StatusCode::SUCCESS;
}

int32_t sample_app::QnnSampleApp::reportError(const std::string &err) {
    LOGE("%s", err.c_str());
    return EXIT_FAILURE;
}

sample_app::StatusCode sample_app::QnnSampleApp::initializeBackend() {
    auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
            m_logHandle, (const QnnBackend_Config_t **) m_backendConfig, &m_backendHandle);
    if (QNN_BACKEND_NO_ERROR != qnnStatus) {
        LOGE("Could not initialize backend due to error = %lu", qnnStatus);
        return StatusCode::FAILURE;
    }
    LOGI("Initialize Backend Returned Status = %lu", qnnStatus);
    m_isBackendInitialized = true;
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::terminateBackend() {
    if ((m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) &&
        QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
        LOGE("Could not terminate backend");
        return StatusCode::FAILURE;
    }
    m_isBackendInitialized = false;
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::registerOpPackages() {
    const size_t pathIdx = 0;
    const size_t interfaceProviderIdx = 1;
    for (auto const &opPackagePath: m_opPackagePaths) {
        std::vector<std::string> opPackage;
        split(opPackage, opPackagePath, ':');
        LOGD("opPackagePath: %s", opPackagePath.c_str());
        if (opPackage.size() != 2) {
            LOGE("Malformed opPackageString provided: %s", opPackagePath.c_str());
            return StatusCode::FAILURE;
        }
        if (nullptr == m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
            LOGE("backendRegisterOpPackageFnHandle is nullptr.");
            return StatusCode::FAILURE;
        }
        if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
                m_backendHandle,
                (char *) opPackage[pathIdx].c_str(),
                (char *) opPackage[interfaceProviderIdx].c_str(),
                nullptr)) {
            LOGE("Could not register Op Package: %s and interface provider: %s",
                 opPackage[pathIdx].c_str(),
                 opPackage[interfaceProviderIdx].c_str());
            return StatusCode::FAILURE;
        }
        LOGI("Registered Op Package: %s and interface provider: %s",
             opPackage[pathIdx].c_str(),
             opPackage[interfaceProviderIdx].c_str());
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::createContext() {
    if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextCreate(
            m_backendHandle,
            m_deviceHandle,
            (const QnnContext_Config_t **) &m_contextConfig,
            &m_context)) {
        LOGE("Could not create context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = true;
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::freeContext() {
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle)) {
        LOGE("Could not free context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = false;
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::composeGraphs() {
    auto returnStatus = StatusCode::SUCCESS;
    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR !=
        m_qnnFunctionPointers.composeGraphsFnHandle(
                m_backendHandle,
                m_qnnFunctionPointers.qnnInterface,
                m_context,
                (const qnn_wrapper_api::GraphConfigInfo_t **) m_graphConfigsInfo,
                m_graphConfigsInfoCount,
                &m_graphsInfo,
                &m_graphsCount,
                m_debug,
                log::getLogCallback(),
                log::getLogLevel())) {
        LOGE("Failed in composeGraphs()");
        returnStatus = StatusCode::FAILURE;
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::finalizeGraphs() {
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        if (QNN_GRAPH_NO_ERROR !=
            m_qnnFunctionPointers.qnnInterface.graphFinalize(
                    (*m_graphsInfo)[graphIdx].graph, m_profileBackendHandle, nullptr)) {
            return StatusCode::FAILURE;
        }
    }
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }
    auto returnStatus = StatusCode::SUCCESS;
    if (!m_saveBinaryName.empty()) {
        LOGI("Before saveBinary(): saving context and metadata.");
        returnStatus = saveBinary();
    } else {
        LOGD("m_saveBinaryName is empty()");
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::createFromBinary() {
    if (m_cachedBinaryPath.empty()) {
        LOGE("No name provided to read binary file from.");
        return StatusCode::FAILURE;
    }
    if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
        LOGE("QNN System function pointers are not populated.");
        return StatusCode::FAILURE;
    }
    uint64_t bufferSize{0};
    std::shared_ptr<uint8_t> buffer{nullptr};
    tools::datautil::StatusCode status{tools::datautil::StatusCode::SUCCESS};
    std::tie(status, bufferSize) = tools::datautil::getFileSize(m_cachedBinaryPath);
    LOGI("BinaryPath:%s",m_cachedBinaryPath.c_str());
    if (0 == bufferSize) {
        LOGE("Received path to an empty file. Nothing to deserialize.");
        return StatusCode::FAILURE;
    }
    buffer = std::shared_ptr<uint8_t>(new uint8_t[bufferSize], std::default_delete<uint8_t[]>());
    if (!buffer) {
        LOGE("Failed to allocate memory.");
        return StatusCode::FAILURE;
    }

    status = tools::datautil::readBinaryFromFile(
            m_cachedBinaryPath, reinterpret_cast<uint8_t *>(buffer.get()), bufferSize);
    if (status != tools::datautil::StatusCode::SUCCESS) {
        LOGE("Failed to read binary data.");
        return StatusCode::FAILURE;
    }

    auto returnStatus = StatusCode::SUCCESS;
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
        LOGE("Could not create system handle.");
        returnStatus = StatusCode::FAILURE;
    }
    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};
    if (StatusCode::SUCCESS == returnStatus &&
        QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                sysCtxHandle,
                static_cast<void *>(buffer.get()),
                bufferSize,
                &binaryInfo,
                &binaryInfoSize)) {
        LOGE("Failed to get context binary info");
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
    }

    if (StatusCode::SUCCESS == returnStatus &&
        !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
        LOGE("Failed to copy metadata.");
        returnStatus = StatusCode::FAILURE;
    }
    m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
        LOGE("contextCreateFromBinaryFnHandle is nullptr.");
        returnStatus = StatusCode::FAILURE;
    }
    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
                m_backendHandle,
                m_deviceHandle,
                (const QnnContext_Config_t **) &m_contextConfig,
                static_cast<void *>(buffer.get()),
                bufferSize,
                &m_context,
                m_profileBackendHandle)) {
        LOGE("Could not create context from binary.");
        returnStatus = StatusCode::FAILURE;
    }
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }
    m_isContextCreated = true;
    if (StatusCode::SUCCESS == returnStatus) {
        for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
            if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
                LOGE("graphRetrieveFnHandle is nullptr.");
                returnStatus = StatusCode::FAILURE;
                break;
            }
            if (QNN_SUCCESS !=
                m_qnnFunctionPointers.qnnInterface.graphRetrieve(
                        m_context, (*m_graphsInfo)[graphIdx].graphName,
                        &((*m_graphsInfo)[graphIdx].graph))) {
                LOGE("Unable to retrieve graph handle for graph Idx: %zu", graphIdx);
                returnStatus = StatusCode::FAILURE;
            }
            LOGI("graphName:%s", (*m_graphsInfo)[graphIdx].graphName);
        }
    }
    if (StatusCode::SUCCESS != returnStatus) {
        LOGD("Cleaning up graph Info structures.");
        qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::saveBinary() {
    if (m_saveBinaryName.empty()) {
        LOGE("No name provided to save binary file.");
        return StatusCode::FAILURE;
    }
    if (nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinarySize ||
        nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinary) {
        LOGE("contextGetBinarySizeFnHandle or contextGetBinaryFnHandle is nullptr.");
        return StatusCode::FAILURE;
    }
    uint64_t requiredBufferSize{0};
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextGetBinarySize(m_context, &requiredBufferSize)) {
        LOGE("Could not get the required binary size.");
        return StatusCode::FAILURE;
    }
    std::unique_ptr<uint8_t[]> saveBuffer(new uint8_t[requiredBufferSize]);
    if (nullptr == saveBuffer) {
        LOGE("Could not allocate buffer to save binary.");
        return StatusCode::FAILURE;
    }
    uint64_t writtenBufferSize{0};
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextGetBinary(m_context,
                                                            reinterpret_cast<void *>(saveBuffer.get()),
                                                            requiredBufferSize,
                                                            &writtenBufferSize)) {
        LOGE("Could not get binary.");
        return StatusCode::FAILURE;
    }
    if (requiredBufferSize < writtenBufferSize) {
        LOGE(
                "Illegal written buffer size [%lu] bytes. Cannot exceed allocated memory of [%lu] bytes",
                writtenBufferSize,
                requiredBufferSize);
        return StatusCode::FAILURE;
    }
    auto dataUtilStatus = tools::datautil::writeBinaryToFile(
            m_outputPath, m_saveBinaryName + ".bin", (uint8_t *) saveBuffer.get(),
            writtenBufferSize);
    if (tools::datautil::StatusCode::SUCCESS != dataUtilStatus) {
        LOGE("Error while writing binary to file.");
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractBackendProfilingInfo(
        Qnn_ProfileHandle_t profileHandle) {
    if (nullptr == m_profileBackendHandle) {
        LOGE("Backend Profile handle is nullptr; may not be initialized.");
        return StatusCode::FAILURE;
    }
    const QnnProfile_EventId_t *profileEvents{nullptr};
    uint32_t numEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEvents(
            profileHandle, &profileEvents, &numEvents)) {
        LOGE("Failure in profile get events.");
        return StatusCode::FAILURE;
    }
    LOGD("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
    for (size_t event = 0; event < numEvents; event++) {
        extractProfilingEvent(*(profileEvents + event));
        extractProfilingSubEvents(*(profileEvents + event));
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractProfilingSubEvents(
        QnnProfile_EventId_t profileEventId) {
    const QnnProfile_EventId_t *profileSubEvents{nullptr};
    uint32_t numSubEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(
            profileEventId, &profileSubEvents, &numSubEvents)) {
        LOGE("Failure in profile get sub events.");
        return StatusCode::FAILURE;
    }
    LOGD("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
    for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
        extractProfilingEvent(*(profileSubEvents + subEvent));
        extractProfilingSubEvents(*(profileSubEvents + subEvent));
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::extractProfilingEvent(
        QnnProfile_EventId_t profileEventId) {
    QnnProfile_EventData_t eventData;
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId, &eventData)) {
        LOGE("Failure in profile get event type.");
        return StatusCode::FAILURE;
    }
    LOGD("Printing Event Info - Event Type: [%d], Event Value: [%" PRIu64
                 "], Event Identifier: [%s], Event Unit: [%d]",
         eventData.type,
         eventData.value,
         eventData.identifier,
         eventData.unit);
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::verifyFailReturnStatus(Qnn_ErrorHandle_t errCode) {
    auto returnStatus = sample_app::StatusCode::FAILURE;
    switch (errCode) {
        case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
            returnStatus = sample_app::StatusCode::FAILURE_SYSTEM_COMMUNICATION_ERROR;
            break;
        case QNN_COMMON_ERROR_SYSTEM:
            returnStatus = sample_app::StatusCode::FAILURE_SYSTEM_ERROR;
            break;
        case QNN_COMMON_ERROR_NOT_SUPPORTED:
            returnStatus = sample_app::StatusCode::QNN_FEATURE_UNSUPPORTED;
            break;
        default:
            break;
    }
    return returnStatus;
}

sample_app::StatusCode sample_app::QnnSampleApp::isDevicePropertySupported() {
    if (nullptr != m_qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnStatus =
                m_qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus ||
            QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
            LOGE("Device property not supported or not known to backend");
            return StatusCode::FAILURE;
        }
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::createDevice() {
    if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceCreate) {
        auto qnnStatus =
                m_qnnFunctionPointers.qnnInterface.deviceCreate(m_logHandle, nullptr,
                                                                &m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus) {
            LOGE("Failed to create device");
            return verifyFailReturnStatus(qnnStatus);
        }
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::freeDevice() {
    if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceFree) {
        auto qnnStatus = m_qnnFunctionPointers.qnnInterface.deviceFree(m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus) {
            LOGE("Failed to free device");
            return verifyFailReturnStatus(qnnStatus);
        }
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::initializePerformance() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr =
            m_qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        QNN_ERROR("device error");
        return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra =
            static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t *perfInfra = &htpInfra->perfInfra;
    uint32_t powerConfigId = 1;
    uint32_t deviceId = 0;
    uint32_t coreId = 0;
    perfInfra->createPowerConfigId(deviceId, coreId, &powerConfigId);
    m_perfInfra = perfInfra;
    m_powerConfigId = powerConfigId;
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::setHighPerformanceMode() {
    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable = 0;
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId = m_powerConfigId;
    powerConfig.dcvsV3Config.powerMode =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency =
            1; // True to consider Latency parameter otherwise False
    powerConfig.dcvsV3Config.setBusParams = 1; // True to consider Bus parameter otherwise False
    powerConfig.dcvsV3Config.setCoreParams = 1; // True to consider Core parameter otherwise False
    powerConfig.dcvsV3Config.sleepDisable = 0; // True to consider sleep/LPM modes, False to enable
    powerConfig.dcvsV3Config.setSleepDisable =
            0; // True to consider sleep disable/enable parameter otherwise False
// Set Sleep latency parameter
    uint32_t latencyValue = 40; // V73
    powerConfig.dcvsV3Config.sleepLatency = latencyValue; // range 40-2000 micro sec
// set Bus Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
    powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
// set Core Clock Parameters (refer QnnHtpPerfInfrastructure_VoltageCorner_t enum)
    powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
// Set power config with different performance parameters
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] =
            {&powerConfig, NULL};
    if (m_perfInfra) {
        m_perfInfra->setPowerConfig(m_powerConfigId, powerConfigs);
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::setRpcPolling() {
    if (m_rpcPollingTime > 0) {
        QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;
        memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
        rpcPollingTime.option =
                QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
        rpcPollingTime.rpcPollingTimeConfig = m_rpcPollingTime;
        const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] =
                {&rpcPollingTime, NULL};
        if (m_perfInfra) {
            m_perfInfra->setPowerConfig(m_powerConfigId, powerConfigs);
        }
    }
    return StatusCode::SUCCESS;
}

sample_app::StatusCode sample_app::QnnSampleApp::execute_common_Graphs(
        const std::vector<uint8_t *> &input_node_values,
        std::vector<float *> &output_node_values,
        std::vector<size_t> output_node_sizes) {
    Timer timer("qnn inference");
    auto returnStatus = StatusCode::SUCCESS;
    assert(m_graphsCount == 1);
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        LOGD("Starting execution for graphIdx: %zu", graphIdx);

        Qnn_Tensor_t *inputs = nullptr;
        Qnn_Tensor_t *outputs = nullptr;
        if (iotensor::StatusCode::SUCCESS !=
            m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx])) {
            LOGE("Error in setting up Input and output Tensors for graphIdx: %zu", graphIdx);
            returnStatus = StatusCode::FAILURE;
            return returnStatus;
        }
        auto graphInfo = (*m_graphsInfo)[graphIdx];

        std::vector<uint8_t *> input_vec;
        for (auto input_node_value : input_node_values) {
            input_vec.push_back(input_node_value);
        }

        if (iotensor::StatusCode::SUCCESS !=
            m_ioTensor.populateInputTensors(graphIdx, input_vec, inputs, graphInfo,
                                            m_inputDataType)) {
            LOGE("Error in populateInputTensors");
            returnStatus = StatusCode::FAILURE;
            return returnStatus;
        }
        timer.add_record_point("prepare");
        if (StatusCode::SUCCESS == returnStatus) {
            LOGD("Successfully populated input tensors for graphIdx: %zu", graphIdx);
            Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
            executeStatus =
                    m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                                    inputs,
                                                                    graphInfo.numInputTensors,
                                                                    outputs,
                                                                    graphInfo.numOutputTensors,
                                                                    m_profileBackendHandle,
                                                                    nullptr);


            if (QNN_GRAPH_NO_ERROR != executeStatus) {
                LOGE("Error in graph execute!");
                returnStatus = StatusCode::FAILURE;
                return returnStatus;
            }
            timer.add_record_point("inference model");
            for (int i = 0; i < graphInfo.numOutputTensors; ++i) {
                if (nullptr == output_node_values.at(i)) {
                    continue;
                }
                float *output_tmp = nullptr;
                if (iotensor::StatusCode::SUCCESS !=
                    m_ioTensor.converQnntensortoFloatBuffer(&(outputs[i]), &output_tmp)) {
                    LOGE("Error in convert output at idx %d!", i);
                    returnStatus = StatusCode::FAILURE;
                    return returnStatus;
                }
                memcpy(output_node_values.at(i), output_tmp, output_node_sizes[i] * sizeof(float));
            }
            timer.add_record_point("output");
        }
        m_ioTensor.tearDownInputAndOutputTensors(
                inputs, outputs, graphInfo.numInputTensors, graphInfo.numOutputTensors);
        inputs = nullptr;
        outputs = nullptr;
    }

    return returnStatus;
}