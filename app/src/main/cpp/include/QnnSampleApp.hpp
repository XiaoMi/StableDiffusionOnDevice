#pragma once

#include <memory>
#include <queue>
#include "QnnTypes.h"
#include "IOTensor.hpp"
#include "SampleApp.hpp"
#include "utils/utils.h"
#include <HTP/QnnHtpDevice.h>

namespace qnn {
    namespace tools {
        namespace sample_app {

            enum class StatusCode {
                SUCCESS,
                FAILURE,
                FAILURE_INPUT_LIST_EXHAUSTED,
                FAILURE_SYSTEM_ERROR,
                FAILURE_SYSTEM_COMMUNICATION_ERROR,
                QNN_FEATURE_UNSUPPORTED
            };

            class QnnSampleApp {
            public:
                QnnSampleApp(QnnFunctionPointers qnnFunctionPointers,
                             std::string inputListPaths,
                             std::string opPackagePaths,
                             void *backendHandle,
                             std::string outputPath = s_defaultOutputPath,
                             bool debug = false,
                             iotensor::OutputDataType outputDataType = iotensor::OutputDataType::FLOAT_ONLY,
                             iotensor::InputDataType inputDataType = iotensor::InputDataType::FLOAT,
                             ProfilingLevel profilingLevel = ProfilingLevel::OFF,
                             bool dumpOutputs = false,
                             std::string cachedBinaryPath = "",
                             std::string saveBinaryName = "");

                // @brief Print a message to STDERR then return a nonzero
                //  exit status.
                int32_t reportError(const std::string &err);

                StatusCode initialize();

                StatusCode initializeBackend();

                StatusCode createContext();

                StatusCode composeGraphs();

                StatusCode finalizeGraphs();

                StatusCode executeGraphs();

                StatusCode registerOpPackages();

                StatusCode createFromBinary();

                StatusCode saveBinary();

                StatusCode freeContext();

                StatusCode terminateBackend();

                StatusCode freeGraphs();

                Qnn_ContextHandle_t getContext();

                StatusCode initializeProfiling();

                std::string getBackendBuildId();

                StatusCode isDevicePropertySupported();

                StatusCode createDevice();

                StatusCode freeDevice();

                StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

                StatusCode execute_common_Graphs(const std::vector<uint8_t *>& input_node_values,
                                                 std::vector<float *> &output_node_values,
                                                 std::vector<size_t> output_node_sizes);

                StatusCode initializePerformance();

                StatusCode setHighPerformanceMode();

                StatusCode setRpcPolling();

                virtual ~QnnSampleApp();

                uint32_t m_graphsCount;
                qnn_wrapper_api::GraphInfo_t **m_graphsInfo;

            private:
                StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

                StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

                StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId);

                static const std::string s_defaultOutputPath;

                QnnFunctionPointers m_qnnFunctionPointers;
                std::vector<std::string> m_inputListPaths;
                std::vector<std::vector<std::queue<std::string>>> m_inputFileLists;
                std::vector<std::string> m_opPackagePaths;
                std::string m_outputPath;
                std::string m_saveBinaryName;
                std::string m_cachedBinaryPath;
                QnnBackend_Config_t **m_backendConfig = nullptr;
                Qnn_ContextHandle_t m_context = nullptr;
                QnnContext_Config_t **m_contextConfig = nullptr;
                bool m_debug;
                iotensor::OutputDataType m_outputDataType;
                iotensor::InputDataType m_inputDataType;
                ProfilingLevel m_profilingLevel;
                bool m_dumpOutputs;
                iotensor::IOTensor m_ioTensor;

                void *m_backendLibraryHandle;

                bool m_isBackendInitialized;
                bool m_isContextCreated;
                Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
                qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
                uint32_t m_graphConfigsInfoCount;
                Qnn_LogHandle_t m_logHandle = nullptr;
                Qnn_BackendHandle_t m_backendHandle = nullptr;
                Qnn_DeviceHandle_t m_deviceHandle = nullptr;
                QnnHtpDevice_PerfInfrastructure_t *m_perfInfra = nullptr;
                uint32_t m_powerConfigId = 1;
                uint32_t m_rpcPollingTime = 9999; // 0-10000 us for high performing
            };
        }  // namespace sample_app
    }  // namespace tools
}  // namespace qnn
