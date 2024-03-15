# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("stablediffusion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -static-openmp")

# 1 QCOM 8550, 2 QCOM 8650
set(DL_MODE 2)


if(DL_MODE EQUAL 1)
    add_definitions(-DUSE_DL_QCOM)
    add_definitions(-DQCOM_VERSION="8550")
    set(BINARY_MODELS src/qcom/QCOMModel.cpp src/qcom/time_all.cpp
            include/qnnUtils_8550/PAL/src/common/GetOpt.cpp include/qnnUtils_8550/PAL/src/common/StringOp.cpp include/qnnUtils_8550/PAL/src/linux/Directory.cpp
            include/qnnUtils_8550/PAL/src/linux/DynamicLoading.cpp include/qnnUtils_8550/PAL/src/linux/FileOp.cpp include/qnnUtils_8550/PAL/src/linux/Path.cpp include/qnnUtils_8550/Log/Logger.cpp
            include/qnnUtils_8550/Log/LogUtils.cpp include/qnnUtils_8550/Utils/DataUtil.cpp include/qnnUtils_8550/Utils/DynamicLoadUtil.cpp include/qnnUtils_8550/Utils/IOTensor.cpp include/qnnUtils_8550/Utils/QnnSampleAppUtils.cpp
            include/qnnUtils_8550/WrapperUtils/QnnWrapperUtils.cpp)
    set(QCOM_HEADER include/qnnUtils_8550 include/qnnUtils_8550/Utils include/qnnUtils_8550/Log include/qnnUtils_8550/PAL/include include/qnnUtils_8550/WrapperUtils qnn_8550/include/QNN)

elseif(DL_MODE EQUAL 2)
    add_definitions(-DUSE_DL_QCOM)
    add_definitions(-DQCOM_VERSION="8650")
    set(BINARY_MODELS src/qcom/QCOMModel.cpp src/qcom/time_all.cpp
            include/qnnUtils_8650/PAL/src/common/GetOpt.cpp
            include/qnnUtils_8650/PAL/src/common/StringOp.cpp
            include/qnnUtils_8650/PAL/src/linux/Directory.cpp
            include/qnnUtils_8650/PAL/src/linux/DynamicLoading.cpp
            include/qnnUtils_8650/PAL/src/linux/FileOp.cpp
            include/qnnUtils_8650/PAL/src/linux/Path.cpp
            include/qnnUtils_8650/Log/Logger.cpp
            include/qnnUtils_8650/Log/LogUtils.cpp
            include/qnnUtils_8650/Utils/DataUtil.cpp
            include/qnnUtils_8650/Utils/DynamicLoadUtil.cpp
            include/qnnUtils_8650/Utils/IOTensor.cpp
            include/qnnUtils_8650/Utils/QnnSampleAppUtils.cpp
            include/qnnUtils_8650/WrapperUtils/QnnWrapperUtils.cpp)
    set(QCOM_HEADER
            include/qnnUtils_8650
            include/qnnUtils_8650/Utils
            include/qnnUtils_8650/Log
            include/qnnUtils_8650/PAL/include
            include/qnnUtils_8650/WrapperUtils
            qnn_8650/include/QNN
            )

else()
endif ()

set(BOOST_ROOT ${CMAKE_SOURCE_DIR}/boost/arm64-v8a)
set(Boost_INCLUDE_DIR ${BOOST_ROOT}/include/boost-1_82)
set(Boost_LIBRARY_DIR ${BOOST_ROOT}/lib)
#用通配符同时添加多个lib库文件依赖
file(GLOB Boost_LIBRARIES "${Boost_LIBRARY_DIR}/libboost*.a")

include_directories(
        include
        ${Boost_INCLUDE_DIR}
        ${QCOM_HEADER}
)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

file(GLOB SRC src/*.cpp src/scheduler/*.cpp src/utils/*.cpp src/models/*.cpp  ${BINARY_MODELS})
set(COMPILE_CODE ${SRC})
add_definitions(-DUSE_ASSET_MANAGER)
add_library( # Sets the name of the library.
        stablediffusion
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${COMPILE_CODE})

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        stablediffusion
        # Links the target library to the log library
        # included in the NDK.
        ${OpenCV_LIBS}
        android
        z
        ${log-lib}
        ${android-lib}
        jnigraphics)