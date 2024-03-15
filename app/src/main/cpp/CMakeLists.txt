# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("stablediffusion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti -fopenmp -static-openmp")

set(BINARY_MODELS src/onnx/*.cpp)

set(BOOST_ROOT ${CMAKE_SOURCE_DIR}/boost/arm64-v8a)
set(Boost_INCLUDE_DIR ${BOOST_ROOT}/include/boost-1_82)
set(Boost_LIBRARY_DIR ${BOOST_ROOT}/lib)
#用通配符同时添加多个lib库文件依赖
file(GLOB Boost_LIBRARIES "${Boost_LIBRARY_DIR}/libboost*.a")

set(onnx_runtime_DIR ${CMAKE_SOURCE_DIR}/onnx_runtime)
set(MI_SECURITY_DIR ${CMAKE_SOURCE_DIR}/mi_security)
include_directories(
        include
        ${QCOM_HEADER}
        ${onnx_runtime_DIR}/include/onnxruntime
        ${Boost_INCLUDE_DIR}
        ${MI_SECURITY_DIR}/include
)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${onnx_runtime_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so)


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
        ${Boost_LIBRARY_DIR}
        # Links the target library to the log library
        # included in the NDK.
        ${OpenCV_LIBS} onnxruntime
        android
        z
        ${log-lib}
        ${android-lib}
        jnigraphics)