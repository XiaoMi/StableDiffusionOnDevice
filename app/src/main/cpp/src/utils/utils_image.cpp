#include "utils/utils_image.h"


void matToBitmap(JNIEnv *env, cv::Mat &src, jobject bitmap) {
    AndroidBitmapInfo info;
    void *pixels = nullptr;

    LOGI("nMatToBitmap");
    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
              info.width == (uint32_t) src.cols);
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
        if (src.type() == CV_8UC1) {
            LOGI("nMatToBitmap: CV_8UC1 -> RGBA_8888");
            cvtColor(src, tmp, cv::COLOR_GRAY2RGBA);
        } else if (src.type() == CV_8UC3) {
            LOGI("nMatToBitmap: CV_8UC3 -> RGBA_8888");
            cvtColor(src, tmp, cv::COLOR_RGB2RGBA);
        } else if (src.type() == CV_8UC4) {
            LOGI("nMatToBitmap: CV_8UC4 -> RGBA_8888");
            src.copyTo(tmp);
        }
    } else {
        cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
        if (src.type() == CV_8UC1) {
            LOGI("nMatToBitmap: CV_8UC1 -> RGB_565");
            cvtColor(src, tmp, cv::COLOR_GRAY2BGR565);
        } else if (src.type() == CV_8UC3) {
            LOGI("nMatToBitmap: CV_8UC3 -> RGB_565");
            cvtColor(src, tmp, cv::COLOR_RGB2BGR565);
        } else if (src.type() == CV_8UC4) {
            LOGI("nMatToBitmap: CV_8UC4 -> RGB_565");
            cvtColor(src, tmp, cv::COLOR_RGBA2BGR565);
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}


cv::Mat FourierFeatures(const float t) {
    cv::Mat cv_x(cv::Size(640, 1), CV_32FC1);
    cv::RNG rng(0);
    rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
    std::vector<float> out(1280);
    for (int i = 0; i < 640; i++) {
        double f = *((float *) cv_x.data + i) * 2 * 3.141592653589793 * t;
        out.emplace_back(cos(f));
    }
    for (int i = 0; i < 640; i++) {
        double f = *((float *) cv_x.data + i) * 2 * 3.141592653589793 * t;
        out.emplace_back(sin(f));
    }
    cv::Mat cv_out(cv::Size(1280, 1), CV_32FC1);
    memcpy(cv_out.data, out.data(), 1280 * sizeof(float));
    return cv_out;
}