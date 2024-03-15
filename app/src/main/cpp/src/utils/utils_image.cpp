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


int convert_hwc_to_chw(const float *src, float *target, int height, int width, int channel) {
    for (int k = 0; k < channel; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                *(target + k * height * width + i * width + j) =
                        *(src + i * width * channel + j * channel + k);
            }
        }
    }
    return 0;
}

int convert_chw_to_hwc(const float *src, float *target, int height, int width, int channel) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channel; k++) {
                *(target + i * width * channel + j * channel + k) =
                        *(src + k * height * width + i * width + j);
            }
        }
    }
    return 0;
}
