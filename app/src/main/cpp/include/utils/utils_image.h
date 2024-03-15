#ifndef STABLEDIFFUSION_UTILS_IMAGE_H
#define STABLEDIFFUSION_UTILS_IMAGE_H

#include "utils_base.h"

void bitmapToMat(JNIEnv *env, jobject bitmap, cv::Mat &dst);

void matToBitmap(JNIEnv *env, cv::Mat &src, jobject bitmap);

int convert_hwc_to_chw(const float *src, float *target, int height, int width, int channel);

int convert_chw_to_hwc(const float *src, float *target, int height, int width, int channel);

cv::Mat FourierFeatures (const float t);

#endif //STABLEDIFFUSION_UTILS_IMAGE_H
