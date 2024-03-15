#ifndef STABLEDIFFUSION_UTILS_BASE_H
#define STABLEDIFFUSION_UTILS_BASE_H

#define TAG "StableDiffusion"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,TAG ,__VA_ARGS__) // 定义LOGF类型

#define ORT_NO_EXCEPTIONS
#define BOOST_REGEX_STANDALONE

#include <jni.h>
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <unordered_map>
#include <set>
#include <codecvt>
#include <stdio.h>
#include <math.h>
#include <android/bitmap.h>
#include <algorithm>
#include <time.h>
#include <random>
#include <cmath>
#include <numeric>
#include <sys/stat.h>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class Timer {
public:
    Timer(const char *str) {
        _str = str;
        gettimeofday(&_begin, NULL);
    }

    void add_record_point(const char *str) {
        struct timeval current;
        gettimeofday(&current, NULL);
        _record_tm.push_back(current);
        _record_str.push_back(str);
    }

    void add_record_avg() {
        gettimeofday(&_end, NULL);
        _elapsed = get_time_diff(_begin, _end);

        extern int timer_count;
        extern double timer_sum_elapsed;
        timer_sum_elapsed += _elapsed;
        ++timer_count;
        LOGD("timer %s avg cost %f", _str, timer_sum_elapsed / (double) timer_count);
    }

    ~Timer() {
        if (!_record_tm.empty()) {
            struct timeval record_start = _begin;
            for (int i = 0; i < _record_tm.size(); ++i) {
                auto record_end = _record_tm[i];
                auto record_str = _record_str[i];
                auto record_dur = get_time_diff(record_start, record_end);
                LOGD("%s breakpoint %s cost %f", _str, record_str.c_str(), record_dur);

                record_start = record_end;
            }
        }

        gettimeofday(&_end, NULL);
        _elapsed = get_time_diff(_begin, _end);
        LOGD("timer %s cost %f", _str, _elapsed);
    }

private:
    double get_time_diff(struct timeval begin_tm, struct timeval end_tm) {
        double begin_t = begin_tm.tv_sec * 1000.0 + begin_tm.tv_usec / 1000.0;
        double end_t = end_tm.tv_sec * 1000.0 + end_tm.tv_usec / 1000.0;
        return end_t - begin_t;
    }

private:
    struct timeval _begin;
    struct timeval _end;

    std::vector<struct timeval> _record_tm;
    std::vector<std::string> _record_str;
    double _elapsed;
    const char *_str;
};


#endif //STABLEDIFFUSION_UTILS_BASE_H
