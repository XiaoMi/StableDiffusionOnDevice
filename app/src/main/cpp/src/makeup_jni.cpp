#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>

#include "prompt_solver.h"
#include "diffusion_solver.h"
#include "models/DecoderModel.h"

static std::string UTF16StringToUTF8String(const char16_t *chars, size_t len) {
    std::u16string u16_string(chars, len);
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t>{}
            .to_bytes(u16_string);
}

std::string JavaStringToString(JNIEnv *env, jstring str) {
    if (env == nullptr || str == nullptr) {
        return "";
    }
    const jchar *chars = env->GetStringChars(str, nullptr);
    if (chars == nullptr) {
        return "";
    }
    std::string u8_string = UTF16StringToUTF8String(
            reinterpret_cast<const char16_t *>(chars), env->GetStringLength(str));
    env->ReleaseStringChars(str, chars);
    return u8_string;
}

static PromptSolver prompt_solver;
static DiffusionSolver diffusion_solver;
static DecoderModel decode_solver;


extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnUnload");
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jint JNICALL
Java_com_xiaomi_stablediffusion_StableDiffusion_Init(JNIEnv *env,
                                                     jobject thiz,
                                                     jobject assetManager,
                                                     jstring jpath) {
    std::string path = JavaStringToString(env, jpath);
    if (setenv("LD_LIBRARY_PATH", (path + "/stable_diffusion/qnn_lib_" + QCOM_VERSION +
                                   ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images").c_str(),
               1) == 0) {
        // 成功设置环境变量
        LOGI("setenv finished");
    } else {
        LOGE("setenv failed");
        // 设置环境变量失败
    }
    if (setenv("ADSP_LIBRARY_PATH", (path + "/stable_diffusion/qnn_lib_" + QCOM_VERSION +
                                     ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp").c_str(),
               1) == 0) {
        // 成功设置环境变量
        LOGI("setenv finished");
    } else {
        LOGE("setenv failed");
        // 设置环境变量失败
    }

    if (prompt_solver.load(path) < 0) {
        LOGE("prompt_solver load failed!");
        return -3;
    }

    if (diffusion_solver.load(path) < 0) {
        LOGE("diffusion_solver load failed!");
        return -3;
    }

    if (decode_solver.load(path) < 0) {
        LOGE("decode_solver load failed!");
        return -3;
    }
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_xiaomi_stablediffusion_StableDiffusion_txt2imgProcess(JNIEnv *env,
                                                               jobject thiz,
                                                               jobject show_bitmap,
                                                               jint jstep,
                                                               jint jseed,
                                                               jstring jpositivePromptEn,
                                                               jstring jnegativePromptEn
) {

    Timer timer("txt2img");
    std::string positive_prompt_en = "" + JavaStringToString(env, jpositivePromptEn);
    std::string negative_prompt_en = "" + JavaStringToString(env, jnegativePromptEn);
    std::string default_positive_prompt_en = "mini hamburgers shaped like the faces of cats, in the style of makoto shinkai, hergé, party kei, sculpted, naoto hattori, exquisite detailing";

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return -6;

    diffusion_solver.set_latent_size(info.height / 8, info.width / 8);
    decode_solver.set_latent_size(info.height / 8, info.width / 8);
    int step = jstep;
    int seed = jseed;
    cv::Mat cond, uncond;
    timer.add_record_point("prepare");
    if (prompt_solver.get_conditioning(positive_prompt_en, default_positive_prompt_en, cond) < 0) {
        LOGE("prompt_solver positive prompt inference failed!");
        return -4;
    }
    if (prompt_solver.get_conditioning(negative_prompt_en, "", uncond) < 0) {
        LOGE("prompt_solver negative prompt inference failed!");
        return -4;
    }
    timer.add_record_point("prompt");
    cv::Mat sample;
    if (diffusion_solver.sampler_txt2img(seed, step, cond, uncond, sample) < 0) {
        LOGE("diffusion_solver inference failed!");
        return -4;
    }
    timer.add_record_point("unet");
    cv::Mat x_samples_ddim;
    if (decode_solver.decode(sample, x_samples_ddim) < 0) {
        LOGE("decode_solver inference failed!");
        return -4;
    }
    timer.add_record_point("decoder");
    matToBitmap(env, x_samples_ddim, show_bitmap);
    return 0;
}
}
extern "C"
JNIEXPORT jint JNICALL
Java_com_xiaomi_stablediffusion_StableDiffusion_release(JNIEnv *env, jobject thiz) {
    // TODO: implement release()
    if (prompt_solver.unload() < 0)
        return -7;
    if (diffusion_solver.unload() < 0)
        return -7;
    if (decode_solver.unload() < 0)
        return -7;
    return 0;
}