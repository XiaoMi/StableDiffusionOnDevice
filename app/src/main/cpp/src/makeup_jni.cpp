
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
                                                               jstring jpositivePromptCh,
                                                               jstring jnegativePromptCh,
                                                               jstring jpositivePromptEn,
                                                               jstring jnegativePromptEn
) {

    Timer timer("txt2img");
    std::string positive_prompt_ch = "" + JavaStringToString(env, jpositivePromptCh);
    std::string negative_prompt_ch = "" + JavaStringToString(env, jnegativePromptCh);
    std::string positive_prompt_en = "" + JavaStringToString(env, jpositivePromptEn);
    std::string negative_prompt_en = "" + JavaStringToString(env, jnegativePromptEn);
    std::string default_positive_prompt_en = "Japanese garden at wildlife river and mountain range, highly detailed, digital illustration, artstation, concept art, matte, sharp focus, illustration, dramatic, sunset, hearthstone, art by Artgerm and Greg Rutkowski and Alphonse Mucha.";

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return -6;

    diffusion_solver.set_latent_size(info.height / 8, info.width / 8);
    decode_solver.set_latent_size(info.height / 8, info.width / 8);
    int step = jstep;
    int seed = jseed;
    timer.add_record_point("prepare");
    auto cond = prompt_solver.get_conditioning(positive_prompt_ch, positive_prompt_en,
                                               "", default_positive_prompt_en, 0);
    auto uncond = prompt_solver.get_conditioning(negative_prompt_ch, negative_prompt_en,
                                                 "", "", 0);
    timer.add_record_point("prompt");
    cv::Mat sample = diffusion_solver.sampler_txt2img(seed, step, cond, uncond);
    timer.add_record_point("unet");
    cv::Mat x_samples_ddim = decode_solver.decode(sample);
    timer.add_record_point("decoder");
    matToBitmap(env, x_samples_ddim, show_bitmap);
    return 0;
}
}