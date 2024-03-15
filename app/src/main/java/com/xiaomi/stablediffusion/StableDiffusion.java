package com.xiaomi.stablediffusion;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class StableDiffusion {
    public native int Init(AssetManager mgr, String path);

    public native int txt2imgProcess(Bitmap show_bitmap, int step, int seed,
                                     String positivePromptEn, String negativePromptEn);

    public native int release();

    static {
        System.loadLibrary("stablediffusion");
    }
}
