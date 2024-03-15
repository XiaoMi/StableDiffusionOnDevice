package com.xiaomi.stablediffusion;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.text.SpannableString;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import android.os.Handler;
import android.os.Message;

import java.util.HashMap;
import java.util.Map;

public class Text2GraphActivity extends AppCompatActivity {
    public static final StableDiffusion sd = new StableDiffusion();
    private ImageView imageView;
    private EditText positivePromptText;
    private EditText negativePromptText;
    private TextView tv_time;
    private EditText seedText;
    private EditText stepText;
    private EditText img_sizeText;
    private int imgHeight = 512;
    private int imgWidth = 512;

    private Bitmap showBitmap;
    private int step;
    private int seed;
    private String postivePrompt_ch = "", negativePrompt_ch = "", postivePrompt_en, negativePrompt_en;
    private SDTimer sdTimer;
    private Map<String, String> resultMap = new HashMap<>();
    public static AIThread handlerThread;
    private boolean modelLoadStatus = false;
    private AlertDialog.Builder barBuilder;
    private Dialog dialog;
    private long startTime, endTime;
    public static InitThread loadHandlerThread;
    private int init_res, txt2img_res;
    private Handler uiHandler;

    public class InitThread extends HandlerThread {

        private static final String TAG = "Load Thread";

        public static final int TYPE_START = 8;//主线程通知任务开始
        public static final int TYPE_COPY_FINISHED = 9;//通知主线程COPY任务结束
        public static final int TYPE_FINISHED = 10;//通知主线程任务结束

        private Handler mHandler;//主线程的Handler

        public InitThread(String name) {
            super(name);
        }

        //注入主线程Handler
        public void setUIHandler(Handler UIhandler) {
            mHandler = UIhandler;
            Log.i(TAG, "setUIHandler: 2.主线程的handler传入到Init线程");
        }

        public void startCopy() {
            Log.i(TAG, "startCopy: 3.接收主线程通知,此时Init线程开始进行模型拷贝");
            try {
                copyAssets(Text2GraphActivity.this.getResources().getAssets(), "stable_diffusion", new File(Text2GraphActivity.this.getFilesDir().getAbsolutePath()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            mHandler.sendEmptyMessage(TYPE_COPY_FINISHED);
        }

        public void startLoad() {
            Log.i(TAG, "startLoad: 4.接收主线程通知,此时Init线程开始加载模型");
            String path = Text2GraphActivity.this.getFilesDir().getAbsolutePath();
            startTime = System.currentTimeMillis();
            init_res = sd.Init(getAssets(), path);

            Log.i(TAG, "startLoad: 5.通知主线程,此时Init线程加载模型完成" + init_res);
            mHandler.sendEmptyMessage(TYPE_FINISHED);
        }
    }

    public class AIThread extends HandlerThread {

        private static final String TAG = "AI Thread";

        public static final int TYPE_START = 0;//主线程通知任务开始
        public static final int TYPE_FINISHED = 1;//通知主线程任务结束

        private Handler mUIHandler;//主线程的Handler

        public AIThread(String name) {
            super(name);
        }

        //注入主线程Handler
        public void setUIHandler(Handler UIhandler) {
            mUIHandler = UIhandler;
            Log.i(TAG, "setUIHandler: 2.主线程的handler传入到AI线程");
        }

        public void startTXT2IMG() {
            Log.i(TAG, "startTXT2IMG: 3.接收主线程通知,此时AI线程开始文生图");
            txt2img_res = sd.txt2imgProcess(showBitmap, step, seed, postivePrompt_ch, negativePrompt_ch, postivePrompt_en, negativePrompt_en);
            Log.i(TAG, "startTXT2IMG: 6.通知主线程,此时AI线程文生图完成");
            mUIHandler.sendEmptyMessage(TYPE_FINISHED);
        }
    }


    /**
     * 用于计时，在主线程中使用此方法
     */
    private class SDTimer {

        private int time = 0;
        private long base = SystemClock.elapsedRealtime();
        private int interval = 10;//设置间隔时间
        private Runnable mRunnable = new MyRunnable(); // 定时器

        /**
         * 创建对象开始计时
         *
         * @param interval 间隔时间通知(使用第一个方法，默认1秒钟刷新一次)
         */
        public SDTimer(int interval) {
            this.interval = interval;
        }

        /**
         * 开始计时
         */
        public void start() {
            base = (long) ((int) SystemClock.elapsedRealtime() + interval * 1);
            time = 0;
            mRunnable = new MyRunnable();
            mHandler.postDelayed(mRunnable, interval);
        }

        /**
         * 终止计时
         */
        public void stop() {
            mHandler.removeCallbacks(mRunnable);
            mRunnable = null;
        }

        private Handler mHandler = new Handler(Looper.getMainLooper());

        private class MyRunnable implements Runnable {
            @Override
            public void run() {
                time = (int) (SystemClock.elapsedRealtime() - base) + interval;
                int min = time / 1000 / 60;
                int sec = time / 1000 % 60;
                int msec = time % 1000;
                String timeString = String.format("%02d:%02d:%03d", min, sec, msec);
                tv_time.setText(timeString);
                mHandler.postDelayed(this, interval);
            }
        }

    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_text2_graph);

        imageView = (ImageView) findViewById(R.id.resView);
        positivePromptText = (EditText) findViewById(R.id.pos);
        negativePromptText = (EditText) findViewById(R.id.neg);
        stepText = (EditText) findViewById(R.id.step);
        seedText = (EditText) findViewById(R.id.seed);
        img_sizeText = findViewById(R.id.image_size);
        imgWidth = Integer.valueOf(img_sizeText.getText().toString());
        imgHeight = Integer.valueOf(img_sizeText.getText().toString());
        showBitmap = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888);
        tv_time = (TextView) findViewById(R.id.tv_timer);
        sdTimer = new SDTimer(100);
        modelLoadStatus = false;
        uiHandler = new Handler(getMainLooper());
        barBuilder = new AlertDialog.Builder(this);
        barBuilder.setCancelable(false);
        barBuilder.setView(R.layout.processbar);
        dialog = barBuilder.create();
        // Long click to save image
        imageView.setOnLongClickListener(new LongClickHandler());
        Button buttonTXT2IMG = (Button) findViewById(R.id.txt2img);
        buttonTXT2IMG.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                postivePrompt_en = positivePromptText.getText().toString();
                negativePrompt_en = negativePromptText.getText().toString();
                String text = seedText.getText().toString().replaceAll(" ", "");
                if (text.equals("")) {
                    Toast.makeText(Text2GraphActivity.this, "Seed设置无效，请重新输入", Toast.LENGTH_LONG).show();
                    getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                    return;
                } else {
                    seed = Integer.parseInt(text);
                }
                step = Integer.valueOf(stepText.getText().toString());
                imgWidth = Integer.valueOf(img_sizeText.getText().toString());
                imgHeight = Integer.valueOf(img_sizeText.getText().toString());
                showBitmap = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888);

                Handler mUIhandler = new Handler(handlerThread.getLooper()) {
                    @Override
                    public void handleMessage(Message msg) {
                        super.handleMessage(msg);
                        //判断mHandlerThread里传来的msg，根据msg进行主页面的UI更改
                        switch (msg.what) {
                            case AIThread.TYPE_START:
                                runOnUiThread(new Runnable() {
                                    @SuppressLint("UseCompatLoadingForDrawables")
                                    @Override
                                    public void run() {
                                        imageView.setImageDrawable(getDrawable(R.drawable.robot));
                                        if (postivePrompt_en.replaceAll(" ", "").equals("")) {
                                            postivePrompt_en = "Japanese garden at wildlife river and mountain range, highly detailed, digital illustration, artstation, concept art, matte, sharp focus, illustration, dramatic, sunset, hearthstone, art by Artgerm and Greg Rutkowski and Alphonse Mucha.";
                                            positivePromptText.setText(postivePrompt_en);
                                        }
                                        sdTimer.start();
                                    }
                                });
                                handlerThread.startTXT2IMG();
                                break;
                            case AIThread.TYPE_FINISHED:
                                Log.i("AI", "主线程知道AI线程生图完成了...这时候可以更改主界面UI，收工");

                                final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888, true);
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        sdTimer.stop();
                                        if (txt2img_res < 0)
                                            Toast.makeText(Text2GraphActivity.this, "生图推理失败，请重试", Toast.LENGTH_LONG).show();
                                        else
                                            imageView.setImageBitmap(styledImage);
                                        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                                    }
                                });
                                break;
                            default:
                                break;
                        }
                    }
                };
                handlerThread.setUIHandler(mUIhandler);
                if (modelLoadStatus)
                    mUIhandler.sendEmptyMessage(AIThread.TYPE_START);
                else {
                    Toast.makeText(Text2GraphActivity.this, " 模型未加载，请加载完成后再次点击生图", Toast.LENGTH_LONG).show();
                    init_model();
                    getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                }
            }
        });
    }

    protected void onResume() {
        super.onResume();
        handlerThread = new AIThread("SD HandlerThread");
        handlerThread.start();
        loadHandlerThread = new InitThread("Init HandlerThread");
        loadHandlerThread.start();
        init_model();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        handlerThread.quitSafely();
        loadHandlerThread.quitSafely();
        try {
            handlerThread.join();
            loadHandlerThread.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private void init_model() {

        Handler mhandler = new Handler(loadHandlerThread.getLooper()) {
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                //判断mHandlerThread里传来的msg，根据msg进行主页面的UI更改
                switch (msg.what) {
                    case InitThread.TYPE_START:
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                dialog.show();
                            }
                        });
                        loadHandlerThread.startCopy();
                        break;
                    case InitThread.TYPE_COPY_FINISHED:
                        loadHandlerThread.startLoad();
                        break;
                    case InitThread.TYPE_FINISHED:
                        endTime = System.currentTimeMillis();
                        float duration = (float) ((endTime - startTime) / 1000.0f);
                        if (init_res <= -3) {
                            modelLoadStatus = false;
                            Toast.makeText(Text2GraphActivity.this, "模型加载失败，请检查APK", Toast.LENGTH_LONG).show();
                            Log.e("MainActivity", "SD Init failed");
                        } else if (init_res < 0) {
                            modelLoadStatus = false;
                            Toast.makeText(Text2GraphActivity.this, "模型完整性检测失败，请检查模型", Toast.LENGTH_LONG).show();
                            Log.e("MainActivity", "SD Init failed");
                        } else {
                            modelLoadStatus = true;
                            uiHandler.post(new Runnable() {
                                @Override
                                public void run() {
                                    dialog.dismiss();
                                }
                            });
                        }
                        break;
                    default:
                        break;
                }
            }
        };
        loadHandlerThread.setUIHandler(mhandler);
        if (!modelLoadStatus) {
            mhandler.sendEmptyMessage(InitThread.TYPE_START);
        }
    }

    private static void copyFile(AssetManager assetManager, String fileName, File outPath) throws IOException {
        File file = new File(outPath, fileName);
        if (!file.exists()) {
            Log.v("copyFile", "Copy " + fileName + " to " + outPath);
            InputStream in = assetManager.open(fileName);
            OutputStream out = new FileOutputStream(outPath + "/" + fileName);
            byte[] buffer = new byte[7168];

            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }

            in.close();
            out.close();
        }
    }

    public static void copyAssets(AssetManager assetManager, String path, File outPath) throws IOException {
        String[] assets = assetManager.list(path);

        if (assets != null) {
            if (assets.length == 0) {
                copyFile(assetManager, path, outPath);
            } else {
                File dir = new File(outPath, path);
                if (!dir.exists()) {
                    if (!dir.mkdirs()) {
                        Log.v("copyAssets", "Failed to create directory " + dir.getAbsolutePath());
                    }
                }

                String[] var5 = assets;
                int var6 = assets.length;

                for (int var7 = 0; var7 < var6; ++var7) {
                    String asset = var5[var7];
                    copyAssets(assetManager, path + "/" + asset, outPath);
                }
            }

        }
    }

    class LongClickHandler implements View.OnLongClickListener {
        @Override
        public boolean onLongClick(View view) {
            boolean bRet = SaveJpg((ImageView) view);
            if (bRet) {
                Toast.makeText(Text2GraphActivity.this, "图片保存成功", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(Text2GraphActivity.this, "图片保存失败", Toast.LENGTH_SHORT).show();
            }
            return true;
        }

        public boolean SaveJpg(ImageView view) {

            try {
                Drawable drawable = view.getDrawable();
                if (drawable == null) {
                    return false;
                }

                ContentValues values = new ContentValues();
                values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");

                Uri dataUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
                Uri fileUri = view.getContext().getContentResolver().insert(dataUri, values);

                if (fileUri == null) {
                    return false;
                }

                OutputStream outStream = view.getContext().getContentResolver().openOutputStream(fileUri);

                Bitmap bitmap = ((BitmapDrawable) drawable).getBitmap();
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, outStream);
                outStream.flush();
                outStream.close();

                // 刷新相册
                view.getContext().sendBroadcast(new Intent("com.android.camera.NEW_PICTURE", fileUri));

                return true;

            } catch (IOException ex) {
                return false;
            }
        }
    }

}
