# 高通库文件
由于高通库文件法律原因不能直接释放，需要再高通官网上自行下载：
1. 首先登录高通官网，https://www.qualcomm.com，可以用任意邮箱进行注册
2. 下载QPM或者QPM3：QPM地址：https://www.qualcomm.com/support/software-tools/tools.qualcomm-package-manager.5828f45e-a704-4f0c-937b-f9730a923652#overview
3. 在QPM中找到Qualcomm AI Stack，在AI Stack中选择Qualcomm AI Engine Direct SDK，找到对应版本进行下载
4. qnn_lib_8550中所需要的文件来自于v2.14.0.230828:
    * libQnnCpu.so, libQnnHtpNetRunExtensions.so, libQnnHtpProfilingReader.so, libQnnHtpV73CalculatorStub.so, libQnnHtpV73Stub.so, libQnnGpu.so, libQnnHtpPrepare.so, libQnnHtp.so, libQnnSystem.so来自$ROOT/lib/aarch64-android
    * libQnnHtpV73Skel.so来自$ROOT/lib/hexagon-v73/unsigned/
5. qnn_lib_8650中所需要的文件来自于v2.16.0.231029:
    * libQnnCpu.so, libQnnHtpNetRunExtensions.so, libQnnHtpProfilingReader.so, libQnnHtpV75CalculatorStub.so, libQnnHtpV75Stub.so, libQnnGpu.so, libQnnHtpPrepare.so, libQnnHtp.so, libQnnSystem.so来自$ROOT/lib/aarch64-android
    * libQnnHtpV75Skel.so来自$ROOT/2.16.0.231029/lib/hexagon-v75/unsigned