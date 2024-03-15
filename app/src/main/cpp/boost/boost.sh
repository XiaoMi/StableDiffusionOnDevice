#!/bin/bash

echo "Download Boost-for-Android"
git clone https://github.com/moritz-wundke/Boost-for-Android.git

cd "Boost-for-Android"

echo "Build Boost-for-Android"

echo "Enter your NDK_root!!!"

# replace your NDK_ROOT here
# ./build-android.sh ~/Android/Sdk/ndk/android-ndk-r25c --with-libraries=regex --arch=arm64-v8a --boost="1.82.0"
./build-android.sh $NDK_ROOT --with-libraries="regex" --arch="arm64-v8a" --boost="1.82.0"

cp -r "./build/out/arm64-v8a" "../"
rm -r "../Boost-for-Android"
