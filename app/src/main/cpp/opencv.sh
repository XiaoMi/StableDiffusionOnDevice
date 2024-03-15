#!/bin/bash

echo "Download OpenCV mobile from https://github.com/nihui/opencv-mobile/releases/tag/v16"
echo "Download opencv-mobile-4.6.0-android"

wget https://github.com/nihui/opencv-mobile/releases/download/v16/opencv-mobile-4.6.0-android.zip
unzip opencv-mobile-4.6.0-android.zip
rm opencv-mobile-4.6.0-android.zip