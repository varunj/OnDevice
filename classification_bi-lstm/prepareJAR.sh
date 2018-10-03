#!/bin/bash
# copy to /home/varunj/Downloads/tf_on_mobile/tensorflow and execute

ANDROID_HOME=/home/varunj/Downloads/android-sdk
NDK_HOME=/home/varunj/Downloads/android-sdk/ndk-bundle
DEPLOY_DIR=$(dirname $0)/deploy_android/tensorflow

for cpu in x86 x86_64 armeabi-v7a arm64-v8a
do
	bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=$cpu --cxxopt=-std=c++11
	retval=$?
	if [ $retval -ne 0 ]; then
		exit $retval
	fi

	lib_dir=$DEPLOY_DIR/prebuiltLibs/$cpu
	mkdir -p $lib_dir
	chmod -R u+w $lib_dir # to make sure we can override existing files
	cp bazel-bin/tensorflow/contrib/android/libtensorflow_inference.so $lib_dir
done


bazel build //tensorflow/contrib/android:android_tensorflow_inference_java
retval=$?
if [ $retval -ne 0 ]; then
	exit $retval
fi
chmod -R u+w $DEPLOY_DIR # to make sure we can override previous jar
cp bazel-bin/tensorflow/contrib/android/libandroid_tensorflow_inference_java.jar $DEPLOY_DIR
