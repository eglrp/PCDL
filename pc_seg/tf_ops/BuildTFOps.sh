#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc TFNeighborKernel.cu -o build/TFNeighborKernel.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_forward_files="TFNeighborScatter.cc \
                  TFLocationWeightFeatSumForward.cc
                  TFLocationWeightSumForward.cc
                 "
tf_backward_files="TFNeighborGather.cc\
                  TFLocationWeightFeatSumBackward.cc
                  TFLocationWeightSumBackward.cc
                  "

echo ${tf_forward_files}
echo ${tf_backward_files}

g++ -std=c++11 -shared build/TFNeighborKernel.cu.o ${tf_forward_files} -o build/libTFNeighborForwardOps.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework \
         -D_GLIBCXX_USE_CXX11_ABI=0


g++ -std=c++11 -shared build/TFNeighborKernel.cu.o ${tf_backward_files} -o build/libTFNeighborBackwardOps.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework \
         -D_GLIBCXX_USE_CXX11_ABI=0