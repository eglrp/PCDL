TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared IndicesPool.cc -o build/libIndicesPool.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework -O2 \
         -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared IndicesPoolGrad.cc -o build/libIndicesPoolGrad.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework -O2 \
         -D_GLIBCXX_USE_CXX11_ABI=0
