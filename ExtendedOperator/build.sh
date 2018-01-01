TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#g++ -std=c++11 -shared IndicesPool.cc -o build/libIndicesPool.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared IndicesPoolGrad.cc -o build/libIndicesPoolGrad.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared GlobalIndicesPool.cc -o build/libGlobalIndicesPool.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared GlobalIndicesPoolGrad.cc -o build/libGlobalIndicesPoolGrad.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared ContextBlockPool.cc -o build/libContextBlockPool.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared ContextBlockPoolGrad.cc -o build/libContextBlockPoolGrad.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared ContextBatchPool.cc -o build/libContextBatchPool.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 -shared ContextBatchPoolGrad.cc -o build/libContextBatchPoolGrad.so \
#         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
#         -L$TF_LIB -ltensorflow_framework -O2 \
#         -D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 -shared ContextBatchPool.cc GlobalIndicesPool.cc ContextBlockPool.cc  -o build/libPool.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework -O2 \
         -D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 -shared ContextBatchPoolGrad.cc GlobalIndicesPoolGrad.cc ContextBlockPoolGrad.cc  -o build/libPoolGrad.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework -O2 \
         -D_GLIBCXX_USE_CXX11_ABI=0