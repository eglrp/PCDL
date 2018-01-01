cmake_minimum_required(VERSION 3.5)
project(ExtendedOperator)

#execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INCLUDE)
#execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB)

#set (TF_INCLUDE /usr/local/lib/python2.7/dist-packages/tensorflow/include)
#set (TF_LIB /usr/local/lib/python2.7/dist-packages/tensorflow)

exec_program(python ARGS "-c \"import tensorflow; print(tensorflow.sysconfig.get_include())\"" OUTPUT_VARIABLE TF_INCLUDE)
exec_program(python ARGS "-c \"import tensorflow; print(tensorflow.sysconfig.get_lib())\"" OUTPUT_VARIABLE TF_LIB)

message(STATUS ${TF_INCLUDE})
message(STATUS ${TF_LIB})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(SOURCE_FILES IndicesPool.cc)
add_library(IndicesPool SHARED ${SOURCE_FILES})
target_include_directories(IndicesPool PUBLIC ${TF_INCLUDE};${TF_INCLUDE}/external/nsync/public)
target_link_libraries(IndicesPool ${TF_LIB}/libtensorflow_framework.so)


add_library(IndicesPoolGrad SHARED IndicesPoolGrad.cc)
target_include_directories(IndicesPoolGrad PUBLIC ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
target_link_libraries(IndicesPoolGrad ${TF_LIB}/libtensorflow_framework.so)

add_library(GlobalIndicesPool SHARED GlobalIndicesPool.cc)
target_include_directories(GlobalIndicesPool PUBLIC ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
target_link_libraries(GlobalIndicesPool ${TF_LIB}/libtensorflow_framework.so)


add_library(GlobalIndicesPoolGrad SHARED GlobalIndicesPoolGrad.cc)
target_include_directories(GlobalIndicesPoolGrad PUBLIC ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
target_link_libraries(GlobalIndicesPoolGrad ${TF_LIB}/libtensorflow_framework.so)


add_library(ContextBlockPool SHARED ContextBlockPool.cc)
target_include_directories(ContextBlockPool PUBLIC ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
target_link_libraries(ContextBlockPool ${TF_LIB}/libtensorflow_framework.so)

add_library(ContextBlockPoolGrad SHARED ContextBlockPoolGrad.cc)
target_include_directories(ContextBlockPoolGrad PUBLIC PUBLIC ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
target_link_libraries(ContextBlockPoolGrad ${TF_LIB}/libtensorflow_framework.so)

add_library(ContextBatchPool SHARED ContextBatchPool.cc)
target_include_directories(ContextBatchPool PUBLIC ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
target_link_libraries(ContextBatchPool ${TF_LIB}/libtensorflow_framework.so)

add_library(ContextBatchPoolGrad SHARED ContextBatchPoolGrad.cc)
target_include_directories(ContextBatchPoolGrad PUBLIC ${TF_INCLUDE} ${TF_INCLUDE}/external/nsync/public)
target_link_libraries(ContextBatchPoolGrad ${TF_LIB}/libtensorflow_framework.so)