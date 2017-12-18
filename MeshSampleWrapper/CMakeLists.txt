cmake_minimum_required(VERSION 2.8)
project(MeshSampleWrapper)

set(CMAKE_CXX_STANDARD 11)

set(MESH_SAMPLE_SOURCE batch_file_io.cpp batch_file_io.h mesh_sample.cpp mesh_sample.h preprocess.cpp preprocess.h)

set(MESH_SAMPLE_TEST_SOURCE ${MESH_SAMPLE_SOURCE} test.cpp)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

add_library(mesh_sample SHARED ${MESH_SAMPLE_SOURCE})
add_executable(mesh_sample_test ${MESH_SAMPLE_TEST_SOURCE})
#add_library(mesh_sample_python SHARED python_wrapper.cpp)

#target_include_directories(mesh_sample_python PRIVATE /usr/include/python2.7)