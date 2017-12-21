cmake_minimum_required(VERSION 2.8)
project(ExtendedOperator)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(ExtendedOperator ${SOURCE_FILES})
target_include_directories(ExtendedOperator PUBLIC "/usr/local/lib/python2.7/dist-packages/tensorflow/include")
target_link_libraries(ExtendedOperator )