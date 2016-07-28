cmake_minimum_required(VERSION 3.1)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_definitions(-std=c++11 -g)

find_library(TF_LIBRARIES tensorflow-cpu
    PATHS "/usr/local/cuda/lib64"
    DOC "tensorflow-cpu library")

include_directories(
    ${Eigen_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_libraries(
    ${OpenCV_LIBS}
)

add_executable(example_node
    src/example_node.cpp)

target_link_libraries(example_node
    ${TF_LIBRARIES})
