cmake_minimum_required(VERSION 3.15)
project(rbm)
set(CMAKE_CXX_STANDARD 17)
add_executable(rbm
    src/main.cpp
    src/maths/matrix.cpp
    src/maths/activations.cpp
    src/maths/loss.cpp
    src/machine/rbm.cpp
    src/data/mnist.cpp
)
target_include_directories(rbm PRIVATE src)
