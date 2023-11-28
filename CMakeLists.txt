cmake_minimum_required(VERSION 3.22)
project(libsvm C)

set(CMAKE_C_STANDARD 11)

#add_library(${PROJECT_NAME} SHARED svm.c svm.h)
add_executable(svm-predict.exe svm-predict.c svm.c)
add_executable(svm-scale.exe svm-scale.c svm.c)
add_executable(svm-train.exe svm-train.c svm.c)

target_link_libraries(svm-predict.exe m)
target_link_libraries(svm-scale.exe m)
target_link_libraries(svm-train.exe m)
