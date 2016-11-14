cmake_minimum_required (VERSION 2.6)

find_package(OpenCV REQUIRED)

add_executable(FaceEmbedding FaceEmbedding.cpp)
include_directories(${seeta_facedet_lib_SOURCE_DIR}/include ${seeta_fa_lib_SOURCE_DIR}/include ${viplnet_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(FaceEmbedding viplnet seeta_facedet_lib seeta_fa_lib ${OpenCV_LIBS})
