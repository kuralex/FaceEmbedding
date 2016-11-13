cmake_minimum_required (VERSION 2.6)

set(BUILD_EXAMPLES OFF CACHE BOOL "Set to ON to build examples")

add_subdirectory(SeetaFaceEngine/FaceDetection)
add_subdirectory(SeetaFaceEngine/FaceAlignment)
add_subdirectory(SeetaFaceEngine/FaceIdentification)

set_target_properties(./test_face_verification.bin PROPERTIES EXCLUDE_FROM_ALL 1)

add_subdirectory(src)
