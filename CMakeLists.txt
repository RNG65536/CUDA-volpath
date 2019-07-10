cmake_minimum_required(VERSION 3.12)
project(volpath)

set(srcs
"src/param.h"
"src/volumeRender_kernel.cu"
"src/volumeRender.cpp"
)

enable_language(CUDA)
include_directories($ENV{NVCUDASAMPLES_ROOT}/common/inc)

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)

add_executable(volpath ${srcs})

target_include_directories(volpath PRIVATE ${GLM_INCLUDE_DIRS} ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR})
target_link_libraries(volpath PRIVATE
    debug ${GLUT_glut_LIBRARY_DEBUG}
    optimized ${GLUT_glut_LIBRARY_RELEASE}
    debug ${GLEW_LIBRARY_DEBUG}
    optimized ${GLEW_LIBRARY_RELEASE}
)