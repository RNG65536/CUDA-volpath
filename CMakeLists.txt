cmake_minimum_required(VERSION 3.12)
project(volpath)

include(D:/deps.cmake)

option(USE_OPENVDB "Build OpenVDB loader" ON)

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

if (USE_OPENVDB)
    add_subdirectory(vdbloader)
    add_definitions("-DUSE_OPENVDB=1")
endif()

add_executable(volpath ${srcs})

target_include_directories(volpath PRIVATE ${GLM_INCLUDE_DIRS} ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIR})
target_link_libraries(volpath PRIVATE
    GLUT::GLUT
    GLEW::GLEW
)

if (USE_OPENVDB)
    target_link_libraries(volpath PRIVATE vdbloader)
endif()