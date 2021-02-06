cmake_minimum_required(VERSION 3.12)
project(volpath)

include(D:/deps.cmake) # append thirdparty search paths to CMAKE_PREFIX_PATH

option(USE_OPENVDB "Build OpenVDB loader" ON)

file(GLOB_RECURSE srcs "src/*.*")

set(CMAKE_CXX_STANDARD 11)

enable_language(CUDA)
include_directories($ENV{NVCUDASAMPLES_ROOT}/common/inc)

find_package(openmp QUIET)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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