find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)

add_library(vdbloader STATIC
    "load_vdb.h"
    "load_vdb.cpp"
)

target_link_libraries(vdbloader PRIVATE OpenVDB::openvdb)

target_include_directories(vdbloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})