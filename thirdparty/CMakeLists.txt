


#set OIDN libraries 
set(OID_DIR "${CMAKE_CURRENT_SOURCE_DIR}/OpenImageDenoise")
find_library(OIDN_LIB "OpenImageDenoise" "${OID_DIR}/lib")
find_library(OIDN_MKLDNN_LIB "mkldnn" "${OID_DIR}/lib")
find_library(OIDN_COMMON_LIB "common" "${OID_DIR}/lib")

set(OIDN_LIBRARIES ${OIDN_LIB} ${OIDN_MKLDNN_LIB} ${OIDN_COMMON_LIB})

add_library(oidn INTERFACE)
target_include_directories(oidn INTERFACE ${OID_DIR}/include)
target_link_libraries(oidn INTERFACE ${OIDN_LIBRARIES})


add_library(optix7 INTERFACE)
target_include_directories(optix7 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/OptiX7/include)

