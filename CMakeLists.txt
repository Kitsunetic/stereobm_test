cmake_minimum_required(VERSION 2.8)
project( opencv_stereobm_module_test )

set( INCLUDE_DIR "./srcs" )
set( SRCS )
aux_source_directory( "./srcs" SRCS)

set( PLANE_TEST_SRC )
aux_source_directory( "./plane_test" PLANE_TEST_SRC)
set( SPHERICAL_TEST_SRC )
aux_source_directory( "./spherical_test" SPHERICAL_TEST_SRC)
set( MULTI_LAYER_TEST_SRC )
aux_source_directory( "./multi_layer_test" MULTI_LAYER_TEST_SRC)

set( ld_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/lib )
set( include_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/include )
set( libs opencv_img_hash opencv_world OpenMP::OpenMP_CXX )
find_package(OpenMP REQUIRED)

include_directories( ${include_flag} ${INCLUDE_DIR} )
link_directories( ${ld_flag} )

add_executable( "plane_test.out" ${SRCS} ${PLANE_TEST_SRC} )
add_executable( "spherical_test.out" ${SRCS} ${SPHERICAL_TEST_SRC} )
add_executable( "multi_layer_test.out" ${SRCS} ${MULTI_LAYER_TEST_SRC} )

target_link_libraries(plane_test.out ${libs} ${CMAKE_THREAD_LIBS_INIT} ${CERES_LIBRARIES} )
target_link_libraries(spherical_test.out ${libs} ${CMAKE_THREAD_LIBS_INIT} ${CERES_LIBRARIES} )
target_link_libraries(multi_layer_test.out ${libs} ${CMAKE_THREAD_LIBS_INIT} ${CERES_LIBRARIES} )
