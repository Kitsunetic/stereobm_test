cmake_minimum_required(VERSION 2.8)
project( opencv_stereobm_module_test )

set( COMMON_SRCS "yg_stereobm.cpp" )
set( PLANE_SRC ${COMMON_SRCS} "plane_test.cpp" )
set( SPHERICAL_SRC ${COMMON_SRCS} "spherical_test.cpp" )

set( ld_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/lib )
set( include_flag /home/choyg/workspace/opencv_ffmpeg/opencv-3.4.2/install/include )
set( libs opencv_img_hash opencv_world )

include_directories( ${include_flag} )
link_directories( ${ld_flag} )

add_executable( "plane_test.out" ${PLANE_SRC} )
add_executable( "sphercial_test.out" ${SPHERICAL_SRC} )

target_link_libraries( "plane_test.out" ${libs} )
target_link_libraries( "sphercial_test.out" ${libs} )
