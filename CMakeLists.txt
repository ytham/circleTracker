cmake_minimum_required(VERSION 2.8)
project( CircleTracker )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( CircleTracker circleTracker.cpp )
target_link_libraries( CircleTracker ${OpenCV_LIBS} )