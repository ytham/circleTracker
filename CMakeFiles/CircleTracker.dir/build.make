# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.8.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.8.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yujiang.tham/cv/circleTracker

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yujiang.tham/cv/circleTracker

# Include any dependencies generated for this target.
include CMakeFiles/CircleTracker.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CircleTracker.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CircleTracker.dir/flags.make

CMakeFiles/CircleTracker.dir/circleTracker.cpp.o: CMakeFiles/CircleTracker.dir/flags.make
CMakeFiles/CircleTracker.dir/circleTracker.cpp.o: circleTracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yujiang.tham/cv/circleTracker/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CircleTracker.dir/circleTracker.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CircleTracker.dir/circleTracker.cpp.o -c /Users/yujiang.tham/cv/circleTracker/circleTracker.cpp

CMakeFiles/CircleTracker.dir/circleTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CircleTracker.dir/circleTracker.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yujiang.tham/cv/circleTracker/circleTracker.cpp > CMakeFiles/CircleTracker.dir/circleTracker.cpp.i

CMakeFiles/CircleTracker.dir/circleTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CircleTracker.dir/circleTracker.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yujiang.tham/cv/circleTracker/circleTracker.cpp -o CMakeFiles/CircleTracker.dir/circleTracker.cpp.s

CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.requires:

.PHONY : CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.requires

CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.provides: CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/CircleTracker.dir/build.make CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.provides.build
.PHONY : CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.provides

CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.provides.build: CMakeFiles/CircleTracker.dir/circleTracker.cpp.o


# Object files for target CircleTracker
CircleTracker_OBJECTS = \
"CMakeFiles/CircleTracker.dir/circleTracker.cpp.o"

# External object files for target CircleTracker
CircleTracker_EXTERNAL_OBJECTS =

CircleTracker: CMakeFiles/CircleTracker.dir/circleTracker.cpp.o
CircleTracker: CMakeFiles/CircleTracker.dir/build.make
CircleTracker: /usr/local/lib/libopencv_videostab.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_ts.a
CircleTracker: /usr/local/lib/libopencv_superres.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_stitching.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_contrib.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_nonfree.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_ocl.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_gpu.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_photo.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_objdetect.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_legacy.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_video.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_ml.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_calib3d.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_features2d.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_highgui.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_imgproc.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_flann.2.4.13.dylib
CircleTracker: /usr/local/lib/libopencv_core.2.4.13.dylib
CircleTracker: CMakeFiles/CircleTracker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yujiang.tham/cv/circleTracker/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CircleTracker"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CircleTracker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CircleTracker.dir/build: CircleTracker

.PHONY : CMakeFiles/CircleTracker.dir/build

CMakeFiles/CircleTracker.dir/requires: CMakeFiles/CircleTracker.dir/circleTracker.cpp.o.requires

.PHONY : CMakeFiles/CircleTracker.dir/requires

CMakeFiles/CircleTracker.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CircleTracker.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CircleTracker.dir/clean

CMakeFiles/CircleTracker.dir/depend:
	cd /Users/yujiang.tham/cv/circleTracker && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yujiang.tham/cv/circleTracker /Users/yujiang.tham/cv/circleTracker /Users/yujiang.tham/cv/circleTracker /Users/yujiang.tham/cv/circleTracker /Users/yujiang.tham/cv/circleTracker/CMakeFiles/CircleTracker.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CircleTracker.dir/depend

