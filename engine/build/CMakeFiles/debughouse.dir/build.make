# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ben/Desktop/debughouse/engine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ben/Desktop/debughouse/engine/build

# Include any dependencies generated for this target.
include CMakeFiles/debughouse.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/debughouse.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/debughouse.dir/flags.make

CMakeFiles/debughouse.dir/main.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/debughouse.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/main.cpp.o -c /home/ben/Desktop/debughouse/engine/main.cpp

CMakeFiles/debughouse.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/main.cpp > CMakeFiles/debughouse.dir/main.cpp.i

CMakeFiles/debughouse.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/main.cpp -o CMakeFiles/debughouse.dir/main.cpp.s

CMakeFiles/debughouse.dir/engine.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/engine.cpp.o: ../engine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/debughouse.dir/engine.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/engine.cpp.o -c /home/ben/Desktop/debughouse/engine/engine.cpp

CMakeFiles/debughouse.dir/engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/engine.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/engine.cpp > CMakeFiles/debughouse.dir/engine.cpp.i

CMakeFiles/debughouse.dir/engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/engine.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/engine.cpp -o CMakeFiles/debughouse.dir/engine.cpp.s

CMakeFiles/debughouse.dir/planes.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/planes.cpp.o: ../planes.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/debughouse.dir/planes.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/planes.cpp.o -c /home/ben/Desktop/debughouse/engine/planes.cpp

CMakeFiles/debughouse.dir/planes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/planes.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/planes.cpp > CMakeFiles/debughouse.dir/planes.cpp.i

CMakeFiles/debughouse.dir/planes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/planes.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/planes.cpp -o CMakeFiles/debughouse.dir/planes.cpp.s

CMakeFiles/debughouse.dir/bugboard.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/bugboard.cpp.o: ../bugboard.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/debughouse.dir/bugboard.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/bugboard.cpp.o -c /home/ben/Desktop/debughouse/engine/bugboard.cpp

CMakeFiles/debughouse.dir/bugboard.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/bugboard.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/bugboard.cpp > CMakeFiles/debughouse.dir/bugboard.cpp.i

CMakeFiles/debughouse.dir/bugboard.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/bugboard.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/bugboard.cpp -o CMakeFiles/debughouse.dir/bugboard.cpp.s

CMakeFiles/debughouse.dir/network.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/network.cpp.o: ../network.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/debughouse.dir/network.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/network.cpp.o -c /home/ben/Desktop/debughouse/engine/network.cpp

CMakeFiles/debughouse.dir/network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/network.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/network.cpp > CMakeFiles/debughouse.dir/network.cpp.i

CMakeFiles/debughouse.dir/network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/network.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/network.cpp -o CMakeFiles/debughouse.dir/network.cpp.s

CMakeFiles/debughouse.dir/node.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/node.cpp.o: ../node.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/debughouse.dir/node.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/node.cpp.o -c /home/ben/Desktop/debughouse/engine/node.cpp

CMakeFiles/debughouse.dir/node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/node.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/node.cpp > CMakeFiles/debughouse.dir/node.cpp.i

CMakeFiles/debughouse.dir/node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/node.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/node.cpp -o CMakeFiles/debughouse.dir/node.cpp.s

CMakeFiles/debughouse.dir/clock.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/clock.cpp.o: ../clock.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/debughouse.dir/clock.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/clock.cpp.o -c /home/ben/Desktop/debughouse/engine/clock.cpp

CMakeFiles/debughouse.dir/clock.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/clock.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/clock.cpp > CMakeFiles/debughouse.dir/clock.cpp.i

CMakeFiles/debughouse.dir/clock.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/clock.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/clock.cpp -o CMakeFiles/debughouse.dir/clock.cpp.s

CMakeFiles/debughouse.dir/searchthread.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/searchthread.cpp.o: ../searchthread.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/debughouse.dir/searchthread.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/searchthread.cpp.o -c /home/ben/Desktop/debughouse/engine/searchthread.cpp

CMakeFiles/debughouse.dir/searchthread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/searchthread.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/searchthread.cpp > CMakeFiles/debughouse.dir/searchthread.cpp.i

CMakeFiles/debughouse.dir/searchthread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/searchthread.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/searchthread.cpp -o CMakeFiles/debughouse.dir/searchthread.cpp.s

CMakeFiles/debughouse.dir/agent.cpp.o: CMakeFiles/debughouse.dir/flags.make
CMakeFiles/debughouse.dir/agent.cpp.o: ../agent.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/debughouse.dir/agent.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/debughouse.dir/agent.cpp.o -c /home/ben/Desktop/debughouse/engine/agent.cpp

CMakeFiles/debughouse.dir/agent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debughouse.dir/agent.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ben/Desktop/debughouse/engine/agent.cpp > CMakeFiles/debughouse.dir/agent.cpp.i

CMakeFiles/debughouse.dir/agent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debughouse.dir/agent.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ben/Desktop/debughouse/engine/agent.cpp -o CMakeFiles/debughouse.dir/agent.cpp.s

# Object files for target debughouse
debughouse_OBJECTS = \
"CMakeFiles/debughouse.dir/main.cpp.o" \
"CMakeFiles/debughouse.dir/engine.cpp.o" \
"CMakeFiles/debughouse.dir/planes.cpp.o" \
"CMakeFiles/debughouse.dir/bugboard.cpp.o" \
"CMakeFiles/debughouse.dir/network.cpp.o" \
"CMakeFiles/debughouse.dir/node.cpp.o" \
"CMakeFiles/debughouse.dir/clock.cpp.o" \
"CMakeFiles/debughouse.dir/searchthread.cpp.o" \
"CMakeFiles/debughouse.dir/agent.cpp.o"

# External object files for target debughouse
debughouse_EXTERNAL_OBJECTS =

debughouse: CMakeFiles/debughouse.dir/main.cpp.o
debughouse: CMakeFiles/debughouse.dir/engine.cpp.o
debughouse: CMakeFiles/debughouse.dir/planes.cpp.o
debughouse: CMakeFiles/debughouse.dir/bugboard.cpp.o
debughouse: CMakeFiles/debughouse.dir/network.cpp.o
debughouse: CMakeFiles/debughouse.dir/node.cpp.o
debughouse: CMakeFiles/debughouse.dir/clock.cpp.o
debughouse: CMakeFiles/debughouse.dir/searchthread.cpp.o
debughouse: CMakeFiles/debughouse.dir/agent.cpp.o
debughouse: CMakeFiles/debughouse.dir/build.make
debughouse: /usr/local/cuda/lib64/libcudart_static.a
debughouse: /usr/lib/x86_64-linux-gnu/librt.so
debughouse: /usr/lib/x86_64-linux-gnu/libnvinfer.so
debughouse: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
debughouse: /usr/lib/x86_64-linux-gnu/libnvparsers.so
debughouse: Fairy-Stockfish/libFairy-Stockfish.a
debughouse: /usr/lib/x86_64-linux-gnu/librt.so
debughouse: /usr/lib/x86_64-linux-gnu/libnvinfer.so
debughouse: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
debughouse: /usr/lib/x86_64-linux-gnu/libnvparsers.so
debughouse: CMakeFiles/debughouse.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ben/Desktop/debughouse/engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable debughouse"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/debughouse.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/debughouse.dir/build: debughouse

.PHONY : CMakeFiles/debughouse.dir/build

CMakeFiles/debughouse.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/debughouse.dir/cmake_clean.cmake
.PHONY : CMakeFiles/debughouse.dir/clean

CMakeFiles/debughouse.dir/depend:
	cd /home/ben/Desktop/debughouse/engine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ben/Desktop/debughouse/engine /home/ben/Desktop/debughouse/engine /home/ben/Desktop/debughouse/engine/build /home/ben/Desktop/debughouse/engine/build /home/ben/Desktop/debughouse/engine/build/CMakeFiles/debughouse.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/debughouse.dir/depend

