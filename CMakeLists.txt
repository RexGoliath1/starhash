cmake_minimum_required(VERSION 3.6)
project(StarHash)
set(CMAKE_CXX_STANDARD 17)  # enable C++17 standard
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check OS
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif()

# Build Configures
message("Build with " ${CMAKE_CXX_COMPILER_ID})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall ")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Set StartUp Project
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

# Find Package
option(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package(PkgConfig)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

list(APPEND USER_INC ${PROJECT_SOURCE_DIR}/include)

if (OpenCV_FOUND)
  list(APPEND USER_INC ${OpenCV_INCLUDE_DIRS})
  list(APPEND LINK_LIBS ${OpenCV_LIBS})
endif()

if (EIGEN3_FOUND)
    message("EIGEN3_INC: ${EIGEN3_INCLUDE_DIR}")
    list(APPEND USER_INC ${EIGEN3_INCLUDE_DIR})
endif()

if (HDF5_FOUND)
    message("${HDF5_INCLUDE_DIRS}")
    message("${HDF5_CXX_LIBRARIES}")
    list(APPEND USER_INC ${HDF5_INCLUDE_DIRS})
    list(APPEND LINK_LIBS ${HDF5_CXX_LIBRARIES})
endif()


list(APPEND USER_INC ${PROJECT_SOURCE_DIR/include})

# Only link stdc++fs on Linux
if (NOT MACOSX)
    list(APPEND LINK_LIBS stdc++fs)
endif()

# Additional Include Directories
include_directories(${USER_INC})

message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

file(GLOB ALL_SRC
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.c"
)

add_executable(${CMAKE_PROJECT_NAME} ${ALL_SRC})

# Additional Library Directories
link_directories(${OpenCV_LIB_DIR} ${HDF5_LIBRARY_DIRS})

# Additional Dependencies
target_link_libraries(${CMAKE_PROJECT_NAME} ${LINK_LIBS} Eigen3::Eigen yaml-cpp::yaml-cpp)

