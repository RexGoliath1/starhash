cmake_minimum_required( VERSION 3.6 )

# Create Project
project( StarHash )
set(CMAKE_CXX_STANDARD 17)  # enable C++17 standard
#set(CMAKE_CXX_STANDARD 11)  # enable C++11 standard

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Set StartUp Project
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME} )

# Find Package
set( OpenCV_DIR /home/gonk/opencv/build ) 
option( OpenCV_STATIC OFF )
find_package( OpenCV REQUIRED )
find_package( Eigen3 3.3 REQUIRED )

if (OpenCV_FOUND)
  list(APPEND USER_INC ${OpenCV_INCLUDE_DIRS} )
  list( APPEND LINK_LIBS ${OpenCV_LIBS})
endif()

if (EIGEN3_FOUND)
  list( APPEND USER_INC ${EIGEN3_INCLUDE_DIR} )
  list( APPEND LINK_LIBS ${EIGEN3_LIBRARIES})
endif()

list( APPEND USER_INC ${PROJECT_SOURCE_DIR/include})

# Set Static Link Runtime Library
if( OpenCV_STATIC )
  foreach( flag_var
           CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
           CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
           CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
           CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO )
    if( ${flag_var} MATCHES "/MD" )
      string( REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}" )
    endif()
  endforeach()
endif()

# Additional Include Directories
include_directories( ${USER_INC} )

file(GLOB ALL_SRC
  "${PROJECT_SOURCE_DIR}/include/*.h"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.c"
)

add_executable( ${CMAKE_PROJECT_NAME} ${ALL_SRC} )

# Additional Library Directories
link_directories( ${OpenCV_LIB_DIR} )

# Additional Dependencies
target_link_libraries( ${CMAKE_PROJECT_NAME} ${LINK_LIBS} )
