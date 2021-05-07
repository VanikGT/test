cmake_minimum_required(VERSION 3.9)
project(tensor C CXX)
# ##################### OpenCV ############################

set( OpenCV_DIR C:/OpenCV-android-sdk/sdk/native/jni )

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS}
            )
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else(OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif(OpenCV_FOUND)

# ###################### End OpenCV ###########################

FIND_PACKAGE(PythonLibs 3)
FIND_PACKAGE(PythonInterp 3)
include_directories(${PYTHON_INCLUDE_DIRS})

set(TENSORFLOW_SOURCE_DIR "C:\\Users\\vansa\\Desktop\\tensorflow_src" CACHE PATH
        "Directory that contains the TensorFlow project"
        )
if(NOT TENSORFLOW_SOURCE_DIR)
    get_filename_component(C:\\Users\\vansa\\Desktop\\tensorflow_src
            "${CMAKE_CURRENT_LIST_DIR}/../../../../"
            ABSOLUTE
            )
endif()

add_subdirectory(
        "C:\\Users\\vansa\\Desktop\\tensorflow_src\\tensorflow\\lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
        EXCLUDE_FROM_ALL
)

set(CMAKE_CXX_STANDARD 11)

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        src/main/cpp/conv_sdk.cpp
        src/main/cpp/native_lib.cpp)

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

target_link_libraries( # Specifies the target library.
        native-lib
        tensor
        ${CMAKE_DL_LIBS}
        ${OpenCV_LIBS}
        ${PYTHON_LIBRARIES}
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib} )