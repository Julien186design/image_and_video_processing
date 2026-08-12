cmake_minimum_required(VERSION 3.16)
project(Image_Processing)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Standard C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inclusion du dossier stb
include_directories(${CMAKE_SOURCE_DIR}/stb)

# Détection d'OpenCV
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Activation de OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found; multithreading support enabled.")
else()
    message(FATAL_ERROR "OpenMP not found. Install it using 'sudo apt install libomp-dev'.")
endif()

# Exécutable principal
add_executable(Image_and_video_processing
        ColorConfig.h
        EdgeDetector.h
        Image.cpp
        Image.h
        ImageBuffer.h
        ImageCreation.cpp
        ImageCreation.h
        main.cpp
        ProcessingConfig.h
        ProcessingConfig.cpp
        schrift.h
        VideoProcessing.h
        VideoCreation.cpp
        VideoCreation.h
)

# Lien avec OpenCV et OpenMP
target_link_libraries(Image_and_video_processing PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
CHECK_CXX_COMPILER_FLAG("-fopenmp" COMPILER_SUPPORTS_OPENMP)

target_compile_options(Image_and_video_processing PRIVATE
        -O3
        -march=native
        -mavx2
        -mfma
)

if(COMPILER_SUPPORTS_OPENMP)
    message(STATUS "OpenMP support, enabling compilation flags.")
    target_compile_options(Image_and_video_processing PRIVATE -fopenmp)
else()
    message(FATAL_ERROR "The compiler does not support OpenMP. Check your GCC installation.")
endif()