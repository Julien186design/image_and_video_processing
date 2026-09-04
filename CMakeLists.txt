cmake_minimum_required(VERSION 3.16)
project(Image_Processing CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++20 Standard Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Static Analysis Tools (clang-tidy & cppcheck)
# ------------------------------------------------------------------------------
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY
            ${CLANG_TIDY_EXE}
            "-checks=bugprone-*,cert-*,clang-analyzer-*,performance-*,portability-*,readability-*,\
-readability-identifier-length,\
-readability-magic-numbers,\
-bugprone-easily-swappable-parameters,\
-readability-function-cognitive-complexity,\
-readability-braces-around-statements"
            "-header-filter=${CMAKE_SOURCE_DIR}/(include|src)/.*"
    )
endif()

find_program(CPPCHECK_EXE NAMES "cppcheck")
if(CPPCHECK_EXE)
    set(CMAKE_CXX_CPPCHECK
            ${CPPCHECK_EXE}
            "--enable=warning,style,performance,portability"
            "--inline-suppr"
            # Ignore the stb folder completely for cppcheck
            "--suppress=*:stb/*"
    )
endif()

# ------------------------------------------------------------------------------
# External Dependencies
# ------------------------------------------------------------------------------
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# ------------------------------------------------------------------------------
# Main Target Executable
# ------------------------------------------------------------------------------
add_executable(Image_and_video_processing
        src/main.cpp
        src/Image.cpp
        src/ImageCreation.cpp
        src/ProcessingConfig.cpp
        src/VideoCreation.cpp

        include/ColorConfig.h
        include/EdgeDetector.h
        include/Image.h
        include/ImageBuffer.h
        include/ImageCreation.h
        include/ProcessingConfig.h
        include/schrift.h
        include/VideoProcessing.h
        include/VideoCreation.h
        include/ColorBandMask.h
        #CLD
        src/CLD.cpp
        src/ETF.cpp
        include/const.h
        include/CLD.h
        include/ETF.h
)

# Include directories: 'include' as PRIVATE, 'stb' as SYSTEM to suppress third-party compiler warnings
target_include_directories(Image_and_video_processing PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(Image_and_video_processing SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/stb
)

# Compiler Warning Flags
target_compile_options(Image_and_video_processing PRIVATE
        -Wall
        -Wextra
        -Wconversion
        -Wshadow
        -Wpedantic
)

# ------------------------------------------------------------------------------
# Build Options: Sanitizers vs Performance Optimizations
# ------------------------------------------------------------------------------
option(ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" ON)

if(ENABLE_SANITIZERS)
    message(STATUS "Sanitizers (ASan + UBSan) enabled")
    target_compile_options(Image_and_video_processing PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
            -g
            -O1
    )
    target_link_options(Image_and_video_processing PRIVATE
            -fsanitize=address,undefined
    )
else()
    target_compile_options(Image_and_video_processing PRIVATE
            -O3
            -march=native
            -mavx2
            -mfma
    )
endif()

# ------------------------------------------------------------------------------
# Target Link Libraries
# ------------------------------------------------------------------------------
target_link_libraries(Image_and_video_processing PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)