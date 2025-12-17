cmake_minimum_required(VERSION 3.14)
project(diff_sim_core VERSION 25.1 LANGUAGES CXX)

include_directories(include)

set(CMAKE_CXX_STANDARD 17)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -march=native)
endif()

# 1. Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 2. Find Pybind11
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 config not found, attempting to locate via Python...")
    
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmake
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if (PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
    
    find_package(pybind11 REQUIRED)
endif()

# 3. Find Eigen (try system first, then FetchContent)
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, downloading via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# 4. Find SDL2
if(WIN32)
    find_package(SDL2 REQUIRED)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2 REQUIRED sdl2)
endif()

# 5. Define the Module
set(SOURCES
    src/bindings.cpp
    src/engine/tensor.cpp
    src/engine/activations.cpp
    src/engine/optimizers.cpp
    src/engine/body.cpp
    src/engine/contact.cpp
    src/renderer/sdl_renderer.cpp
    src/engine/engine.cpp
)
pybind11_add_module(rigidRL ${SOURCES})

# Link Libraries
if(WIN32)
    target_link_libraries(rigidRL PRIVATE Eigen3::Eigen SDL2::SDL2)
    
    add_custom_command(TARGET rigidRL POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:SDL2::SDL2>
        $<TARGET_FILE_DIR:rigidRL>
    )
else()
    target_include_directories(rigidRL PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(rigidRL PRIVATE Eigen3::Eigen ${SDL2_LIBRARIES})
    target_link_options(rigidRL PRIVATE -static-libstdc++ -static-libgcc)
endif()