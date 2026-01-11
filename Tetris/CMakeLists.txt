cmake_minimum_required(VERSION 3.20)
project(Tetris CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML from workspace root
get_filename_component(WORKSPACE_ROOT "${CMAKE_SOURCE_DIR}" DIRECTORY)

# Read SFML_DIR from the file saved during SFML build
if(EXISTS "${WORKSPACE_ROOT}/_sfml/SFML_DIR.txt")
    file(READ "${WORKSPACE_ROOT}/_sfml/SFML_DIR.txt" SFML_DIR_ABS)
    string(STRIP "${SFML_DIR_ABS}" SFML_DIR_ABS)
    # Convert Windows paths to CMake format (handles backslashes)
    file(TO_CMAKE_PATH "${SFML_DIR_ABS}" SFML_DIR_ABS)
    set(SFML_DIR "${SFML_DIR_ABS}" CACHE PATH "Path to SFML install directory")
else()
    # Fallback: try to find it in the install directory
    file(TO_CMAKE_PATH "${WORKSPACE_ROOT}/_sfml/install/lib/cmake/SFML" SFML_DIR_ABS)
    set(SFML_DIR "${SFML_DIR_ABS}" CACHE PATH "Path to SFML install directory")
endif()

# Set SFML to use static libraries (since FetchContent builds static by default)
set(SFML_STATIC_LIBRARIES ON CACHE BOOL "Use SFML static libraries" FORCE)

# Set CMAKE_PREFIX_PATH to find SFML dependencies (Ogg, Vorbis, FLAC, etc.)
list(APPEND CMAKE_PREFIX_PATH "${WORKSPACE_ROOT}/_sfml/install")

# Find SFML 3.x
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System Audio)

add_executable(tetris main.cpp squareShape.cpp)
target_link_libraries(tetris PRIVATE SFML::Graphics SFML::Window SFML::System SFML::Audio)

# Set working directory to project root so your relative asset paths work
set_property(TARGET tetris PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
