cmake_minimum_required(VERSION 3.20)
project(photoViewer CXX)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard explicitly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# Find SFML package
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)

add_executable(photoViewer photoViewer.cpp)

# Set C++ standard on target as well
set_target_properties(photoViewer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Link against SFML
target_link_libraries(photoViewer PRIVATE 
    SFML::Graphics 
    SFML::Window 
    SFML::System
)

# Set working directory to project root so your relative asset paths work
set_property(TARGET photoViewer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
