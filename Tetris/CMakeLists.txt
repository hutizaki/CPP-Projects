cmake_minimum_required(VERSION 3.20)
project(Tetris CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML 3.x from Homebrew
find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System Audio)

add_executable(tetris main.cpp squareShape.cpp)
target_link_libraries(tetris PRIVATE SFML::Graphics SFML::Window SFML::System SFML::Audio)

# Help the app find dylibs at runtime (so you can run from build/ directly)
set_target_properties(tetris PROPERTIES
    BUILD_RPATH "/opt/homebrew/lib"
    INSTALL_RPATH "/opt/homebrew/lib"
    MACOSX_RPATH ON
)

# Set working directory to project root so your relative asset paths work
set_property(TARGET tetris PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
