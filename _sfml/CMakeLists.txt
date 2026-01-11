cmake_minimum_required(VERSION 3.20)
project(SFML_Workspace CXX)

include(FetchContent)

FetchContent_Declare(
    SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.2
    GIT_SHALLOW TRUE
)

set(SFML_BUILD_EXAMPLES OFF CACHE BOOL "Build the SFML examples" FORCE)
set(SFML_BUILD_DOC OFF CACHE BOOL "Build the SFML documentation" FORCE)
set(SFML_INSTALL_PKGCONFIG_FILES OFF CACHE BOOL "Install pkg-config files" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

FetchContent_MakeAvailable(SFML)
