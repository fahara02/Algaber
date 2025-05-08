
cmake_minimum_required(VERSION 3.23.1) 
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")

set(CMAKE_CXX_STANDARD 23)
project("Algaber"
        VERSION 0.0.1
        DESCRIPTION "Header Only Math libraries in cpp23"
        LANGUAGES CXX)  
set(BINARY_NAME ${PROJECT_NAME})  


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(ClangFormat)

add_clang_format_target(format-code ${PROJECT_SOURCE_DIR})

# Add library target for Algaber
add_library(${PROJECT_NAME}_lib INTERFACE)
target_include_directories(${PROJECT_NAME}_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/linear_algebra)

# Main executable
add_executable(${BINARY_NAME} src/main.cpp)
target_link_libraries(${BINARY_NAME} PRIVATE ${PROJECT_NAME}_lib)

# Google Test integration
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Add test executable
add_executable(matrix_test tests/matrix_test.cpp)
target_link_libraries(matrix_test PRIVATE ${PROJECT_NAME}_lib GTest::gtest_main)

# Register tests
include(GoogleTest)
gtest_discover_tests(matrix_test)
