cmake_minimum_required(VERSION 3.25)
project(DCReg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(TBB QUIET)
find_package(OpenMP QUIET)

add_executable(dcreg_runner src/main.cpp)
add_executable(dcreg_minimal_example src/dcreg_minimal_example.cpp)

function(configure_dcreg_target target_name)
  target_include_directories(
      ${target_name}
      PRIVATE
      include
      ${PCL_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE Eigen3::Eigen ${PCL_LIBRARIES})
  target_compile_definitions(
      ${target_name}
      PRIVATE
      ${PCL_DEFINITIONS}
      DCREG_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  if(TBB_FOUND)
    target_link_libraries(${target_name} PRIVATE TBB::tbb)
    target_compile_definitions(${target_name} PRIVATE DCREG_HAS_TBB=1)
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target_name} PRIVATE -O3 -march=native)
  endif()
endfunction()

configure_dcreg_target(dcreg_runner)
configure_dcreg_target(dcreg_minimal_example)
