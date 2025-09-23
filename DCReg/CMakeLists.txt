cmake_minimum_required(VERSION 3.25)
project(DCReg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release) # Controls optimization (-O3) and NDEBUG macro

# --- Find Packages ---
# if you have Open3D installed in a custom location, set Open3D_DIR accordingly
set(Open3D_DIR "/home/xchu/xicp_ws/devel/.private/open3d_catkin/lib/cmake/Open3D")
message(STATUS "LTLoc: Set Open3D_DIR to ${Open3D_DIR}") # 打印信息，确认设置生效
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
# Prefer Conda's yaml-cpp (to match headers) if available
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/cmake/yaml-cpp")
    set(yaml-cpp_DIR "$ENV{CONDA_PREFIX}/lib/cmake/yaml-cpp")
endif()
find_package(yaml-cpp REQUIRED)  # YAML configuration
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED) # Keep this even if optional later, checks if system has it

# check the version of these necessary packages
message(STATUS "Found Open3D version: ${Open3D_VERSION}")
message(STATUS "Found PCL version: ${PCL_VERSION}")
message(STATUS "Found Ceres version: ${CERES_VERSION}")
# yaml-cpp version can come from the config package as `yaml-cpp_VERSION`
if(DEFINED yaml-cpp_VERSION)
    set(YAML_CPP_VERSION_STR "${yaml-cpp_VERSION}")
elseif(DEFINED YAML_CPP_VERSION)
    set(YAML_CPP_VERSION_STR "${YAML_CPP_VERSION}")
else()
    set(YAML_CPP_VERSION_STR "unknown")
endif()
message(STATUS "Found yaml-cpp version: ${YAML_CPP_VERSION_STR}")
message(STATUS "Found TBB version: ${TBB_VERSION}")

# CMake's FindOpenMP doesn't provide OpenMP_VERSION; use SPEC_DATE to approximate
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found and enabled where used.")
    if(DEFINED OpenMP_CXX_SPEC_DATE)
        set(_omp_date ${OpenMP_CXX_SPEC_DATE})
        set(_omp_version "")
        if(_omp_date GREATER_EQUAL 202111)
            set(_omp_version "5.2")
        elseif(_omp_date GREATER_EQUAL 202011)
            set(_omp_version "5.1")
        elseif(_omp_date GREATER_EQUAL 201811)
            set(_omp_version "5.0")
        elseif(_omp_date GREATER_EQUAL 201511)
            set(_omp_version "4.5")
        elseif(_omp_date GREATER_EQUAL 201307)
            set(_omp_version "4.0")
        elseif(_omp_date GREATER_EQUAL 201107)
            set(_omp_version "3.1")
        elseif(_omp_date GREATER_EQUAL 200805)
            set(_omp_version "3.0")
        elseif(_omp_date GREATER_EQUAL 200505)
            set(_omp_version "2.5")
        else()
            set(_omp_version "pre-2.5")
        endif()
        message(STATUS "Found OpenMP spec date: ${OpenMP_CXX_SPEC_DATE} (approx v${_omp_version})")
    else()
        message(STATUS "Found OpenMP (version unknown)")
    endif()
else ()
    message(WARNING "OpenMP not found. Parallel sections will run serially.")
endif ()


# --- Include Directories ---
include_directories(
        include # Your local include directory
        ${Open3D_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR} # Add Eigen include dir explicitly
        ${CERES_INCLUDE_DIRS} # Add Ceres include dir explicitly
        ${YAML_CPP_INCLUDE_DIR}  # Added for YAML (may be empty when using imported target)
)

# Create executable
add_executable(icp_test_runner
        src/icp_test_runner.cpp
        src/superloc.cpp
        src/xicp.cpp
        src/icp_main.cpp
        )
target_link_libraries(icp_test_runner
        ${PCL_LIBRARIES}
        Open3D::Open3D
        Eigen3::Eigen
        # Link yaml-cpp via imported target when available (e.g. Conda)
        $<$<TARGET_EXISTS:yaml-cpp::yaml-cpp>:yaml-cpp::yaml-cpp>
        # Fallback to variable if only variables are defined
        $<$<NOT:$<TARGET_EXISTS:yaml-cpp::yaml-cpp>>:${YAML_CPP_LIBRARIES}>
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
        ${CERES_LIBRARIES}
        TBB::tbb
        )

# Set optimization flags
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(icp_test_runner PRIVATE -O3 -march=native)
endif ()
