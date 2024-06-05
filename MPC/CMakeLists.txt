cmake_minimum_required(VERSION 3.0.0)
project(MPC_demo VERSION 0.1.0 LANGUAGES C CXX)

aux_source_directory(. ROOT_SOURCE)
set(EIGEN_DIR "${PROJECT_SOURCE_DIR}/external/eigen")

include_directories(${PROJECT_SOURCE_DIR}/external)

find_package(PythonLibs 3.10 COMPONENTS Development NumPy)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

aux_source_directory(src DIR_SRCS)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
add_executable(MPC_demo ${DIR_SRCS})
target_include_directories(MPC_demo PUBLIC ${EIGEN_DIR})
target_include_directories(MPC_demo PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(MPC_demo ${PYTHON_LIBRARIES})