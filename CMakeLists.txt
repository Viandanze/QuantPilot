cmake_minimum_required(VERSION 3.15)
project(QuantPilotMatching VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Native limit order book core + test harness + micro-benchmark.
# No external dependencies: plain C++17 and the standard library.
# ---------------------------------------------------------------------------
add_executable(qp_orderbook_test cpp/order_book_test.cpp cpp/order_book.cpp)
target_include_directories(qp_orderbook_test PRIVATE cpp)

add_executable(qp_orderbook_bench cpp/order_book_bench.cpp cpp/order_book.cpp)
target_include_directories(qp_orderbook_bench PRIVATE cpp)

# ---------------------------------------------------------------------------
# Python extension module (qp_matching) built with pybind11.
#
# Typical setup:
#   pip install pybind11
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#         -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
#   cmake --build build
#
# The shared object is placed in the repository root so that `import
# qp_matching` works from the project directory (it is gitignored).
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(qp_matching cpp/bindings.cpp cpp/order_book.cpp)
    target_include_directories(qp_matching PRIVATE cpp)
    set_target_properties(qp_matching PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
else()
    message(WARNING
        "pybind11 not found - the qp_matching Python module will NOT be built. "
        "Install it with `pip install pybind11` and point CMake at it, e.g. "
        "-Dpybind11_DIR=\"$(python -m pybind11 --cmakedir)\". "
        "The native test and benchmark binaries are unaffected.")
endif()
