cmake_minimum_required(VERSION 3.20...4.3)
project(attentiondb-server VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-Wall -Wextra -Wpedantic)

option(ATTENTIONDB_CUDA "Enable CUDA pinned memory and GDS support" OFF)
option(ATTENTIONDB_BUILD_TESTS "Build unit tests" ON)
option(ATTENTIONDB_BUILD_BENCH "Build benchmarks" ON)
option(ATTENTIONDB_BUILD_PYTHON "Build pybind11 Python module" ON)

add_subdirectory(third_party)

# ── Core library ──────────────────────────────────────────────────────────────

add_library(attentiondb_core STATIC
    src/core/config.cpp
    src/core/status.cpp
    src/index/concurrent_hashmap.cpp
    src/memory/slab_allocator.cpp
    src/storage/blob_store.cpp
    src/storage/uring_engine.cpp
    src/storage/posix_engine.cpp
    src/eviction/cw_slru.cpp
    src/admission/admission.cpp
    src/checkpoint/checkpoint.cpp
    src/logging/logger.cpp
    src/engine.cpp
    src/c_api.cpp
)

target_include_directories(attentiondb_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(Threads REQUIRED)

target_link_libraries(attentiondb_core PUBLIC
    libcuckoo
    xxHash::xxhash
    yaml-cpp
    Threads::Threads
)

# Link liburing on Linux only
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(attentiondb_core PUBLIC uring)
endif()

# Link jemalloc
target_link_libraries(attentiondb_core PUBLIC jemalloc)

if(ATTENTIONDB_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(attentiondb_core PUBLIC ATTENTIONDB_CUDA=1)
    target_link_libraries(attentiondb_core PUBLIC CUDA::cudart)
endif()

# ── Shared library for C API ─────────────────────────────────────────────────

add_library(attentiondb SHARED src/c_api_shim.cpp)
target_link_libraries(attentiondb PRIVATE attentiondb_core)

set_target_properties(attentiondb PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ── Python bindings ──────────────────────────────────────────────────────────

if(ATTENTIONDB_BUILD_PYTHON)
    pybind11_add_module(_attentiondb python/attentiondb_py.cpp)
    target_link_libraries(_attentiondb PRIVATE attentiondb_core)
    install(TARGETS _attentiondb DESTINATION python/attentiondb)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────

if(ATTENTIONDB_BUILD_TESTS)
    enable_testing()

    function(add_attentiondb_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE attentiondb_core GTest::gtest_main)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_attentiondb_test(test_core_types   tests/test_core_types.cpp)
    add_attentiondb_test(test_hashmap      tests/test_hashmap.cpp)
    add_attentiondb_test(test_slab         tests/test_slab.cpp)
    add_attentiondb_test(test_blob_store   tests/test_blob_store.cpp)
    add_attentiondb_test(test_eviction     tests/test_eviction.cpp)
    add_attentiondb_test(test_admission    tests/test_admission.cpp)
    add_attentiondb_test(test_checkpoint   tests/test_checkpoint.cpp)
    add_attentiondb_test(test_engine       tests/test_engine.cpp)
endif()

# ── Benchmarks ───────────────────────────────────────────────────────────────

if(ATTENTIONDB_BUILD_BENCH)
    add_executable(bench_hashmap bench/bench_hashmap.cpp)
    target_link_libraries(bench_hashmap PRIVATE attentiondb_core benchmark::benchmark benchmark::benchmark_main)

    add_executable(bench_engine bench/bench_engine.cpp)
    target_link_libraries(bench_engine PRIVATE attentiondb_core benchmark::benchmark benchmark::benchmark_main)
endif()
