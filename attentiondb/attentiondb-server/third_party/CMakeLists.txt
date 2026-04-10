include(FetchContent)

# Allow older cmake_minimum_required in fetched deps (CMake 4.x compat)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "")

# ── libcuckoo (header-only concurrent hash map) ─────────────────────────────

FetchContent_Declare(
    libcuckoo
    GIT_REPOSITORY https://github.com/efficient/libcuckoo.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(libcuckoo)

# ── xxHash ───────────────────────────────────────────────────────────────────

FetchContent_Declare(
    xxHash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.3
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cmake_unofficial
)
FetchContent_MakeAvailable(xxHash)

# ── yaml-cpp ─────────────────────────────────────────────────────────────────

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        yaml-cpp-0.9.0
    GIT_SHALLOW    TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# ── jemalloc (system or FetchContent) ────────────────────────────────────────

find_library(JEMALLOC_LIB jemalloc)
if(JEMALLOC_LIB)
    add_library(jemalloc INTERFACE)
    target_link_libraries(jemalloc INTERFACE ${JEMALLOC_LIB})
    message(STATUS "Using system jemalloc: ${JEMALLOC_LIB}")
else()
    message(STATUS "jemalloc not found on system — using malloc as fallback")
    add_library(jemalloc INTERFACE)
endif()

# ── liburing (Linux only, system package) ────────────────────────────────────

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(URING_LIB uring)
    if(NOT URING_LIB)
        message(WARNING "liburing not found. Install with: apt install liburing-dev")
    endif()
endif()

# ── pybind11 ─────────────────────────────────────────────────────────────────

if(ATTENTIONDB_BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v3.0.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── GoogleTest ───────────────────────────────────────────────────────────────

if(ATTENTIONDB_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(googletest)
endif()

# ── Google Benchmark ─────────────────────────────────────────────────────────

if(ATTENTIONDB_BUILD_BENCH)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.5
        GIT_SHALLOW    TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()
