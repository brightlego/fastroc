cmake_minimum_required(VERSION 3.26)
project(fastroc C)

set(CMAKE_C_STANDARD 90)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(fastroc fastrocmodule.c)

target_link_libraries(fastroc PRIVATE Threads::Threads)

