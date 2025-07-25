cmake_minimum_required(VERSION 3.10)
project(MLP_Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(SOURCES
    main.cpp
    genetic.cpp
    Perceptrone.cpp
)


set(HEADERS
    genetic.h
    Perceptrone.h
    mlpActivators.hpp
)


add_executable(MLP ${SOURCES} ${HEADERS})

