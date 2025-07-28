cmake_minimum_required(VERSION 3.10)
project(GeneticSnakeTrainer)

set(CMAKE_CXX_STANDARD 17)


find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIR})


add_executable(MLP 
    main.cpp 
    genetic.cpp 
    Perceptrone.cpp
)

target_link_libraries(MLP ${CURSES_LIBRARIES})