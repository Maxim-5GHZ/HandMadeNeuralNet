cmake_minimum_required(VERSION 3.10)
project(MLP_Project)

# Устанавливаем стандарт C++ (17 или выше)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Исходные файлы проекта
set(SOURCES
    main.cpp
    mlp.cpp
)

# Заголовочные файлы
set(HEADERS
    mlp.h
)

# Создаем исполняемый файл
add_executable(MLP ${SOURCES} ${HEADERS})

