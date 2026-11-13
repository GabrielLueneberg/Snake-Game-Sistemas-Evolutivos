cmake_minimum_required(VERSION 3.10)
project(SnakeGame)
 
set(CMAKE_CXX_STANDARD 17)
 
add_executable(main main.cpp)
target_link_libraries(main PRIVATE raylib opengl32 gdi32 winmm)
 
# Otimizacao: o treino roda ~10x mais rapido que sem ela (mesmo no preset Debug)
target_compile_options(main PRIVATE -O2)