# CMakeList.txt: LDA 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)
project(LDA)

set(Boost_INCLUDE_DIR D:/boost_1_74_0)
set(Boost_LIBRARY_DIR D:/boost_1_74_0/stage/lib)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# 将源代码添加到此项目的可执行文件。
add_executable (LDA "LDA.cpp" "main.cpp")
target_link_libraries( LDA ${Boost_LIBRARIES} )

# TODO: 如有需要，请添加测试并安装目标。
