# CMakeList.txt: LDA 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)
include_directories(D:/boost_1_74_0)
LINK_DIRECTORIES(D:/boost_1_74_0/stage/lib)
# 将源代码添加到此项目的可执行文件。
add_executable (LDA "LDA.cpp" "LDA.h" "csv.h""main.cpp")

# TODO: 如有需要，请添加测试并安装目标。
