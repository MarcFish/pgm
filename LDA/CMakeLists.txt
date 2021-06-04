cmake_minimum_required(VERSION 2.8)

project(LDA)

aux_source_directory(. DIR_SRCS)

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

add_executable(LDA ${DIR_SRCS})
TARGET_LINK_LIBRARIES(LDA ${Boost_LIBRARIES})

