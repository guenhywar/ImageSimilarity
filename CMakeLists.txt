cmake_minimum_required(VERSION 2.8.3)
project(image_similarity)
find_package(catkin REQUIRED robosherlock rs_bs)
################################################################################
## Constants for project                                                      ##
################################################################################
set(NAMESPACE image_similarity)
set(TYPESYSTEM_CPP_PATH ${PROJECT_SOURCE_DIR}/include/image_similarity/types)
set(TYPESYSTEM_XML_PATH ${PROJECT_SOURCE_DIR}/descriptors/typesystem)
set(ANNOTATOR_PATH      ${PROJECT_SOURCE_DIR}/descriptors/annotators)
set(ENGINE_PATH         ${PROJECT_SOURCE_DIR}/descriptors/analysis_engines)
set(CMAKE_MODULE_PATH   ${PROJECT_SOURCE_DIR}/cmake)
################################################################################
## Configure project configuration files                                      ##
################################################################################
configure_file(${RS_PROJECT_CONFIG} ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}_config.cmake)
################################################################################
## Find all include directories                                               ##
################################################################################
find_include_dirs(RS_INCLUDE_DIRS_LIST)
catkin_package(
   INCLUDE_DIRS include
   CFG_EXTRAS ${PROJECT_NAME}_config.cmake
   )
################################################################################
## Package dependencies                                                       ##
################################################################################
#Add required catkin packages here
#find_package(catkin REQUIRED COMPONENTS xxx)
#Add required system packages here
#find_package(xxx)
################################################################################
## Build                                                                      ##
################################################################################
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        ${RS_INCLUDE_DIRS_LIST}
        ${catkin_INCLUDE_DIRS}
        )
################################################################################
## Update analysis engines, typesystem and include all relevant files         ##
################################################################################
## generate classes from the typesystem xml files
generate_type_system(robosherlock image_similarity)
#find all relevant files
find_additional_files()
################################################################################
## Sub Projects                                                               ##
################################################################################
#If you want to divide your projects into subprojects include the subdirectories
#each containing a CMakeLists.txt here
#add_subdirectory(src/xxx)


#rs_add_library(rs_ImageSimilarity src/ImageSimilarity.cpp)
#target_link_libraries(rs_ImageSimilarity ${catkin_LIBRARIES})
#rs_add_library(rs_ImageSimilarityPointCloud src/ImageSimilarityPointCloud.cpp)
#target_link_libraries(rs_ImageSimilarityPointCloud ${catkin_LIBRARIES})
#rs_add_library(rs_SimilaritySummarizer src/SimilaritySummarizer.cpp)
#target_link_libraries(rs_SimilaritySummarizer ${catkin_LIBRARIES})

#rs_add_library(rs_SimClusterColorFromBeliefState src/ClusterColorFromBeliefState.cpp)
#target_link_libraries(rs_SimClusterColorFromBeliefState ${catkin_LIBRARIES})

rs_add_library(rs_imsim src/imsim.cpp)
target_link_libraries(rs_imsim ${catkin_LIBRARIES})

rs_add_library(rs_baseline src/baselineClassifier.cpp)
target_link_libraries(rs_baseline ${catkin_LIBRARIES})

