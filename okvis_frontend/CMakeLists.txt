cmake_minimum_required(VERSION 3.8...3.30)

find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
if(USE_SYSTEM_BRISK)
  find_package(brisk REQUIRED)
endif()
if(USE_SYSTEM_DBOW2)
  find_package(DBoW2 REQUIRED)
endif()
if(USE_SYSTEM_CERES)
  find_package(Ceres REQUIRED)
endif()
if(USE_NN)
  find_package(Torch REQUIRED)
endif()

set(LIB_NAME okvis_frontend)
add_library(${LIB_NAME}
  include/DBoW2/FBrisk.hpp
  include/okvis/triangulation/stereo_triangulation.hpp
  include/opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp
  include/opengv/absolute_pose/LoopclosureNoncentralAbsoluteAdapter.hpp
  include/opengv/relative_pose/FrameRelativeAdapter.hpp
  include/opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp
  src/FBrisk.cpp
  src/FrameNoncentralAbsoluteAdapter.cpp
  src/FrameRelativeAdapter.cpp
  src/Frontend.cpp
  src/LoopclosureNoncentralAbsoluteAdapter.cpp
  src/stereo_triangulation.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${DBoW2_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${OPENGV_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  PUBLIC
    Eigen3::Eigen
    okvis::Ceres
    okvis::CV
    okvis::Kinematics
    okvis::Timing
    okvis::Util
    brisk::brisk
    ${DBoW2_LIBRARIES}
  PRIVATE
    ${GLOG_LIBRARIES}
    ${OPENGV_LIBRARIES}
    ${OpenCV_LIBS}
    Ceres::ceres
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)
if(USE_NN)
  target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_NN C10_USE_GLOG)
  target_compile_options(${LIB_NAME} PUBLIC ${TORCH_CXX_FLAGS})
  target_include_directories(${LIB_NAME}
  PUBLIC
    ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(${LIB_NAME}
    PRIVATE
      ${TORCH_LIBRARIES}
  )
  if(USE_GPU)
    target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_GPU)
  endif()
endif()
add_library(okvis::Frontend ALIAS ${LIB_NAME})

# the DBoW2 pre-built vocabulary
add_custom_command(TARGET ${LIB_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/../resources/small_voc.yml.gz
          ${CMAKE_CURRENT_BINARY_DIR}/..)

# and the network
add_custom_command(TARGET ${LIB_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/../resources/fast-scnn.pt
          ${CMAKE_CURRENT_BINARY_DIR}/..)

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
