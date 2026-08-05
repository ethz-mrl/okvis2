cmake_minimum_required(VERSION 3.8...3.30)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED core imgcodecs imgproc highgui)

# Generate the configured header file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/okvisConfig.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/include/okvis/okvisConfig.hpp" @ONLY)

set(LIB_NAME okvis_multisensor_processing)
add_library(${LIB_NAME}
  include/okvis/DatasetReader.hpp
  include/okvis/DatasetWriter.hpp
  include/okvis/QueuedTrajectory.hpp
  include/okvis/Realsense.hpp
  include/okvis/RealsenseRgbd.hpp
  include/okvis/RpgDatasetReader.hpp
  include/okvis/ThreadedSlam.hpp
  include/okvis/TrajectoryOutput.hpp
  include/okvis/ViSensorBase.hpp
  include/okvis/ViVisualizer.hpp
  include/okvis/implementation/QueuedTrajectory.hpp
  include/okvis/threadsafe/ThreadsafeQueue.hpp
  src/DatasetReader.cpp
  src/DatasetWriter.cpp
  src/RpgDatasetReader.cpp
  src/ThreadedSlam.cpp
  src/TrajectoryOutput.cpp
  src/ViVisualizer.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GLOG_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  glog::glog
  ${OpenCV_LIBS}
  Boost::filesystem
  Eigen3::Eigen
  okvis::CV
  okvis::Ceres
  okvis::Common
  okvis::Frontend
  okvis::Kinematics
  okvis::Time
  okvis::Timing
  okvis::Util
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)
if(HAVE_LIBREALSENSE)
  find_package(realsense2 REQUIRED)
  target_sources(${LIB_NAME}
    PRIVATE
      src/Realsense.cpp
      src/RealsenseRgbd.cpp
  )
  target_link_libraries(${LIB_NAME}
    realsense2::realsense2
  )
endif()
add_library(okvis::MultisensorProcessing ALIAS ${LIB_NAME})

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

if(BUILD_TESTS)
  set(TEST_NAME okvis_multisensor_processing_test)
  add_executable(${TEST_NAME}
    test/test_main.cpp
  )
  target_link_libraries(${TEST_NAME}
    ${LIB_NAME}
    gmock
    gtest
  )
  target_compile_options(${TEST_NAME}
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endif()
