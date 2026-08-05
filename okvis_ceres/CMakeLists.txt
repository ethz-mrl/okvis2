cmake_minimum_required(VERSION 3.8...3.30)

find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED core imgcodecs imgproc)
find_package(OpenMP QUIET)
if(USE_SYSTEM_CERES)
  find_package(Ceres REQUIRED)
endif()

set(LIB_NAME okvis_ceres)
add_library(${LIB_NAME}
  include/okvis/Component.hpp
  include/okvis/PseudoInverse.hpp
  include/okvis/ViGraph.hpp
  include/okvis/ViGraphEstimator.hpp
  include/okvis/ViSlamBackend.hpp
  include/okvis/ceres/CeresIterationCallback.hpp
  include/okvis/ceres/ErrorInterface.hpp
  include/okvis/ceres/HomogeneousPointError.hpp
  include/okvis/ceres/HomogeneousPointLocalParameterization.hpp
  include/okvis/ceres/HomogeneousPointParameterBlock.hpp
  include/okvis/ceres/ImuError.hpp
  include/okvis/ceres/ParameterBlock.hpp
  include/okvis/ceres/ParameterBlockSized.hpp
  include/okvis/ceres/PoseError.hpp
  include/okvis/ceres/PoseLocalParameterization.hpp
  include/okvis/ceres/PoseParameterBlock.hpp
  include/okvis/ceres/RelativePoseError.hpp
  include/okvis/ceres/ReprojectionError.hpp
  include/okvis/ceres/ReprojectionErrorBase.hpp
  include/okvis/ceres/SpeedAndBiasError.hpp
  include/okvis/ceres/SpeedAndBiasParameterBlock.hpp
  include/okvis/ceres/TwoPoseGraphError.hpp
  include/okvis/ceres/TwoPoseExtrinsicsGraphError.hpp
  include/okvis/ceres/implementation/ReprojectionError.hpp
  include/okvis/ceres/ode/ode.hpp
  include/okvis/ceres/DepthError.hpp
  src/CeresIterationCallback.cpp
  src/ErrorInterface.cpp
  src/HomogeneousPointError.cpp
  src/HomogeneousPointLocalParameterization.cpp
  src/HomogeneousPointParameterBlock.cpp
  src/ImuError.cpp
  src/PoseError.cpp
  src/PoseLocalParameterization.cpp
  src/PoseParameterBlock.cpp
  src/RelativePoseError.cpp
  src/SpeedAndBiasError.cpp
  src/SpeedAndBiasParameterBlock.cpp
  src/TwoPoseGraphError.cpp
  src/TwoPoseExtrinsicsGraphError.cpp
  src/ViGraph.cpp
  src/ViGraphEstimator.cpp
  src/ViSlamBackend.cpp
  src/Component.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${GLOG_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  PUBLIC
    ${OpenCV_LIBS}
    Ceres::ceres
    Eigen3::Eigen
    okvis::Common
    okvis::CV
    okvis::Time
    okvis::Timing
    okvis::Util
  PRIVATE
    glog::glog
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)
if(OPENMP_FOUND)
  target_compile_definitions(${LIB_NAME} PRIVATE USE_OPENMP)
endif()
add_library(okvis::Ceres ALIAS ${LIB_NAME})

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

if(BUILD_TESTS)
  set(TEST_NAME okvis_ceres_test)
  add_executable(${TEST_NAME}
    #test/TestPoseGraphError.cpp
    test/TestHomogeneousPointError.cpp
    test/TestImuError.cpp
    test/TestPoseError.cpp
    test/TestReprojectionError.cpp
    test/test_main.cpp
  )
  target_link_libraries(${TEST_NAME}
    ${LIB_NAME}
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
