cmake_minimum_required(VERSION 3.8...3.30)

find_package(Eigen3 REQUIRED)
# OpenCV 5 renamed the features2d module to features; support both.
find_package(OpenCV REQUIRED QUIET COMPONENTS core)
if(OpenCV_VERSION VERSION_LESS 5.0)
  set(OpenCV_FEATURES_COMPONENT features2d)
else()
  set(OpenCV_FEATURES_COMPONENT features)
endif()
find_package(OpenCV REQUIRED COMPONENTS core ${OpenCV_FEATURES_COMPONENT} imgproc)
if(USE_SYSTEM_BRISK)
  find_package(brisk REQUIRED)
endif()
if(USE_NN)
  find_package(Torch REQUIRED)
else()
  message(WARNING "Network keypoint classification deactivated")
endif()

set(LIB_NAME okvis_cv)
add_library(${LIB_NAME}
  include/okvis/Frame.hpp
  include/okvis/MultiFrame.hpp
  include/okvis/implementation/Frame.hpp
  include/okvis/implementation/MultiFrame.hpp
  include/okvis/cameras/CameraBase.hpp
  include/okvis/cameras/DistortionBase.hpp
  include/okvis/cameras/EquidistantDistortion.hpp
  include/okvis/cameras/NCameraSystem.hpp
  include/okvis/cameras/NoDistortion.hpp
  include/okvis/cameras/PinholeCamera.hpp
  include/okvis/cameras/RadialTangentialDistortion.hpp
  include/okvis/cameras/RadialTangentialDistortion8.hpp
  include/okvis/cameras/implementation/CameraBase.hpp
  include/okvis/cameras/implementation/EquidistantDistortion.hpp
  include/okvis/cameras/implementation/NCameraSystem.hpp
  include/okvis/cameras/implementation/PinholeCamera.hpp
  include/okvis/cameras/implementation/RadialTangentialDistortion.hpp
  include/okvis/cameras/implementation/RadialTangentialDistortion8.hpp
  include/okvis/internal/Network.hpp
  src/CameraBase.cpp
  src/Frame.cpp
  src/NCameraSystem.cpp
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(${LIB_NAME}
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    okvis::Kinematics
    okvis::Time
    okvis::Util
  PRIVATE
    brisk::brisk
    ${TORCH_LIBRARIES}
)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
target_compile_options(${LIB_NAME}
  PUBLIC
    ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE
    ${OKVIS_PRIVATE_CXX_FLAGS}
)
if(USE_NN)
  target_compile_options(${LIB_NAME} PUBLIC ${TORCH_CXX_FLAGS})
  target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_NN C10_USE_GLOG)
  if(USE_GPU)
    target_compile_definitions(${LIB_NAME} PUBLIC OKVIS_USE_GPU)
  endif()
endif()
add_library(okvis::CV ALIAS ${LIB_NAME})

install(TARGETS ${LIB_NAME} EXPORT okvisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

if(BUILD_TESTS)
  set(TEST_NAME okvis_cv_test)
  add_executable(${TEST_NAME}
    test/TestFrame.cpp
    test/TestMultiFrame.cpp
    test/TestNCameraSystem.cpp
    test/TestPinholeCamera.cpp
    test/runTests.cpp
  )
  target_include_directories(${TEST_NAME}
    PRIVATE
      ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(${TEST_NAME}
    brisk::brisk
    ${LIB_NAME}
    ${TORCH_LIBRARIES}
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
