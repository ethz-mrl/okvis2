cmake_minimum_required(VERSION 3.8...3.30)
project(okvis VERSION 2.0.0 LANGUAGES CXX)

# options for build configuration
option (USE_SYSTEM_BRISK
        "Use brisk via find_package rather than downloading it as part of okvis" OFF)
option (USE_SYSTEM_DBOW2
        "Use DBoW2 via find_package rather than downloading it as part of okvis" OFF)
option (USE_SYSTEM_CERES
        "Use ceres via find_package rather than downloading it as part of okvis" OFF)
option (BUILD_APPS
        "Builds a demo app (which require boost and OpenCV highgui)" ON)
option (BUILD_TESTS
        "Builds all gtests" OFF)
option (BUILD_ROS2
        "Builds ros2 wrapper" ON)
option (HAVE_LIBREALSENSE
        "Use realsense as part of okvis" ON)
option (USE_NN
        "Use keypoint classification as part of okvis, requires torch" ON)
option (USE_GPU
        "Use keypoint classification with GPU inference, requires torch with Cuda" OFF)
option (DO_TIMING
        "Enable timing" ON)
option (BUILD_SHARED_LIBS
        "Whether to build okvis as a shared library" OFF)

set(OKVIS_CXX_STANDARD 17)
set(OKVIS_PUBLIC_CXX_FLAGS)
if(APPLE)
  set(OKVIS_PRIVATE_CXX_FLAGS -Wall -Wpedantic -Wextra -Wno-dtor-name)
else()
  set(OKVIS_PRIVATE_CXX_FLAGS -Wall -Wpedantic -Wextra)
endif()

set(BLA_VENDOR Generic)
find_package(LAPACK REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Make sure we use Release and warn otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
if (NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE not set to 'Release'. Performance may be terrible.")
endif()

if(USE_NN AND USE_GPU)
  if(APPLE)
    # MPS enabled by default
    add_compile_definitions(OKVIS_USE_MPS)
  else()
    # If CUDA isn't enabled explicitly here it's not detected correctly by Torch.
    enable_language(CUDA)
    add_compile_definitions(OKVIS_USE_CUDA)
  endif()
endif()

if(NOT DO_TIMING)
  add_compile_definitions(DEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers")
endif()

if(BUILD_TESTS)
  # Must be called before adding any tests for them to be registered with CMake
  enable_testing()
endif()

add_subdirectory(external)

add_subdirectory(okvis_util)
add_subdirectory(okvis_kinematics)
add_subdirectory(okvis_time)
add_subdirectory(okvis_cv)
add_subdirectory(okvis_common)
add_subdirectory(okvis_timing)
add_subdirectory(okvis_ceres)
add_subdirectory(okvis_frontend)
add_subdirectory(okvis_multisensor_processing)

if(BUILD_APPS)
  find_package(Boost REQUIRED COMPONENTS filesystem)
  find_package(OpenCV REQUIRED core imgcodecs highgui)
  if(USE_SYSTEM_BRISK)
    find_package(brisk REQUIRED)
  endif()
  if(USE_SYSTEM_DBOW2)
    find_package(DBoW2 REQUIRED)
  endif()
  
  # OKVIS app
  add_executable(okvis_app_synchronous okvis_apps/src/okvis_app_synchronous.cpp)
  target_link_libraries(okvis_app_synchronous
    ${OpenCV_LIBS}
    Boost::filesystem
    okvis::MultisensorProcessing
  )
  target_compile_options(okvis_app_synchronous
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  install(TARGETS okvis_app_synchronous EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  # DBoW2 test app
  add_executable(dbow2_test okvis_apps/src/dbow2_test.cpp)
  target_include_directories(dbow2_test
    PRIVATE
      ${DBoW2_INCLUDE_DIRS}
  )
  target_link_libraries(dbow2_test PRIVATE
    ${OpenCV_LIBS}
    Boost::filesystem
    okvis::Frontend
  )
  target_compile_options(dbow2_test
    PUBLIC
      ${OKVIS_PUBLIC_CXX_FLAGS}
    PRIVATE
      ${OKVIS_PRIVATE_CXX_FLAGS}
  )
  
  if(HAVE_LIBREALSENSE)
    # OKIVS RealSense app
    add_executable(okvis_app_realsense okvis_apps/src/okvis_app_realsense.cpp)
    target_link_libraries(okvis_app_realsense
      ${OpenCV_LIBS}
      okvis::MultisensorProcessing
    )
    target_compile_options(okvis_app_realsense
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    install(TARGETS okvis_app_realsense EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  
    # OKIVS RealSense recorder app
    add_executable(okvis_app_realsense_recorder okvis_apps/src/okvis_app_realsense_recorder.cpp)
    target_link_libraries(okvis_app_realsense_recorder
      ${OpenCV_LIBS}
      okvis::MultisensorProcessing
    )
    target_compile_options(okvis_app_realsense_recorder
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
    install(TARGETS okvis_app_realsense_recorder EXPORT okvisTargets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  
  if(USE_NN)
    find_package(Torch REQUIRED)
  
    # Torch test app
    add_executable(nn_test okvis_apps/src/nn_test.cpp)
    target_compile_options(nn_test PRIVATE ${TORCH_CXX_FLAGS})
    target_include_directories(nn_test
      PRIVATE
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(nn_test
      ${OpenCV_LIBS}
      ${TORCH_LIBRARIES}
    )
    target_compile_features(nn_test PUBLIC cxx_std_${OKVIS_CXX_STANDARD})
    target_compile_options(nn_test
      PUBLIC
        ${OKVIS_PUBLIC_CXX_FLAGS}
      PRIVATE
        ${OKVIS_PRIVATE_CXX_FLAGS}
    )
  endif()
endif()

# ros2
if(BUILD_ROS2)
  if (NOT "$ENV{ROS_VERSION}" STREQUAL "2")
    MESSAGE(WARNING "ROS2 not found, will build without")
    set(BUILD_ROS2 OFF)
  endif()
endif()
if(BUILD_ROS2)
  #add_subdirectory(okvis_ros2)
  include_directories(okvis_ros2/include/)
  MESSAGE(STATUS "Building ROS2 wrapper")

  #LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  # ROS2 DEPENDENCIES
  FIND_PACKAGE(ament_cmake REQUIRED)
  FIND_PACKAGE(ament_cmake_python REQUIRED)
  FIND_PACKAGE(builtin_interfaces REQUIRED)
  FIND_PACKAGE(eigen3_cmake_module REQUIRED)
  FIND_PACKAGE(Eigen3 REQUIRED)
  FIND_PACKAGE(OpenCV REQUIRED)
  FIND_PACKAGE(cv_bridge REQUIRED)  
  FIND_PACKAGE(sensor_msgs REQUIRED)
  FIND_PACKAGE(visualization_msgs REQUIRED)
  FIND_PACKAGE(nav_msgs REQUIRED)
  find_package(Boost REQUIRED)

  find_package(PCL REQUIRED QUIET COMPONENTS common)
  include_directories(${PCL_INCLUDE_DIRS} include)
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  find_package(pcl_msgs REQUIRED)
  FIND_PACKAGE(pcl_conversions REQUIRED)
  FIND_PACKAGE(rclpy REQUIRED)
  FIND_PACKAGE(rclcpp REQUIRED)
  FIND_PACKAGE(rclcpp_components REQUIRED)
  find_package(rosbag2_cpp REQUIRED)
  find_package(rosbag2_storage REQUIRED)
  FIND_PACKAGE(tf2 REQUIRED)
  FIND_PACKAGE(tf2_ros REQUIRED)
  FIND_PACKAGE(image_transport REQUIRED)
  FIND_PACKAGE(message_filters REQUIRED)
  FIND_PACKAGE(cv_bridge REQUIRED)

  # INCLUDES AND LIB DIRECTORIES
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${Eigen3_INCLUDE_DIRS})
  include_directories( include ${PCL_INCLUDE_DIRS} )

  # okvis::Ros2 library
  set(LIB_NAME okvis_ros2)
  add_library(${LIB_NAME}
    okvis_ros2/src/Subscriber.cpp
    okvis_ros2/src/RosbagReader.cpp
    okvis_ros2/src/Publisher.cpp
    okvis_ros2/src/RePublisher.cpp
  )
  target_include_directories(${LIB_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${GLOG_INCLUDE_DIRS}
  )
  target_link_libraries(${LIB_NAME}
      ${OpenCV_LIBS}
      Ceres::ceres
      Eigen3::Eigen
      okvis::Common
      okvis::MultisensorProcessing
      glog::glog
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
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
  add_library(okvis::Ros2 ALIAS ${LIB_NAME})

  # TARGET DEPENDENCIES
  AMENT_TARGET_DEPENDENCIES(
    ${LIB_NAME}
    PCL
    ament_cmake
    rclcpp
    rclcpp_components
    rosbag2_cpp
    rosbag2_storage
    rosidl_default_runtime
    eigen3_cmake_module
    Eigen3
    sensor_msgs
    visualization_msgs
    nav_msgs
    pcl_conversions
    message_filters
    cv_bridge
    image_transport
    tf2
    tf2_ros
  )

  # EXPORT DEPENDENCIES
  AMENT_EXPORT_DEPENDENCIES(
    PCL
    ament_cmake
    rclcpp
    rclcpp_components
    rosbag2_cpp
    rosbag2_storage
    rosidl_default_runtime
    eigen3_cmake_module
    Eigen3
    sensor_msgs
    visualization_msgs
    nav_msgs
    pcl_conversions
    message_filters
    cv_bridge
    image_transport
    tf2
    tf2_ros
  )

  install(TARGETS ${LIB_NAME} EXPORT okvisTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(DIRECTORY include/okvis DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")

  # OKVIS nodes
  
  ADD_EXECUTABLE(okvis_node_synchronous okvis_ros2/src/okvis_node_synchronous.cpp)
  TARGET_LINK_LIBRARIES(
    okvis_node_synchronous
    okvis::Ros2
    ${OpenCV_LIBS}
    ${SUITESPARSE_LIBRARIES}
    ${Boost_LIBRARIES} 
    ${PCL_LIBRARIES}
  )
  INSTALL(TARGETS okvis_node_synchronous DESTINATION lib/${PROJECT_NAME})
  
  ADD_EXECUTABLE(okvis_node_subscriber okvis_ros2/src/okvis_node_subscriber.cpp)
  TARGET_LINK_LIBRARIES(
    okvis_node_subscriber
    okvis::Ros2
    ${OpenCV_LIBS}
    ${SUITESPARSE_LIBRARIES}
    ${Boost_LIBRARIES} 
    ${PCL_LIBRARIES}
  )
  INSTALL(TARGETS okvis_node_subscriber DESTINATION lib/${PROJECT_NAME})
  
  if(HAVE_LIBREALSENSE)
    ADD_EXECUTABLE(okvis_node_realsense okvis_ros2/src/okvis_node_realsense.cpp)
    TARGET_LINK_LIBRARIES(
      okvis_node_realsense
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
    )
    INSTALL(TARGETS okvis_node_realsense DESTINATION lib/${PROJECT_NAME})
    
    ADD_EXECUTABLE(okvis_node_realsense_publisher okvis_ros2/src/okvis_node_realsense_publisher.cpp)
    TARGET_LINK_LIBRARIES(
      okvis_node_realsense_publisher
      okvis::Ros2
      ${OpenCV_LIBS}
      ${SUITESPARSE_LIBRARIES}
      ${Boost_LIBRARIES} 
      ${PCL_LIBRARIES}
    )
    INSTALL(TARGETS okvis_node_realsense_publisher DESTINATION lib/${PROJECT_NAME})
  endif()

  # INSTALL LAUNCH FILES
    INSTALL(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
    INSTALL(DIRECTORY resources DESTINATION share/${PROJECT_NAME}/)
    INSTALL(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

  # FORM ROS2 Package
  AMENT_PACKAGE()

endif(BUILD_ROS2)

# The directory where the .cmake files will be installed
include(GNUInstallDirs)
if(WIN32 AND NOT CYGWIN)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}/cmake")
elseif(APPLE)
    set(INSTALL_CMAKEDIR "${PACKAGE_NAME}.framework/Resources/CMake")
else()
    set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}")
endif()
# Generate okvisConfig.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/okvisConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKEDIR})
# Generate okvisConfigVersion.cmake
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/okvisConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(EXPORT okvisTargets
  FILE okvisTargets.cmake
  NAMESPACE okvis::
  DESTINATION ${INSTALL_CMAKEDIR}/okvis)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/okvisConfigVersion.cmake
  DESTINATION ${INSTALL_CMAKEDIR}/okvis)
