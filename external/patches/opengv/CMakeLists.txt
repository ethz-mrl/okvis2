cmake_minimum_required(VERSION 3.8...3.16)
project(opengv VERSION 1.0 LANGUAGES CXX)

# okvis2: remove hardcoded options to inherit them from okvis2

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/modules/")
find_package(Eigen REQUIRED)
set(ADDITIONAL_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR}/unsupported)

set( OPENGV_SOURCE_FILES
  src/absolute_pose/modules/main.cpp
  src/absolute_pose/modules/gp3p/code.cpp
  src/absolute_pose/modules/gp3p/init.cpp
  src/absolute_pose/modules/gp3p/reductors.cpp
  src/absolute_pose/modules/gp3p/spolynomials.cpp
  src/absolute_pose/modules/Epnp.cpp
  src/absolute_pose/modules/gpnp1/code.cpp
  src/absolute_pose/modules/gpnp1/init.cpp
  src/absolute_pose/modules/gpnp1/reductors.cpp
  src/absolute_pose/modules/gpnp1/spolynomials.cpp
  src/absolute_pose/modules/gpnp2/code.cpp
  src/absolute_pose/modules/gpnp2/init.cpp
  src/absolute_pose/modules/gpnp2/reductors.cpp
  src/absolute_pose/modules/gpnp2/spolynomials.cpp
  src/absolute_pose/modules/gpnp3/code.cpp
  src/absolute_pose/modules/gpnp3/init.cpp
  src/absolute_pose/modules/gpnp3/reductors.cpp
  src/absolute_pose/modules/gpnp3/spolynomials.cpp
  src/absolute_pose/modules/gpnp4/code.cpp
  src/absolute_pose/modules/gpnp4/init.cpp
  src/absolute_pose/modules/gpnp4/reductors.cpp
  src/absolute_pose/modules/gpnp4/spolynomials.cpp
  src/absolute_pose/modules/gpnp5/code.cpp
  src/absolute_pose/modules/gpnp5/init.cpp
  src/absolute_pose/modules/gpnp5/reductors.cpp
  src/absolute_pose/modules/gpnp5/spolynomials.cpp
  src/absolute_pose/modules/upnp2.cpp
  src/absolute_pose/modules/upnp4.cpp
  src/relative_pose/modules/main.cpp
  src/relative_pose/modules/fivept_nister/modules.cpp
  src/relative_pose/modules/fivept_stewenius/modules.cpp
  src/relative_pose/modules/fivept_kneip/code.cpp
  src/relative_pose/modules/fivept_kneip/init.cpp
  src/relative_pose/modules/fivept_kneip/reductors.cpp
  src/relative_pose/modules/fivept_kneip/spolynomials.cpp
  src/relative_pose/modules/sixpt/modules2.cpp
  src/relative_pose/modules/eigensolver/modules.cpp
  src/relative_pose/modules/ge/modules.cpp
  src/math/cayley.cpp
  src/math/quaternion.cpp
  src/math/arun.cpp
  src/math/Sturm.cpp
  src/math/roots.cpp
  src/math/gauss_jordan.cpp
  src/absolute_pose/methods.cpp
  src/absolute_pose/CentralAbsoluteAdapter.cpp
  src/absolute_pose/NoncentralAbsoluteAdapter.cpp
  src/absolute_pose/NoncentralAbsoluteMultiAdapter.cpp
  src/relative_pose/methods.cpp
  src/relative_pose/CentralRelativeAdapter.cpp
  src/relative_pose/CentralRelativeWeightingAdapter.cpp
  src/relative_pose/NoncentralRelativeAdapter.cpp
  src/relative_pose/CentralRelativeMultiAdapter.cpp
  src/relative_pose/NoncentralRelativeMultiAdapter.cpp
  src/triangulation/methods.cpp
  src/point_cloud/methods.cpp
  src/point_cloud/PointCloudAdapter.cpp
  src/sac_problems/absolute_pose/AbsolutePoseSacProblem.cpp
  src/sac_problems/absolute_pose/MultiNoncentralAbsolutePoseSacProblem.cpp
  src/sac_problems/relative_pose/CentralRelativePoseSacProblem.cpp
  src/sac_problems/relative_pose/NoncentralRelativePoseSacProblem.cpp
  src/sac_problems/relative_pose/RotationOnlySacProblem.cpp
  src/sac_problems/relative_pose/TranslationOnlySacProblem.cpp
  src/sac_problems/relative_pose/EigensolverSacProblem.cpp
  src/sac_problems/relative_pose/MultiCentralRelativePoseSacProblem.cpp
  src/sac_problems/relative_pose/MultiNoncentralRelativePoseSacProblem.cpp
  src/sac_problems/point_cloud/PointCloudSacProblem.cpp
  src/absolute_pose/MACentralAbsolute.cpp
  src/absolute_pose/MANoncentralAbsolute.cpp
  src/relative_pose/MACentralRelative.cpp
  src/relative_pose/MANoncentralRelative.cpp
  src/relative_pose/MANoncentralRelativeMulti.cpp
  src/point_cloud/MAPointCloud.cpp )

set( OPENGV_HEADER_FILES
  include/opengv/types.hpp
  include/opengv/OptimizationFunctor.hpp
  include/opengv/absolute_pose/methods.hpp
  include/opengv/relative_pose/methods.hpp
  include/opengv/triangulation/methods.hpp
  include/opengv/point_cloud/methods.hpp
  include/opengv/math/cayley.hpp
  include/opengv/math/quaternion.hpp
  include/opengv/math/arun.hpp
  include/opengv/math/Sturm.hpp
  include/opengv/math/roots.hpp
  include/opengv/math/gauss_jordan.hpp
  include/opengv/absolute_pose/AbsoluteAdapterBase.hpp
  include/opengv/absolute_pose/CentralAbsoluteAdapter.hpp
  include/opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp
  include/opengv/absolute_pose/NoncentralAbsoluteMultiAdapter.hpp
  include/opengv/absolute_pose/AbsoluteMultiAdapterBase.hpp
  include/opengv/relative_pose/RelativeAdapterBase.hpp
  include/opengv/relative_pose/RelativeMultiAdapterBase.hpp
  include/opengv/relative_pose/CentralRelativeAdapter.hpp
  include/opengv/relative_pose/CentralRelativeWeightingAdapter.hpp
  include/opengv/relative_pose/NoncentralRelativeAdapter.hpp
  include/opengv/relative_pose/CentralRelativeMultiAdapter.hpp
  include/opengv/relative_pose/NoncentralRelativeMultiAdapter.hpp
  include/opengv/point_cloud/PointCloudAdapterBase.hpp
  include/opengv/point_cloud/PointCloudAdapter.hpp
  include/opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp
  include/opengv/sac_problems/absolute_pose/MultiNoncentralAbsolutePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/NoncentralRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/MultiCentralRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/MultiNoncentralRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp
  include/opengv/sac_problems/relative_pose/RotationOnlySacProblem.hpp
  include/opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp
  include/opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp
  include/opengv/absolute_pose/MACentralAbsolute.hpp
  include/opengv/absolute_pose/MANoncentralAbsolute.hpp
  include/opengv/relative_pose/MACentralRelative.hpp
  include/opengv/relative_pose/MANoncentralRelative.hpp
  include/opengv/relative_pose/MANoncentralRelativeMulti.hpp
  include/opengv/point_cloud/MAPointCloud.hpp )

add_library( opengv ${OPENGV_SOURCE_FILES} ${OPENGV_HEADER_FILES} )
# okvis2: remove library needed only for tests
set_target_properties( opengv PROPERTIES
                    SOVERSION ${PROJECT_VERSION}
                    VERSION ${PROJECT_VERSION}
                    CXX_STANDARD 14
                    CXX_STANDARD_REQUIRED ON)
                    # okvis2: remove DEBUG_POSTFIX to easily build in Debug

# Hack: Eigen 5 doesn't have assert() ...
# Target-specific forced inclusion based on the compiler:
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(opengv PRIVATE "/FI${CMAKE_CURRENT_SOURCE_DIR}/../patches/opengv/eigen_assert_override.h")
else()
    # For GCC, Clang, and AppleClang
    target_compile_options(opengv PRIVATE "-include" "${CMAKE_CURRENT_SOURCE_DIR}/../patches/opengv/eigen_assert_override.h")
endif()

target_include_directories( opengv PUBLIC 
                # only when building from the source tree
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # okvis2: make include directory more robust
                # only when using the lib from the install path
                $<INSTALL_INTERFACE:include>
                ${ADDITIONAL_INCLUDE_DIRS} )

# okvis2: remove tests
# Configuration
set(config_install_dir "lib/cmake/${PROJECT_NAME}-${PROJECT_VERSION}")
set(include_install_dir "include")
set(version_config "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Note: PROJECT_VERSION is used as a VERSION
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion )

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * targets_export_name
#   * PROJECT_NAME
configure_package_config_file(
    "modules/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}")

# Export '<PROJECT-NAME>Targets.cmake' to build dir (to find package in build dir without install)
export(TARGETS opengv FILE "${CMAKE_CURRENT_BINARY_DIR}/${targets_export_name}.cmake")

# Targets:
install(TARGETS opengv
        EXPORT "${targets_export_name}"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        INCLUDES DESTINATION "${include_install_dir}")

# Config
#   * <prefix>/lib/cmake/opengv/opengvConfig.cmake
#   * <prefix>/lib/cmake/opengv/opengvConfigVersion.cmake
install(FILES "${project_config}" "${version_config}"
        DESTINATION "${config_install_dir}")

# Config
#   * <prefix>/lib/cmake/opengv/opengvTargets.cmake
install(EXPORT "${targets_export_name}"
        NAMESPACE "${namespace}"
        DESTINATION "${config_install_dir}")

# Headers
install(DIRECTORY include/ 
        DESTINATION ${include_install_dir} 
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
