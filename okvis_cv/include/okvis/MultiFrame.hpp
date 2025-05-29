/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab, ETH Zurich, Smart Robotics Lab,
 *     Imperial College London, Technical University of Munich, nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/

/**
 * @file okvis/MultiFrame.hpp
 * @brief Header file for the MultiFrame class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_MULTIFRAME_HPP_
#define INCLUDE_OKVIS_MULTIFRAME_HPP_

#include <memory>
#include <okvis/assert_macros.hpp>
#include <okvis/Frame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \class MultiFrame
/// \brief A multi camera frame that uses okvis::Frame underneath.
class MultiFrame
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Default constructor
  inline MultiFrame();

  /// \brief Construct from NCameraSystem
  /// @param[in] cameraSystem The camera system for which this is a multi-frame.
  /// @param[in] timestamp The time this frame was recorded.
  /// @param[in] id A unique frame Id.
  inline MultiFrame(const cameras::NCameraSystem & cameraSystem,
                    const okvis::Time & timestamp, uint64_t id = 0);

  /// \brief Destructor...
  inline virtual ~MultiFrame();

  /// \brief (Re)set the NCameraSystem -- which clears the frames as well.
  /// @param[in] cameraSystem The camera system for which this is a multi-frame.
  inline void resetCameraSystemAndFrames(
      const cameras::NCameraSystem & cameraSystem);

  /// \brief (Re)set the timestamp
  /// @param[in] timestamp The time this frame was recorded.
  inline void setTimestamp(const okvis::Time & timestamp);

  /// \brief Set the extrinsics.
  /// @param[in] cameraIdx The camera index that took the image.
  /// @param[in] T_SCi The extrinsics.
  inline void setExtrinsics(size_t cameraIdx, const kinematics::Transformation & T_SCi);

  /// \brief (Re)set the id
  /// @param[in] id A unique frame Id.
  inline void setId(uint64_t id);

  /// \brief Obtain the frame timestamp
  /// \return The time this frame was recorded.
  inline const okvis::Time & timestamp() const;

  /// \brief Obtain the frame id
  /// \return The unique frame Id.
  inline uint64_t id() const;

  /// \brief The number of frames/cameras.
  /// \return How many individual frames/cameras there are.
  inline size_t numFrames() const;

  /// \brief Get the extrinsics of a camera.
  /// @param[in] cameraIdx The camera index for which the extrinsics are queried.
  /// \return The extrinsics as T_SC.
  inline std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
      size_t cameraIdx) const;

  /// \brief Get the N-camera system.
  /// \return Const reference to the N-camera system.
  inline const cameras::NCameraSystem & cameraSystem() const;

  //////////////////////////////////////////////////////////////
  /// \name The following mirror the Frame functionality.
  /// @{

  /// \brief Set the frame image;
  /// @param[in] cameraIdx The camera index that took the image.
  /// @param[in] image The image.
  inline void setImage(size_t cameraIdx, const cv::Mat & image);

  /// \brief Set the frame depth image;
  /// @param[in] cameraIdx The camera index that took the image.
  /// @param[in] depthImage The depth image.
  inline void setDepthImage(size_t cameraIdx, const cv::Mat & depthImage);

  /// \brief Set the geometry
  /// @param[in] cameraIdx The camera index.
  /// @param[in] cameraGeometry The camera geometry.
  inline void setGeometry(
      size_t cameraIdx,
      std::shared_ptr<const cameras::CameraBase> cameraGeometry);

  /// \brief Set the detector
  /// @param[in] cameraIdx The camera index.
  /// @param[in] detector The detector to be used.
  inline void setDetector(size_t cameraIdx,
                          std::shared_ptr<cv::FeatureDetector> detector);

  /// \brief Set the extractor
  /// @param[in] cameraIdx The camera index.
  /// @param[in] extractor The extractor to be used.
  inline void setExtractor(
      size_t cameraIdx,
      std::shared_ptr<cv::DescriptorExtractor> extractor);

  /// \brief Set the network
  /// @param[in] cameraIdx The camera index.
  /// @param[in] network The network to be used.
  inline void setNetwork(size_t cameraIdx, std::shared_ptr<Network> network);

  /// \brief Obtain the image
  /// @param[in] cameraIdx The camera index.
  /// \return The image.
  inline const cv::Mat & image(size_t cameraIdx) const;

  /// \brief Obtain the depth image
  /// @param[in] cameraIdx The camera index.
  /// \return The image.
  inline const cv::Mat & depthImage(size_t cameraIdx) const;

  /// \brief get the base class geometry (will be slow to use)
  /// @param[in] cameraIdx The camera index.
  /// \return The camera geometry.
  inline std::shared_ptr<const cameras::CameraBase> geometry(
      size_t cameraIdx) const;

  /// \brief Get the specific geometry (will be fast to use)
  /// \tparam GEOMETRY_T The type for the camera geometry requested.
  /// @param[in] cameraIdx The camera index.
  /// \return The camera geometry.
  template<class GEOMETRY_T>
  inline std::shared_ptr<const GEOMETRY_T> geometryAs(size_t cameraIdx) const;

  /// \brief Detect keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \return The number of detected points.
  inline int detect(size_t cameraIdx);

  /// \brief Describe keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// @param[in] cameraIdx The camera index.
  /// \return the number of detected points.
  inline int describe(size_t cameraIdx);

  /// \brief Compute the back projections. Caching them should speed up things like repeated
  /// triangulations.
  /// @param[in] cameraIdx The camera index.
  /// \return The number of back projected points.
  inline int computeBackProjections(size_t cameraIdx);

  /// \brief Get a specific back-projection
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] backProjection The corresponding ray direction.
  /// \return whether or not the operation was successful.
  inline bool getBackProjection(size_t cameraIdx, size_t keypointIdx,
                                Eigen::Vector3d& backProjection) const;

  /// \brief Classify keypoints with CNN.
  /// @param[in] cameraIdx The camera index.
  /// @param[in] sizeU Segmentation input size u.
  /// @param[in] sizeV Segmentation input size v.
  /// \return the number of detected points.
  inline int computeClassifications(size_t cameraIdx, int sizeU=192, int sizeV=192);

  /// \brief Access a specific keypoint in OpenCV format
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getCvKeypoint(size_t cameraIdx, size_t keypointIdx,
                            cv::KeyPoint & keypoint) const;

  /// \brief Get a specific keypoint
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getKeypoint(size_t cameraIdx, size_t keypointIdx,
                          Eigen::Vector2d & keypoint) const;

  /// \brief Get the size of a specific keypoint
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypointSize The requested keypoint's size.
  /// \return whether or not the operation was successful.
  inline bool getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                              double & keypointSize) const;

  /// \brief Has classification (segmentation) been run?
  /// @param[in] cameraIdx The camera index.
  /// \return True if classification (segmentation) has been run.
  inline bool isClassified(size_t cameraIdx) const {return frames_[cameraIdx].isClassified();}

  /// \brief Get classification (segmentation) by keypoint.
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] classification The classification (as vector of class probabilties).
  /// \return True on success.
  inline bool getClassification(size_t cameraIdx, size_t keypointIdx,
                                cv::Mat & classification) const;

  /// \brief Access the descriptor -- CAUTION: high-speed version.
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The descriptor data pointer; nullptr if out of bounds.
  inline const unsigned char * keypointDescriptor(size_t cameraIdx,
                                                  size_t keypointIdx) const;

  /// \brief Set the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[in] landmarkId The landmark Id.
  /// \return whether or not the operation was successful.
  inline bool setLandmarkId(size_t cameraIdx, size_t keypointIdx,
                            uint64_t landmarkId);

  /// \brief Set the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[in] landmark The landmark.
  /// @param[in] initialised The landmark initialisation status.
  /// \return whether or not the operation was successful.
  inline bool setLandmark(size_t cameraIdx, size_t keypointIdx, const Eigen::Vector4d & landmark,
                          bool initialised);

  /// \brief Access the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The landmark Id.
  inline uint64_t landmarkId(size_t cameraIdx, size_t keypointIdx) const;

  /// \brief Set the landmark ID
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] landmark The landmark.
  /// @param[out] isInitialised The landmark initialisation status.
  /// \return whether or not the operation was successful.
  inline bool getLandmark(size_t cameraIdx, size_t keypointIdx, Eigen::Vector4d & landmark,
                          bool & isInitialised) const;

  /// \brief provide keypoints externally
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypoints A vector of keyoints.
  /// \return whether or not the operation was successful.
  inline bool resetKeypoints(size_t cameraIdx,
                             const std::vector<cv::KeyPoint> & keypoints);

  /// \brief provide descriptors externally
  /// @param[in] cameraIdx The camera index.
  /// @param[in] descriptors A vector of descriptors.
  /// \return whether or not the operation was successful.
  inline bool resetDescriptors(size_t cameraIdx, const cv::Mat & descriptors);

  /// \brief Release memory of all images
  inline void clearAllImages();

  /// \brief Release memory of all depth images
  inline void clearAllDepthImages();

  /// \brief the number of keypoints
  /// @param[in] cameraIdx The camera index.
  /// \return The number of keypoints.
  inline size_t numKeypoints(size_t cameraIdx) const;

  /// @}

  /// \brief Get the total number of keypoints in all frames.
  /// \return The total number of keypoints.
  inline size_t numKeypoints() const;

  /// \brief Get the overlap mask. Sorry for the weird syntax, but remember that
  /// cv::Mat is essentially a shared pointer.
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return The overlap mask image.
  inline const cv::Mat overlap(size_t cameraIndexSeenBy,
                               size_t cameraIndex) const
  {
    return cameraSystem_.overlap(cameraIndexSeenBy, cameraIndex);
  }

  /// \brief Can the first camera see parts of the FOV of the second camera?
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return True, if there is at least one pixel of overlap.
  inline bool hasOverlap(size_t cameraIndexSeenBy, size_t cameraIndex) const
  {
    return cameraSystem_.hasOverlap(cameraIndexSeenBy, cameraIndex);
  }

 protected:
  okvis::Time timestamp_;  ///< the frame timestamp
  uint64_t id_;  ///< the frame id
  std::vector<Frame, Eigen::aligned_allocator<Frame>> frames_;  ///< the individual frames
  cameras::NCameraSystem cameraSystem_;  ///< the camera system
};

typedef std::shared_ptr<MultiFrame> MultiFramePtr;  ///< For convenience.

}  // namespace okvis

#include "implementation/MultiFrame.hpp"

#endif /* INCLUDE_OKVIS_MULTIFRAME_HPP_ */
