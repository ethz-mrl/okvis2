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
 * @file implementation/MultiFrame.hpp
 * @brief Header implementation file for the MultiFrame class.
 * @author Stefan Leutenegger
 */

#pragma once

#include <okvis/MultiFrame.hpp>


/// \brief okvis Main namespace of this package.
namespace okvis {

// Default constructor
MultiFrame::MultiFrame() : id_(0) {}

// Construct from NCameraSystem
MultiFrame::MultiFrame(const cameras::NCameraSystem & cameraSystem,
                       const okvis::Time & timestamp, uint64_t id)
    : timestamp_(timestamp),
      id_(id)
{
  resetCameraSystemAndFrames(cameraSystem);
}

MultiFrame::~MultiFrame()
{

}

// (Re)set the NCameraSystem -- which clears the frames as well.
void MultiFrame::resetCameraSystemAndFrames(
    const cameras::NCameraSystem & cameraSystem)
{
  cameraSystem_ = cameraSystem;
  frames_.clear();  // erase -- for safety
  frames_.resize(cameraSystem.numCameras());

  // copy cameras
  for(size_t c = 0; c<numFrames(); ++c){
    frames_[c].setGeometry(cameraSystem.cameraGeometry(c));
  }
}

const cameras::NCameraSystem & MultiFrame::cameraSystem() const {
  return cameraSystem_;
}

// (Re)set the timestamp
void MultiFrame::setTimestamp(const okvis::Time & timestamp)
{
  timestamp_ = timestamp;
}

// (Re)set the id
void MultiFrame::setId(uint64_t id)
{
  id_ = id;
}

// Obtain the frame timestamp
const okvis::Time & MultiFrame::timestamp() const
{
  return timestamp_;
}

// Obtain the frame id
uint64_t MultiFrame::id() const
{
  return id_;
}

// The number of frames/cameras
size_t MultiFrame::numFrames() const
{
  return frames_.size();
}

std::shared_ptr<const okvis::kinematics::Transformation> MultiFrame::T_SC(size_t cameraIdx) const {
  return cameraSystem_.T_SC(cameraIdx);
}

void MultiFrame::setExtrinsics(size_t cameraIdx, const kinematics::Transformation &T_SCi)
{
  cameraSystem_.setExtrinsics(cameraIdx, T_SCi);
}

//////////////////////////////////////////////////////////////
// The following mirror the Frame functionality.
//

// Set the frame image;
void MultiFrame::setImage(size_t cameraIdx, const cv::Mat & image)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setImage(image);
}

// Set the frame image;
void MultiFrame::setDepthImage(size_t cameraIdx, const cv::Mat & depthImage)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setDepthImage(depthImage);
}

// Set the geometry
void MultiFrame::setGeometry(
    size_t cameraIdx, std::shared_ptr<const cameras::CameraBase> cameraGeometry)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setGeometry(cameraGeometry);
}

// Set the detector
void MultiFrame::setDetector(size_t cameraIdx,
                             std::shared_ptr<cv::FeatureDetector> detector)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setDetector(detector);
}

// Set the extractor
void MultiFrame::setExtractor(
    size_t cameraIdx, std::shared_ptr<cv::DescriptorExtractor> extractor)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  frames_[cameraIdx].setExtractor(extractor);
}

// Set the network
void MultiFrame::setNetwork(size_t cameraIdx, std::shared_ptr<Network> network) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  frames_[cameraIdx].setNetwork(network);
}

// Obtain the image
const cv::Mat & MultiFrame::image(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].image();
}

// Obtain the image
const cv::Mat & MultiFrame::depthImage(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].depthImage();
}

// get the base class geometry (will be slow to use)
std::shared_ptr<const cameras::CameraBase> MultiFrame::geometry(
    size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].geometry();
}

// Get the specific geometry (will be fast to use)
template<class GEOMETRY_T>
std::shared_ptr<const GEOMETRY_T> MultiFrame::geometryAs(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].geometryAs<GEOMETRY_T>();
}

// Detect keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        returns the number of detected points.
int MultiFrame::detect(size_t cameraIdx)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].detect();
}

// Describe keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        returns the number of detected points.
int MultiFrame::describe(size_t cameraIdx)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].describe();
}

// Compute the back projections. Caching them should speed up things like repeated
int MultiFrame::computeBackProjections(size_t cameraIdx) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].computeBackProjections();
}

// Get a specific back-projection
bool MultiFrame::getBackProjection(size_t cameraIdx, size_t keypointIdx,
                                   Eigen::Vector3d& backProjection) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getBackProjection(keypointIdx, backProjection);
}

// Remove dynamic points with the CNN.
// \return the number of (remaining) detected points.
inline int MultiFrame::computeClassifications(size_t cameraIdx, int sizeU, int sizeV) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].computeClassifications(sizeU, sizeV);
}

// Access a specific keypoint in OpenCV format
bool MultiFrame::getCvKeypoint(size_t cameraIdx, size_t keypointIdx,
                               cv::KeyPoint & keypoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getCvKeypoint(keypointIdx, keypoint);
}

// Get a specific keypoint
bool MultiFrame::getKeypoint(size_t cameraIdx, size_t keypointIdx,
                             Eigen::Vector2d & keypoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getKeypoint(keypointIdx, keypoint);
}

// Get the size of a specific keypoint
bool MultiFrame::getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                                 double & keypointSize) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getKeypointSize(keypointIdx, keypointSize);
}

bool MultiFrame::getClassification(
    size_t cameraIdx, size_t keypointIdx, cv::Mat & classification) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getClassification(keypointIdx, classification);
}

// Access the descriptor -- CAUTION: high-speed version.
///        returns nullptr if out of bounds.
const unsigned char * MultiFrame::keypointDescriptor(size_t cameraIdx,
                                                     size_t keypointIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].keypointDescriptor(keypointIdx);
}

// Set the landmark ID
bool MultiFrame::setLandmarkId(size_t cameraIdx, size_t keypointIdx,
                               uint64_t landmarkId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].setLandmarkId(keypointIdx, landmarkId);
}

bool MultiFrame::setLandmark(size_t cameraIdx, size_t keypointIdx,
                             const Eigen::Vector4d & landmark, bool isInitialised) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].setLandmark(keypointIdx, landmark, isInitialised);
}

// Access the landmark ID
uint64_t MultiFrame::landmarkId(size_t cameraIdx, size_t keypointIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].landmarkId(keypointIdx);
}

bool MultiFrame::getLandmark(size_t cameraIdx, size_t keypointIdx, Eigen::Vector4d & landmark,
                             bool & isInitialised) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].getLandmark(keypointIdx, landmark, isInitialised);
}

// number of keypoints
size_t MultiFrame::numKeypoints(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].numKeypoints();
}

// provide keypoints externally
bool MultiFrame::resetKeypoints(size_t cameraIdx, const std::vector<cv::KeyPoint> & keypoints){
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].resetKeypoints(keypoints);
}

// provide descriptors externally
bool MultiFrame::resetDescriptors(size_t cameraIdx, const cv::Mat & descriptors) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range")
  return frames_[cameraIdx].resetDescriptors(descriptors);
}

/// release memory of all images
void MultiFrame::clearAllImages() {
  for (auto& f : frames_) {
    f.clearImage();
  }
}

/// release memory of all depth images
void MultiFrame::clearAllDepthImages() {
  for (auto& f : frames_) {
    f.clearDepthImage();
  }
}

// get the total number of keypoints in all frames.
size_t MultiFrame::numKeypoints() const
{
  size_t numKeypoints = 0;
  for (size_t i = 0; i < frames_.size(); ++i) {
    numKeypoints += frames_[i].numKeypoints();
  }
  return numKeypoints;
}


}// namespace okvis
