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
 * @file implementation/Frame.hpp
 * @brief Header implementation file for the Frame class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifdef OKVIS_USE_NN
#include <opencv2/imgproc.hpp>
#endif

#pragma once

#include <okvis/Frame.hpp>


/// \brief okvis Main namespace of this package.
namespace okvis {

// a constructor that uses the specified geometry,
/// detector and extractor
Frame::Frame(const cv::Mat & image,
             std::shared_ptr<cameras::CameraBase> & cameraGeometry,
             std::shared_ptr<cv::FeatureDetector> & detector,
             std::shared_ptr<cv::DescriptorExtractor> & extractor,
             std::shared_ptr<Network> & network)
    : image_(image),
      cameraGeometry_(cameraGeometry),
      detector_(detector),
      extractor_(extractor),
      network_(network),
      isClassified_(false)
{
}

// set the frame image;
void Frame::setImage(const cv::Mat & image)
{
  image_ = image;
}

// set the frame image;
void Frame::setDepthImage(const cv::Mat & depthImage)
{
  depthImage_ = depthImage;
}

// set the geometry
void Frame::setGeometry(std::shared_ptr<const cameras::CameraBase> cameraGeometry)
{
  cameraGeometry_ = cameraGeometry;
}

// set the detector
void Frame::setDetector(std::shared_ptr<cv::FeatureDetector> detector)
{
  detector_ = detector;
}

// set the extractor
void Frame::setExtractor(std::shared_ptr<cv::DescriptorExtractor> extractor)
{
  extractor_ = extractor;
}

// set the network
void Frame::setNetwork(std::shared_ptr<Network> network) {
  network_ = network;
}


// obtain the image
const cv::Mat & Frame::image() const
{
  return image_;
}

// obtain the depth image
const cv::Mat & Frame::depthImage() const
{
  return depthImage_;
}

// get the base class geometry (will be slow to use)
std::shared_ptr<const cameras::CameraBase> Frame::geometry() const
{
  return cameraGeometry_;
}

// get the specific geometry (will be fast to use)
template<class GEOMETRY_T>
std::shared_ptr<const GEOMETRY_T> Frame::geometryAs() const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception, std::dynamic_pointer_cast<const GEOMETRY_T>(cameraGeometry_),
      "incorrect pointer cast requested. " << cameraGeometry_->distortionType())
  return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#else
  return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#endif
}

// detect keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        returns the number of detected points.
int Frame::detect()
{
  // make sure things are set to zero for safety
  keypoints_.clear();
  descriptors_.resize(0);

  // resizing and filling in zeros in Frame::describe() as some keypoints are removed there:
  landmarkIds_.clear();

  // run the detector
  OKVIS_ASSERT_TRUE_DBG(Exception, detector_ != nullptr,
                        "Detector not initialised!")
  detector_->detect(image_, keypoints_);
  return int(keypoints_.size());
}

// describe keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        extractionDirection the extraction direction in camera frame
///        returns the number of detected points.
int Frame::describe()
{
  // check initialisation
  OKVIS_ASSERT_TRUE_DBG(Exception, extractor_ != nullptr,
                        "Detector not initialised!")

  // extraction
  extractor_->compute(image_, keypoints_, descriptors_);

  // resizing
  landmarkIds_ = std::vector<uint64_t>(keypoints_.size(),0);
  landmarks_ = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>(
        keypoints_.size(), Eigen::Vector4d(0,0,0,0));
  landmarkInitialisations_ = std::vector<bool>(keypoints_.size(), false);
  return int(keypoints_.size());
}

// Compute the back projections. Caching them should speed up things like repeated
int Frame::computeBackProjections() {
  backProjections_.resize(keypoints_.size());
  backProjectionsValid_.resize(keypoints_.size());
  int ctr=0;
  for (size_t k = 0; k < keypoints_.size(); ++k) {
    cv::KeyPoint& ckp = keypoints_[k];
    // project ray
    const bool success = cameraGeometry_->backProject(
          Eigen::Vector2d(ckp.pt.x, ckp.pt.y), &backProjections_[k]);
    backProjectionsValid_[k] = success;
    if(success) {
      ctr++;
    }
  }
  return ctr;
}

// Get a specific back projection
bool Frame::getBackProjection(size_t keypointIdx, Eigen::Vector3d& backProjection) const {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  backProjection = backProjections_[keypointIdx];
  return backProjectionsValid_[keypointIdx];
#else
  backProjection = backProjections_[keypointIdx];
  return backProjectionsValid_[keypointIdx];
#endif
}

bool Frame::getClassification(size_t keypointIdx, cv::Mat & classification) const {
  if(!isClassified_) {
    return false; // not computed
  }
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  cv::Mat roi(classifications_, cv::Rect(0,int(keypointIdx),19,1));
  roi.copyTo(classification);
  return true;
#else
  cv::Mat roi(classifications_, cv::Rect(0,keypointIdx,19,1));
  roi.copyTo(classification);
  return true;
#endif
}

// access a specific keypoint in OpenCV format
bool Frame::getCvKeypoint(size_t keypointIdx, cv::KeyPoint & keypoint) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  keypoint = keypoints_[keypointIdx];
  return keypointIdx < keypoints_.size();
#else
  keypoint = keypoints_[keypointIdx];
  return true;
#endif
}

// get a specific keypoint
bool Frame::getKeypoint(size_t keypointIdx, Eigen::Vector2d & keypoint) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x,
                             keypoints_[keypointIdx].pt.y);
  return keypointIdx < keypoints_.size();
#else
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x, keypoints_[keypointIdx].pt.y);
  return true;
#endif
}

// get the size of a specific keypoint
bool Frame::getKeypointSize(size_t keypointIdx, double & keypointSize) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size())
  keypointSize = double(keypoints_[keypointIdx].size);
  return keypointIdx < keypoints_.size();
#else
  keypointSize = double(keypoints_[keypointIdx].size);
  return true;
#endif
}

// access the descriptor -- CAUTION: high-speed version.
///        returns nullptr if out of bounds.
const unsigned char * Frame::keypointDescriptor(size_t keypointIdx) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size "<< keypoints_.size())
  return descriptors_.data + size_t(descriptors_.cols) * keypointIdx;
#else
  return descriptors_.data + size_t(descriptors_.cols) * keypointIdx;
#endif
}

// Set the landmark ID
bool Frame::setLandmarkId(size_t keypointIdx, uint64_t landmarkId)
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarkIds_.size(),
      "keypointIdx " << keypointIdx
        << "out of range: landmarkIds_ has size "<< landmarkIds_.size())
  landmarkIds_[keypointIdx] = landmarkId;
  return keypointIdx < keypoints_.size();
#else
  landmarkIds_[keypointIdx] = landmarkId;
  return true;
#endif
}

// Set the landmark
bool Frame::setLandmark(size_t keypointIdx, const Eigen::Vector4d & landmark, bool isInitialised)
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarks_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarks_ has size " << landmarks_.size())
  landmarks_[keypointIdx] = landmark;
  landmarkInitialisations_[keypointIdx] = isInitialised;
  return keypointIdx < landmarks_.size();
#else
  landmarks_[keypointIdx] = landmark;
  landmarkInitialisations_[keypointIdx] = isInitialised;
  return true;
#endif
}

// Access the landmark ID
uint64_t Frame::landmarkId(size_t keypointIdx) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarkIds_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarkIds has size " << landmarkIds_.size())
  return landmarkIds_[keypointIdx];
#else
  return landmarkIds_[keypointIdx];
#endif
}

// Access the landmark
bool Frame::getLandmark(size_t keypointIdx, Eigen::Vector4d & landmark, bool & isInitialised) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < landmarks_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarks has size " << landmarks_.size())
  landmark = landmarks_[keypointIdx];
  isInitialised = landmarkInitialisations_[keypointIdx];
  return keypointIdx < landmarks_.size();
#else
  landmark = landmarks_[keypointIdx];
  isInitialised = landmarkInitialisations_[keypointIdx];
  return true;
#endif
}

// provide keypoints externally
inline bool Frame::resetKeypoints(const std::vector<cv::KeyPoint> & keypoints) {
  keypoints_ = keypoints;

  // resizing
  landmarkIds_ = std::vector<uint64_t>(keypoints_.size(),0);
  landmarks_ = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>(
    keypoints_.size(), Eigen::Vector4d(0,0,0,0));
  landmarkInitialisations_ = std::vector<bool>(keypoints_.size(), false);

  return true;
}

// provide descriptors externally
inline bool Frame::resetDescriptors(const cv::Mat & descriptors) {
  descriptors_ = descriptors;
  return true;
}

/// release memory of image
void Frame::clearImage() {
  if (!image_.empty()) {
    image_ = cv::Mat();
  }
}

/// release memory of depth image
void Frame::clearDepthImage() {
  if (!depthImage_.empty()) {
    depthImage_ = cv::Mat();
  }
}

size_t Frame::numKeypoints() const {
  return keypoints_.size();
}

}  // namespace okvis
