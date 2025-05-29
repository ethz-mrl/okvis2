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
 * @file ViSlamBackend.cpp
 * @brief Source file for the Estimator class. This does all the backend work.
 * @author Stefan Leutenegger
 */

#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <okvis/ViSlamBackend.hpp>
#include <okvis/PseudoInverse.hpp>
#include <okvis/Component.hpp>
#include <okvis/timing/Timer.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

static const double minDeltaT = 2.0; // [sec]
static int numRealtimePoseGraphFrames = 12;
static const int numPoseGraphFrames = 12;

int ViSlamBackend::addCamera(const CameraParameters &cameraParameters)
{
  fullGraph_.addCamera(cameraParameters);
  return realtimeGraph_.addCamera(cameraParameters);
}

int ViSlamBackend::addImu(const ImuParameters &imuParameters)
{
  fullGraph_.addImu(imuParameters);
  return realtimeGraph_.addImu(imuParameters);
}

bool ViSlamBackend::addStates(MultiFramePtr multiFrame, const ImuMeasurementDeque &imuMeasurements,
                              bool asKeyframe)
{
  AuxiliaryState auxiliaryState;
  auxiliaryState.isImuFrame = true;
  auxiliaryState.isKeyframe = asKeyframe;
  if(multiFrames_.empty()) {
    // initialise
    OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "not allowed")
    const StateId id = realtimeGraph_.addStatesInitialise(multiFrame->timestamp(), imuMeasurements,
                                                          multiFrame->cameraSystem());
    multiFrame->setId(id.value());
    fullGraph_.addStatesInitialise(multiFrame->timestamp(), imuMeasurements,
                                              multiFrame->cameraSystem());
    multiFrames_[id] = multiFrame;
    auxiliaryState.loopId = id;
    auxiliaryStates_[id] = auxiliaryState; // for internal book-keeping
    imuFrames_.insert(id);
    return (id.value()==1 && id.isInitialised());
  } else {
    const StateId id = realtimeGraph_.addStatesPropagate(multiFrame->timestamp(), imuMeasurements,
                                                         asKeyframe);
    // obtain potentially optimised extrinsics
    for (size_t i = 0; i < multiFrame->numFrames(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        const kinematics::Transformation T_SCi = extrinsics(id, i);
        multiFrame->setExtrinsics(i, T_SCi);
      }
    }

    multiFrame->setId(id.value());

    if(isLoopClosing_ || isLoopClosureAvailable_) {
      addStatesBacklog_.push_back(AddStatesBacklog{multiFrame->timestamp(), id, imuMeasurements});
      touchedStates_.insert(id);
    } else {
      // safe to add to the full graph
      fullGraph_.addStatesPropagate(multiFrame->timestamp(), imuMeasurements, asKeyframe);
    }

    multiFrames_[id] = multiFrame;
    auxiliaryState.loopId = id;
    auxiliaryStates_[id] = auxiliaryState; // for internal book-keeping
    imuFrames_.insert(id);
    return id.isInitialised();
  }
}

void ViSlamBackend::printStates(StateId stateId, std::ostream &buffer) const
{
  const ViGraph::State & state = realtimeGraph_.states_.at(stateId);
  if(state.isKeyframe) {
    buffer << "KF ";
  }
  buffer << "pose: ";
  if (state.pose->fixed()) buffer << "(";
  buffer << "id=" << stateId.value() << ":";
  if (state.pose->fixed()) buffer << ")";
  buffer << ", ";
  buffer << "speedAndBias: ";
  if (state.speedAndBias->fixed()) buffer << "(";
  buffer << "id=" << state.speedAndBias->id() << ":";
  if (state.speedAndBias->fixed()) buffer << ")";
  buffer << ", ";
  buffer << "extrinsics: ";
  for (size_t i = 0; i < state.extrinsics.size(); ++i) {
    uint64_t id = state.extrinsics.at(i)->id();
    if (state.extrinsics.at(i)->fixed()) buffer << "(";
    buffer << "id=" << id << ":";
    if (state.extrinsics.at(i)->fixed()) buffer << ")";
    if (i+1 < state.extrinsics.size()) buffer << ", ";
  }
  buffer << std::endl;
}

bool ViSlamBackend::getLandmark(LandmarkId landmarkId, okvis::MapPoint2& mapPoint) const
{
  return realtimeGraph_.getLandmark(landmarkId, mapPoint);
}


size_t ViSlamBackend::getLandmarks(MapPoints & landmarks) const
{
  return realtimeGraph_.getLandmarks(landmarks);
}

double ViSlamBackend::trackingQuality(StateId id) const
{
  const MultiFramePtr frame = multiFrame(id);
  const size_t numFrames = frame->numFrames();
  std::set<LandmarkId> landmarks;
  std::vector<cv::Mat> matchesImg(numFrames);

  // remember matched points
  int intersectionCount = 0;
  int unionCount = 0;
  int matchedPoints = 0;
  for (size_t im = 0; im < numFrames; ++im) {
    const int rows = frame->image(im).rows/10;
    const int cols = frame->image(im).cols/10;
    const double radius = double(std::min(rows,cols))*kptradius_;
    matchesImg.at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
    const size_t num = frame->numKeypoints(im);
    cv::KeyPoint keypoint;
    for (size_t k = 0; k < num; ++k) {
      frame->getCvKeypoint(im, k, keypoint);
      uint64_t lmId = frame->landmarkId(im, k);
      if (lmId != 0 && realtimeGraph_.landmarkExists(LandmarkId(lmId))) {
        // make sure these are observed elsewhere
        for(const auto & obs : realtimeGraph_.landmarks_.at(LandmarkId(lmId)).observations) {
          if(obs.first.frameId != id.value()) {
            matchedPoints++;
            cv::circle(matchesImg.at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                       cv::FILLED);
            break;
          }
        }
      }
    }
    // one point per image does not count.
    const int pointArea = int(radius*radius*M_PI);
    intersectionCount += std::max(0,cv::countNonZero(matchesImg.at(im)) - pointArea);
    unionCount += rows*cols - pointArea;
  }
  return matchedPoints < 8 ? 0.0 : double(intersectionCount)/double(unionCount);
}

bool ViSlamBackend::setKeyframe(StateId id, bool isKeyframe) {
  auxiliaryStates_.at(id).isKeyframe = isKeyframe;
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    fullGraph_.setKeyframe(id, isKeyframe);
  }
  return realtimeGraph_.setKeyframe(id, isKeyframe);
}

bool ViSlamBackend::addLandmark(LandmarkId landmarkId, const Eigen::Vector4d &landmark,
                                bool isInitialised)
{
  bool success = realtimeGraph_.addLandmark(landmarkId, landmark, isInitialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    success &= fullGraph_.addLandmark(landmarkId, landmark, isInitialised);
  }
  return success;
}

LandmarkId ViSlamBackend::addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised)
{
  const LandmarkId lmId = realtimeGraph_.addLandmark(homogeneousPoint, initialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(lmId);
  } else {
    fullGraph_.addLandmark(lmId, homogeneousPoint, initialised);
  }
  return lmId;
}

bool ViSlamBackend::setLandmark(LandmarkId landmarkId, const Eigen::Vector4d & landmark,
                                bool isInitialised) {
  bool success = realtimeGraph_.setLandmark(landmarkId, landmark, isInitialised);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    fullGraph_.setLandmark(landmarkId, landmark, isInitialised);
  }
  return success;
}
bool ViSlamBackend::setLandmarkClassification(LandmarkId landmarkId, int classification) {
  if(realtimeGraph_.landmarks_.count(landmarkId)==0) {
    return false;
  }
  realtimeGraph_.landmarks_.at(landmarkId).classification = classification;
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(landmarkId);
  } else {
    fullGraph_.landmarks_.at(landmarkId).classification = classification;
  }
  return true;
}

bool ViSlamBackend::setObservationInformation(
    StateId stateId, size_t camIdx, size_t keypointIdx, const Eigen::Matrix2d & information) {
  KeypointIdentifier kid(stateId.value(), camIdx, keypointIdx);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(realtimeGraph_.observations_.at(kid).landmarkId);
    touchedStates_.insert(stateId);
  } else {
    fullGraph_.observations_.at(kid).errorTerm->setInformation(information);
  }
  realtimeGraph_.observations_.at(kid).errorTerm->setInformation(information);
  return true;
}

bool ViSlamBackend::removeObservation(StateId stateId, size_t camIdx,
                                      size_t keypointIdx)
{
  KeypointIdentifier kid(stateId.value(), camIdx, keypointIdx);
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedLandmarks_.insert(realtimeGraph_.observations_.at(kid).landmarkId);
    touchedStates_.insert(stateId);
  } else {
    fullGraph_.removeObservation(kid);
  }
  bool success = realtimeGraph_.removeObservation(kid);
  multiFrames_.at(stateId)->setLandmarkId(camIdx, keypointIdx, 0);
  return success;
}

bool ViSlamBackend::convertToPoseGraphMst(const std::set<StateId> & framesToConvert,
                                          const std::set<StateId> & framesToConsider,
                                          std::set<StateId> & affectedFrames) {
  // remember landmarks in frames (transformed to sensor frame)
  for(auto pose : framesToConvert) {
    ViGraph::State& state = realtimeGraph_.states_.at(pose);
    kinematics::Transformation T_SW = state.pose->estimate().inverse();
    MultiFramePtr mFrame = multiFrames_.at(pose);
    for(size_t i=0; i<mFrame->numFrames(); ++i) {
      for(size_t k=0; k<mFrame->numKeypoints(i); ++k) {
        uint64_t lmId = mFrame->landmarkId(i,k);
        if(lmId && realtimeGraph_.landmarkExists(LandmarkId(lmId))) {
          Eigen::Vector4d landmark = realtimeGraph_.landmark(LandmarkId(lmId));
          mFrame->setLandmark(i, k, T_SW*landmark,
                              realtimeGraph_.isLandmarkInitialised(LandmarkId(lmId)));
        }
      }
    }
  }

  std::vector<ViGraphEstimator::PoseGraphEdge> poseGraphEdges;
  std::vector<std::pair<StateId,StateId>> removedTwoPoseErrors;
  std::vector<KeypointIdentifier> removedObservations;
  realtimeGraph_.convertToPoseGraphMst(
        framesToConvert, framesToConsider, &poseGraphEdges, &removedTwoPoseErrors,
        &removedObservations);

  // remember affected frames
  for (auto addedEdge : poseGraphEdges) {
    affectedFrames.insert(addedEdge.otherId);
    affectedFrames.insert(addedEdge.referenceId);
  }
  for (auto removedEdge : removedTwoPoseErrors) {
    affectedFrames.insert(removedEdge.first);
    affectedFrames.insert(removedEdge.second);
  }

  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    for(const auto & poseGraphEdge : poseGraphEdges) {
      touchedStates_.insert(poseGraphEdge.referenceId);
      touchedStates_.insert(poseGraphEdge.otherId);
    }
    for(auto removedTwoPoseError : removedTwoPoseErrors) {
      touchedStates_.insert(removedTwoPoseError.first);
      touchedStates_.insert(removedTwoPoseError.second);
    }
    for(auto obs : removedObservations) {
      touchedStates_.insert(StateId(obs.frameId));
      if(fullGraph_.observations_.count(obs)) {
        LandmarkId lmId = fullGraph_.observations_.at(obs).landmarkId;
        if(lmId.isInitialised()) {
          touchedLandmarks_.insert(fullGraph_.observations_.at(obs).landmarkId); /// \todo hack, fix
        }
      }
    }
  } else {
    // replicate fullGraph_
    for(auto obs : removedObservations) {
      fullGraph_.removeObservation(obs);
    }
    for(auto removedTwoPoseError : removedTwoPoseErrors) {
      fullGraph_.removeTwoPoseConstLink(removedTwoPoseError.first, removedTwoPoseError.second);
    }
    for(const auto & poseGraphEdge : poseGraphEdges) {
      fullGraph_.addExternalTwoPoseLink(
            poseGraphEdge.poseGraphError->cloneTwoPoseGraphErrorConst(),
            poseGraphEdge.referenceId, poseGraphEdge.otherId);

    }
  }

  return true;
}

int ViSlamBackend::expandKeyframe(StateId keyframe)
{
  OKVIS_ASSERT_TRUE(Exception, imuFrames_.count(keyframe) == 0, "must be keyframe")
  OKVIS_ASSERT_TRUE(Exception, keyFrames_.count(keyframe) || loopClosureFrames_.count(keyframe),
                    "must be keyframe")
  OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.states_.at(keyframe).twoPoseLinks.size()>0,
                    "must be frontier frame")
  int ctr = 0;
  std::set<LandmarkId> lms;
  std::set<StateId> cnncts;
  std::vector<ceres::TwoPoseGraphError::Observation> allObservations;
  realtimeGraph_.convertToObservations(keyframe, &lms, &cnncts, &allObservations);

  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(cnncts.begin(), cnncts.end());
    touchedLandmarks_.insert(lms.begin(), lms.end());
  } else {
    fullGraph_.removeTwoPoseConstLinks(keyframe);
    for(auto lm : lms) {
      const auto & landmark = realtimeGraph_.landmarks_.at(lm);
      if(!fullGraph_.landmarkExists(lm)) {
        fullGraph_.addLandmark(lm, landmark.hPoint->estimate(), landmark.hPoint->initialized());
        fullGraph_.setLandmarkQuality(lm, landmark.quality);
      }
    }
    for(const auto & obs : allObservations) {
      const bool useCauchy = obs.lossFunction != nullptr;
      LandmarkId landmarkId(obs.hPoint->id());
      fullGraph_.addExternalObservation(obs.reprojectionError, landmarkId,
                             obs.keypointIdentifier, useCauchy);
    }
  }

  for(auto cnnct : cnncts) {
    if(loopClosureFrames_.count(cnnct)==0 && auxiliaryStates_.at(cnnct).isPoseGraphFrame) {
      keyFrames_.insert(cnnct);
      auxiliaryStates_.at(cnnct).isPoseGraphFrame = false;
      ctr++;
    }
    if(keyFrames_.count(cnnct)==0 && auxiliaryStates_.at(cnnct).isPoseGraphFrame) {
      loopClosureFrames_.insert(cnnct);
      auxiliaryStates_.at(cnnct).isPoseGraphFrame = false;
      ctr++;
    }
  }

  return ctr;
}

void ViSlamBackend::eliminateImuFrames(size_t numImuFrames)
{
  std::set<StateId> imuFrames = imuFrames_; // copy so we can later delete
  for(auto id : imuFrames) {
    if(imuFrames_.size() <= numImuFrames) {
      break; // done -- reduced the IMU frames
    }
    if(realtimeGraph_.states_.at(id).isKeyframe) {
      // move to the keyframes that we keep
      imuFrames_.erase(id);
      keyFrames_.insert(id);
      components_.at(currentComponentIdx_).poseIds.insert(id);
      if(loopClosureFrames_.count(id)) {
        loopClosureFrames_.erase(id); // make sure the two sets are not intersecting
      }
      auxiliaryStates_.at(id).isImuFrame = false;
    } else {
      StateId refId = mostOverlappedStateId(id, false);
      auto observations = realtimeGraph_.states_.at(id).observations;
      realtimeGraph_.removeAllObservations(id);
      realtimeGraph_.eliminateStateByImuMerge(id, refId);

      // also remove from loop closure frames
      if(loopClosureFrames_.count(id)) {
        OKVIS_THROW(Exception, "bug")
      }

      // also manage the full graph, if possible
      if(isLoopClosing_ || isLoopClosureAvailable_) {
        for(const auto & obs : observations) {
          touchedLandmarks_.insert(obs.second.landmarkId);
        }
        eliminateStates_[id] = refId;
      } else {
        fullGraph_.removeAllObservations(id); /// \todo make more efficient (copy over)
        fullGraph_.eliminateStateByImuMerge(id, refId); /// \todo make more efficient (copy over)
      }
      imuFrames_.erase(id);
      multiFrames_.erase(id);
      auxiliaryStates_.erase(id);
    }
  }
}

bool ViSlamBackend::applyStrategy(size_t numKeyframes,
                                  size_t numLoopClosureFrames,
                                  size_t numImuFrames,
                                  std::set<StateId> &affectedFrames,
                                  bool expand)
{
  // check/handle lost
  TimerSwitchable t1("7.1 check/handle lost");
  if(imuFrames_.size()>1 && !keyFrames_.empty()) {
    auto iter = auxiliaryStates_.rbegin();
    const double quality = trackingQuality(iter->first);
    if (quality < 0.01) {
      if (quality > 0.00001) {
        LOG(WARNING) << "Tracking quality weak: quality=" << quality;
      } else {
        LOG(WARNING) << "TRACKING FAILURE: quality=" << quality;
      }
      /// \todo: lost component handling currently disabled. Re-introduce!
    }
  }
  t1.stop();

  // first eliminate the IMU frames that are too much and that are not keyframes
  TimerSwitchable t2("7.2 eliminate IMU frames");
  eliminateImuFrames(numImuFrames);
  t2.stop();

  // now tackle keyframes, if necessary
  StateId currentKeyframeId = currentKeyframeStateId();
  StateId currentFrameId = realtimeGraph_.currentStateId();

  if(!currentKeyframeId.isInitialised()) {
    return true; /// \todo fix this
  }

  // now eliminate keyframes
  bool keyFrameEliminated = false;
  int ctrPg = 0;
  if(keyFrames_.size() > numKeyframes) {
    TimerSwitchable t3("7.3 convert to posegraph");
    while(keyFrames_.size() > numKeyframes) {
      // find keyframe with least common observations with current frame or current keyframe
      realtimeGraph_.computeCovisibilities();
      currentKeyframeId = currentKeyframeStateId();
      currentFrameId = realtimeGraph_.currentStateId();
      StateId minId;
      int minObservations = 100000;
      for(auto keyFrame : keyFrames_) {
        const int coObs = std::max(realtimeGraph_.covisibilities(currentFrameId, keyFrame),
                                   realtimeGraph_.covisibilities(currentKeyframeId, keyFrame));
        if(keyFrame == *keyFrames_.begin()){
          if(coObs>=2) {
            continue; // spare
          }
        }
        if(coObs < minObservations) {
          minObservations = coObs;
          minId = keyFrame;
        }
      }
      StateId maxId;
      int maxCoObs=0;
      std::set<StateId> framesToConsider;
      std::set<StateId> observedKeyframes = keyFrames_;
      observedKeyframes.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
      for(auto frameId : observedKeyframes) {
        const int coObs = realtimeGraph_.covisibilities(minId, frameId);
        if(coObs >= maxCoObs){
          maxId = frameId;
          maxCoObs = coObs;
        }
        if(realtimeGraph_.states_.at(frameId).twoPoseLinks.size()>0) {
          // this is a frontier node
          framesToConsider.insert(frameId);
        }
      }

      // convert it.
      std::set<StateId> convertToPosegraphFrames, allObservedFrames;
      convertToPosegraphFrames.insert(minId);
      auxiliaryStates_.at(minId).isPoseGraphFrame = true; // flag it to be posegraph frame
      keyFrames_.erase(minId);
      for(auto id : keyFrames_) {
        auxiliaryStates_.at(minId).recentLoopClosureFrames.insert(id); // don't re-add immediately.
      }
      framesToConsider.insert(minId);
      framesToConsider.insert(maxId);
      keyFrameEliminated = true;
      if(maxCoObs == 0) {
        // handle weird case that might happen with (almost) no matches
        realtimeGraph_.removeAllObservations(minId);
        // also manage the full graph, if possible
        if(isLoopClosing_ || isLoopClosureAvailable_) {
          touchedStates_.insert(minId);
          affectedFrames.insert(minId);
        } else {
          fullGraph_.removeAllObservations(minId);
        }
        continue;
      }
      if (convertToPosegraphFrames.size() > 0) {
        bool success = convertToPoseGraphMst(convertToPosegraphFrames, framesToConsider,
                                             affectedFrames);

        // also free image memory now
        for (auto id : convertToPosegraphFrames) {
          multiFrames_.at(id)->clearAllImages();
          multiFrames_.at(id)->clearAllDepthImages();
        }

        ctrPg++;
        OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.states_.at(minId).observations.size()==0,
                          "observations at ID " << minId.value() << " , success=" << int(success))
        if(ctrPg>=3) {
          break; // max 3 at the time...
        }
      }
    }
    t3.stop();
  }

  // freeze old states
  TimerSwitchable t4("7.4 freezing");
  if(keyFrameEliminated) {
    auto iter = auxiliaryStates_.rbegin();
    StateId oldestKeyFrameId;
    for (size_t p = 0; p<(numKeyframes+numImuFrames) && iter!=auxiliaryStates_.rend(); ++p) {
      iter++;
    }
    if(iter!=auxiliaryStates_.rend()) {
      oldestKeyFrameId = iter->first;
    }

    int ctr = 0;
    for(auto iter = realtimeGraph_.states_.find(oldestKeyFrameId);
        iter != realtimeGraph_.states_.end(); --iter) {
      if(ctr==numRealtimePoseGraphFrames) {
        Time freezeTime = realtimeGraph_.states_.rbegin()->second.timestamp;
        while((freezeTime - iter->second.timestamp).toSec() < minDeltaT) {
          if(iter==realtimeGraph_.states_.begin()) {
            break;
          }
          --iter;
        }
        if(iter!=realtimeGraph_.states_.begin()) {
          // freezing of poses
          // make sure not to go and unfreeze old stuff
          // -- messes up synchronising if concurrentlz loop-optimising
          const StateId freezeId = std::max(lastFreeze_, iter->first);
          // freezing of poses
          lastFreeze_ = freezeId;
          if(realtimeGraph_.states_.find(freezeId) != realtimeGraph_.states_.begin()) {
            realtimeGraph_.freezePosesUntil(freezeId);
          }
          // freezing of speed/bias
          realtimeGraph_.freezeSpeedAndBiasesUntil(freezeId);
        }
        break;
      }
      if(iter==realtimeGraph_.states_.begin()) {
        break;
      }
      ctr++;
    }
  }
  t4.stop();

  // now tackle loop closure frames
  int ctrLc = 0;
  if(loopClosureFrames_.size() > 0) {
    TimerSwitchable t5("7.5 convert to posegraph loop-closure frames");
    do {

      // find keyframe with least common observations with current frame or current keyframe
      realtimeGraph_.computeCovisibilities();
      currentKeyframeId = currentKeyframeStateId();
      currentFrameId = realtimeGraph_.currentStateId();
      StateId minId;
      int minObservations = 100000;
      for(auto loopClosureFrame : loopClosureFrames_) {
        const int coObs = std::max(realtimeGraph_.covisibilities(currentFrameId, loopClosureFrame),
                                realtimeGraph_.covisibilities(currentKeyframeId, loopClosureFrame));
        //size_t coObs = realtimeGraph_.covisibilities(currentFrameId, loopClosureFrame);
        if(coObs < minObservations) {
          minObservations = coObs;
          minId = loopClosureFrame;
        }
      }

      // eliminate no covisibility in any case
      if(minObservations!=0 && loopClosureFrames_.size()<=numLoopClosureFrames) {
        break;
      }

      std::set<StateId> framesToConsider;
      StateId maxId;
      int maxCoObs = 0;
      std::set<StateId> observedKeyframes = keyFrames_;
      observedKeyframes.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
      for(auto frameId : observedKeyframes) {
        const int coObs = realtimeGraph_.covisibilities(minId, frameId);
        if(coObs >= maxCoObs){
          maxId = frameId;
          maxCoObs = coObs;
        }
        if(realtimeGraph_.states_.at(frameId).twoPoseLinks.size()>0) {
          // this is a frontier node
          framesToConsider.insert(frameId);
        }
      }

      // convert it.
      std::set<StateId> convertToPosegraphFrames, allObservedFrames;
      convertToPosegraphFrames.insert(minId);
      auxiliaryStates_.at(minId).isPoseGraphFrame = true; // flag it to be posegraph frame
      loopClosureFrames_.erase(minId);
      framesToConsider.insert(minId);
      framesToConsider.insert(maxId);
      if(maxCoObs == 0) {
        // handle weird case that might happen with (almost) no matches
        realtimeGraph_.removeAllObservations(minId);
        // also manage the full graph, if possible
        if(isLoopClosing_ || isLoopClosureAvailable_) {
          touchedStates_.insert(minId);
          affectedFrames.insert(minId);
        } else {
          fullGraph_.removeAllObservations(minId);
        }
        continue;
      }
      if(convertToPosegraphFrames.size() > 0 && maxCoObs > 0) {
        convertToPoseGraphMst(convertToPosegraphFrames, framesToConsider, affectedFrames);
        ++ctrLc;
        if(ctrLc>=3) {
          break; // max 3 at the time.
        }
      }
    } while (loopClosureFrames_.size() > numLoopClosureFrames);
    t5.stop();
  }

  /// \todo the following would need re-optimisation of landmarks...
  // expand frontier, if necessary
  if(expand && ctrLc<3 && ctrPg<3) {
    TimerSwitchable t6("7.6 expand");
    currentKeyframeId = currentKeyframeStateId();
    if(currentKeyframeId.isInitialised()) {
      if(realtimeGraph_.states_.at(currentKeyframeId).twoPoseLinks.size()>0) {
        expandKeyframe(currentKeyframeId);
      }
    }
    const StateId currentLoopclosureFrameId = currentLoopclosureStateId();
    if(currentLoopclosureFrameId.isInitialised()) {
      if(realtimeGraph_.states_.at(currentLoopclosureFrameId).twoPoseLinks.size()>0) {
        expandKeyframe(currentLoopclosureFrameId);
      }
    }
    t6.stop();
  }

  return true;
}

void ViSlamBackend::optimiseRealtimeGraph(
  int numIter, std::vector<StateId> & updatedStates, int numThreads, bool verbose,
  bool onlyNewestState, bool isInitialised)
{

  //OKVIS_ASSERT_TRUE(Exception, areLandmarksInFrontOfCameras(), "before optimisation")

  // fix current position, if not initialised
  std::unique_ptr<ceres::PoseError> initialFixation;
  ::ceres::ResidualBlockId initialFixationId = nullptr;
  if(!isInitialised){
    Eigen::Matrix<double, 6, 1> informationDiag = Eigen::Matrix<double, 6, 1>::Ones();
    informationDiag[0] = 1.0e8;
    informationDiag[1] = 1.0e8;
    informationDiag[2] = 1.0e8;
    informationDiag[3] = 0.0;
    informationDiag[4] = 0.0;
    informationDiag[5] = 0.0;
    initialFixation.reset(
      new ceres::PoseError(realtimeGraph_.states_.at(StateId(1)).pose->estimate(), informationDiag));
    initialFixationId = realtimeGraph_.problem_->AddResidualBlock(
        initialFixation.get(), nullptr,
        realtimeGraph_.states_.rbegin()->second.pose->parameters());
    kinematics::Transformation T_WS(
        realtimeGraph_.states_.at(StateId(1)).pose->estimate().r(),
        realtimeGraph_.states_.rbegin()->second.pose->estimate().q());
    realtimeGraph_.setPose(realtimeGraph_.states_.rbegin()->first, T_WS);
  }

  // freeze if requested
  bool frozen = false;
  StateId unfreezeId;
  if(onlyNewestState) {
    // paranoid: find last frozen
    for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
        ++riter) {
      if(riter->second.pose->fixed()) {
        break;
      } else {
        unfreezeId = riter->first;
      }
    }

    auto riter = realtimeGraph_.states_.rbegin();
    riter++;
    if(riter != realtimeGraph_.states_.rend()) {
      realtimeGraph_.freezePosesUntil(riter->first);
      realtimeGraph_.freezeSpeedAndBiasesUntil(riter->first);
      frozen = true;
    }
    for(const auto & lm : realtimeGraph_.landmarks_) {
      realtimeGraph_.problem_->SetParameterBlockConstant(lm.second.hPoint->parameters());
    }

    // freeze extrinsics
    for (size_t i = 0; i < realtimeGraph_.cameraParametersVec_.size(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        realtimeGraph_.problem_->SetParameterBlockConstant(
          realtimeGraph_.states_.rbegin()->second.extrinsics.at(i)->parameters());
      }
    }
  }


  realtimeGraph_.options_.linear_solver_type = ::ceres::DENSE_SCHUR;
  realtimeGraph_.optimise(numIter, numThreads, verbose);

  // unfreeze if necessary
  if(onlyNewestState) {
    // unfreeze extrinsics
    for (size_t i = 0; i < realtimeGraph_.cameraParametersVec_.size(); ++i) {
      if (realtimeGraph_.cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
        realtimeGraph_.problem_->SetParameterBlockVariable(
          realtimeGraph_.states_.rbegin()->second.extrinsics.at(i)->parameters());
      }
    }

    if(frozen) {
      realtimeGraph_.unfreezePosesFrom(unfreezeId);
      realtimeGraph_.unfreezeSpeedAndBiasesFrom(unfreezeId);
    }
    if(onlyNewestState) {
      for(const auto & lm : realtimeGraph_.landmarks_) {
        realtimeGraph_.problem_->SetParameterBlockVariable(lm.second.hPoint->parameters());
      }
    }

    // undo initial fixation
    if(initialFixation && initialFixationId) {
      realtimeGraph_.problem_->RemoveResidualBlock(initialFixationId);
    }

    // adopt pose change
    auto riter = realtimeGraph_.states_.rbegin();
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      ViGraph::State & fullState = fullGraph_.states_.at(riter->first);
      fullState.pose->setEstimate(riter->second.pose->estimate());
      fullState.speedAndBias->setEstimate(riter->second.speedAndBias->estimate());
    }

    // check consistency -- currently disabled
    //if(!isLoopClosing_ && !isLoopClosureAvailable_) {
    //  OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.isSynched(fullGraph_), "not synched");
    //}
    return;
  }

  // import landmarks
  if(!onlyNewestState) {
    realtimeGraph_.updateLandmarks();
  }

  // also copy states to observationless and fullGraph (if possible)
  for (auto id : updatedStatesLoopClosureAttempt_) {
    updatedStates.push_back(id); // remember that these were also updated (from initialisation)
  }
  for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
      ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()) {
      break; // done, don't need to iterate further...
    }
    if(!updatedStatesLoopClosureAttempt_.count(riter->first)) {
       updatedStates.push_back(riter->first);
    }
    /// consciously ignoring extrinsics here. \todo generalise for non-fixed extrinsics...
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      ViGraph::State & fullState = fullGraph_.states_.at(riter->first);
      fullState.pose->setEstimate(riter->second.pose->estimate());
      fullState.speedAndBias->setEstimate(riter->second.speedAndBias->estimate());
    }
  }
  updatedStatesLoopClosureAttempt_.clear(); // processed now, so clear

  // ... and landmarks to fullGraph (if possible)
  if(!isLoopClosing_ && !isLoopClosureAvailable_) {
    for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end();
        ++iter) {
      fullGraph_.setLandmark(iter->first, iter->second.hPoint->estimate(),
                             iter->second.hPoint->initialized());
      fullGraph_.setLandmarkQuality(iter->first, iter->second.quality);
    }
  }

  // finally adopt stuff from realtime optimisation results in full and observation-less graphs
  for(auto riter = realtimeGraph_.states_.rbegin(); riter != realtimeGraph_.states_.rend();
      ++riter) {
    if(!riter->second.previousImuLink.errorTerm) {
      continue;
    }
    if(!isLoopClosing_ && !isLoopClosureAvailable_) {
      if (fullGraph_.imuParametersVec_.at(0).use) {
        std::static_pointer_cast<ceres::ImuError>(
          fullGraph_.states_.at(riter->first).previousImuLink.errorTerm)
          ->syncFrom(
            *std::static_pointer_cast<ceres::ImuError>(riter->second.previousImuLink.errorTerm));
      } else {
        std::static_pointer_cast<ceres::PseudoImuError>(
          fullGraph_.states_.at(riter->first).previousImuLink.errorTerm)
          ->syncFrom(*std::static_pointer_cast<ceres::PseudoImuError>(
            riter->second.previousImuLink.errorTerm));
      }
    }
    if(riter->second.pose->fixed()) {
      break;
    }
  }

  // unfreeze if necessary
  if(initialFixation && initialFixationId) {
    realtimeGraph_.problem_->RemoveResidualBlock(initialFixationId);
  }

  // check consistency -- currently disabled
  //if(!isLoopClosing_ && !isLoopClosureAvailable_) {
  //  OKVIS_ASSERT_TRUE(Exception, realtimeGraph_.isSynched(fullGraph_), "not synched");
  //}

  //OKVIS_ASSERT_TRUE(Exception, areLandmarksInFrontOfCameras(), "after optimisation")
}

bool ViSlamBackend::setOptimisationTimeLimit(double timeLimit, int minIterations)
{
  //fullGraph_.setOptimisationTimeLimit(timeLimit, 1); /// \todo Fix hack!
  return realtimeGraph_.setOptimisationTimeLimit(timeLimit, minIterations);
}

bool ViSlamBackend::isLandmarkAdded(LandmarkId landmarkId) const
{
  return realtimeGraph_.isLandmarkAdded(landmarkId);
}

bool ViSlamBackend::isLandmarkInitialised(LandmarkId landmarkId) const
{
  return realtimeGraph_.isLandmarkInitialised(landmarkId);
}

bool ViSlamBackend::setPose(StateId id, const kinematics::TransformationCacheless &pose)
{
  bool success = realtimeGraph_.setPose(id, pose);
  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setPose(id, pose);
  }
  return success;
}

bool ViSlamBackend::setSpeedAndBias(StateId id, const SpeedAndBias &speedAndBias)
{
  bool success = realtimeGraph_.setSpeedAndBias(id, speedAndBias);
  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setSpeedAndBias(id, speedAndBias);
  }
  return success;
}

bool ViSlamBackend::setExtrinsics(StateId id, uchar camIdx,
                                  const kinematics::TransformationCacheless & extrinsics)
{
  bool success = realtimeGraph_.setExtrinsics(id, camIdx, extrinsics);
  // also manage the full graph, if possible
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    touchedStates_.insert(id);
  } else {
    success &= fullGraph_.setExtrinsics(id, camIdx, extrinsics);
  }
  return success;
}

bool ViSlamBackend::isInImuWindow(StateId id) const
{
  return auxiliaryStates_.at(id).isImuFrame;
}

Time ViSlamBackend::timestamp(StateId id) const
{
  return realtimeGraph_.timestamp(id);
}

void ViSlamBackend::drawOverheadImage(cv::Mat &image, int idx) const
{
  const ViGraphEstimator& graph = idx==0 ? realtimeGraph_ : fullGraph_;
  static const double borderPixels = 50;
  // first find max/min
  Eigen::Vector3d min(-0.05,-0.05,-0.05);
  Eigen::Vector3d max(0.05,0.05,0.05);
  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {
    const Eigen::Vector3d pos = iter->second.pose->estimate().r();
    for(int i=0; i<3; ++i) {
      if(pos[i] > max[i]) {
        max[i] = pos[i];
      }
      if(pos[i] < min[i]) {
        min[i] = pos[i];
      }
    }
  }
  const Eigen::Vector3d centre = 0.5*(max+min);
  const double scale = std::min((image.cols-2.0*borderPixels)/(max[0]-min[0]),
      (image.rows-2.0*borderPixels)/(max[1]-min[1]));

  // landmarks
  for(auto iter = graph.landmarks_.begin(); iter != graph.landmarks_.end(); ++iter) {
    const Eigen::Vector4d hp = iter->second.hPoint->estimate();
    const Eigen::Vector3d pos = (hp.head<3>()/hp[3]-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    const double quality = iter->second.quality; /// \todo save/retrieve real quality
    if(cvPos.x > 0.0 && cvPos.y > 0.0 && cvPos.x < (image.cols-1) && cvPos.y < (image.rows-1)) {
        cv::circle(image, cvPos, 1, cv::Scalar(0,std::min(255,20+int(quality/0.03*225.0)),0),
                                               cv::FILLED, cv::LINE_AA);
    }
  }

  // draw co-observations (need to re-compute as cannot call computeCovisibilities in const method)
  std::map<uint64_t, std::map<uint64_t, int>> coObservationCounts;
  for(auto iter=graph.landmarks_.begin(); iter!=graph.landmarks_.end(); ++iter) {
    auto obs = iter->second.observations;
    std::set<uint64> covisibilities;
    for(auto obsiter=obs.begin(); obsiter!=obs.end(); ++obsiter) {
      covisibilities.insert(obsiter->first.frameId);
    }
    for(auto i0=covisibilities.begin(); i0!=covisibilities.end(); ++i0) {
      for(auto i1=covisibilities.begin(); i1!=covisibilities.end(); ++i1) {
        if(*i1>=*i0) {
          continue;
        }
        if(coObservationCounts.find(*i0)==coObservationCounts.end()) {
          coObservationCounts[*i0][*i1] = 1;
        } else {
          if (coObservationCounts.at(*i0).find(*i1)==coObservationCounts.at(*i0).end()) {
            coObservationCounts.at(*i0)[*i1] = 1;
          } else {
            coObservationCounts.at(*i0).at(*i1)++;
          }
        }
      }
    }
  }

  for(auto i0=coObservationCounts.begin(); i0!=coObservationCounts.end(); ++i0) {
    for(auto i1=coObservationCounts.at(i0->first).begin();
        i1!=coObservationCounts.at(i0->first).end(); ++i1) {
      if(!graph.states_.count(StateId(i0->first))) continue;
      if(!graph.states_.count(StateId(i1->first))) continue;
      const Eigen::Vector3d p0 = graph.states_.at(StateId(i0->first)).pose->estimate().r();
      const Eigen::Vector3d p1 = graph.states_.at(StateId(i1->first)).pose->estimate().r();
      const Eigen::Vector3d pos0 = (p0-centre)*scale;
      const Eigen::Vector3d pos1 = (p1-centre)*scale;
      cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
      cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
      double brightness = 50.0 + std::min(205, i1->second*2);
      cv::line(image, cvPos0, cvPos1, cv::Scalar(0,brightness, brightness), 1, cv::LINE_AA);
    }
  }

  // draw old frames
  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {

    const Eigen::Vector3d pos = (iter->second.pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);

    // draw pose graph error
    for(auto piter=iter->second.twoPoseLinks.begin();
        piter != iter->second.twoPoseLinks.end(); piter++) {
      auto refId = piter->second.errorTerm->referencePoseId();
      auto otherId = piter->second.errorTerm->otherPoseId();
      if(refId<otherId) {
        // draw only forward.
        const Eigen::Vector3d p0 = graph.states_.at(refId).pose->estimate().r();
        const Eigen::Vector3d p1 = graph.states_.at(otherId).pose->estimate().r();
        const Eigen::Vector3d pos0 = (p0-centre)*scale;
        const Eigen::Vector3d pos1 = (p1-centre)*scale;
        cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
        cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
        double brightness = 50.0 + std::min(205.0, piter->second.errorTerm->strength());
        cv::line(image, cvPos0, cvPos1, cv::Scalar(brightness/2,brightness, brightness), 3,
                 cv::LINE_AA);
      }
    }
    // draw pose graph error const
    for(auto piter=iter->second.twoPoseConstLinks.begin(); piter !=
        iter->second.twoPoseConstLinks.end(); piter++) {
      const Eigen::Vector3d p0 = graph.states_.at(iter->first).pose->estimate().r();
      const Eigen::Vector3d p1 = graph.states_.at(piter->first).pose->estimate().r();
      const Eigen::Vector3d pos0 = (p0-centre)*scale;
      const Eigen::Vector3d pos1 = (p1-centre)*scale;
      cv::Point2d cvPos0(pos0[0]+image.cols*0.5, -pos0[1]+image.rows*0.5);
      cv::Point2d cvPos1(pos1[0]+image.cols*0.5, -pos1[1]+image.rows*0.5);
      double brightness = 200;
      cv::line(image, cvPos0, cvPos1, cv::Scalar(brightness/2.0,brightness, brightness/2.0), 3,
               cv::LINE_AA);
    }
  }

  // draw frames
  for(auto iter=graph.states_.begin(); iter!=graph.states_.end(); ++iter) {

    const Eigen::Vector3d pos = (iter->second.pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,0,0);
    auto colour2 = cv::Scalar(255,255,255);
    if(!iter->second.isKeyframe) {
      colour1 = cv::Scalar(127,127,127);
    }
    if(idx<=2) {
        if(keyFrames_.count(iter->first)) {
          colour1 = cv::Scalar(255,255,0);
        }
        const bool isPoseGraphFrame = auxiliaryStates_.at(iter->first).isPoseGraphFrame;
        if(isPoseGraphFrame) {
          colour2 = cv::Scalar(127,127,127);
          if(iter->second.pose->fixed()) {
            colour2 = cv::Scalar(25,25,200);
          }
        }
    }

    // draw IMU error
    if(iter->second.nextImuLink.errorTerm) {
      auto iterNext = iter;
      iterNext++;
      const Eigen::Vector3d nextPos = (iterNext->second.pose->estimate().r()-centre)*scale;
      cv::Point2d nextCvPos(nextPos[0]+image.cols*0.5, -nextPos[1]+image.rows*0.5);
      cv::line(image, cvPos, nextCvPos, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }

    // draw point
    cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cvPos, 3, colour2, 1, cv::LINE_AA);
    std::stringstream stream;
    stream << iter->first.value();
    cv::putText(image, stream.str(), cvPos+cv::Point2d(6,3),
                cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  }

  if(idx >2) {
      // some text:
      auto T_WS = graph.states_.rbegin()->second.pose->estimate();
      const SpeedAndBias speedAndBias = graph.states_.rbegin()->second.speedAndBias->estimate();
      std::stringstream postext;
      postext << "position = ["
              << T_WS.r()[0] << ", " << T_WS.r()[1] << ", " << T_WS.r()[2] << "]";
      cv::putText(image, postext.str(), cv::Point(15,15),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream veltext;
      veltext << "velocity = ["
              << speedAndBias[0] << ", " << speedAndBias[1] << ", " << speedAndBias[2] << "]";
      cv::putText(image, veltext.str(), cv::Point(15,35),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream gyroBiasText;
      gyroBiasText << "gyro bias = ["
                   << speedAndBias[3] << ", " << speedAndBias[4] << ", " << speedAndBias[5] << "]";
      cv::putText(image, gyroBiasText.str(), cv::Point(15,55),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      std::stringstream accBiasText;
      accBiasText << "acc bias = ["
                  << speedAndBias[6] << ", " << speedAndBias[7] << ", " << speedAndBias[8] << "]";
      cv::putText(image, accBiasText.str(), cv::Point(15,75),
                      cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      return;
  }

  // check loopClosureFrame co-observability
  std::map<StateId, size_t> counts;
  for(auto iter = loopClosureFrames_.begin(); iter!=loopClosureFrames_.end(); ++iter) {
    counts[*iter] = 0;
  }
  const StateId cId = currentStateId();
  for (auto pit = graph.landmarks_.begin(); pit != graph.landmarks_.end(); ++pit) {
    const auto& residuals = pit->second.observations;
    bool isObservedInNewFrame = false;
    std::set<uint64_t> loopClosureFramesObserved;
    for (const auto& r : residuals) {
      uint64_t poseId = r.first.frameId;
      if(cId.value() == poseId) {
        isObservedInNewFrame = true;
      }
      if(counts.find(StateId(poseId)) != counts.end()) {
        loopClosureFramesObserved.insert(poseId);
      }
    }

    if(isObservedInNewFrame) {
      for(auto iter = loopClosureFramesObserved.begin(); iter!=loopClosureFramesObserved.end();
          ++iter) {
        counts.at(StateId(*iter))++;
      }
    }
  }

  // draw loop closure frames
  for(auto iter=loopClosureFrames_.begin(); iter!=loopClosureFrames_.end(); ++iter) {
    const Eigen::Vector3d pos = (graph.states_.at(*iter).pose->estimate().r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,255,255);
    auto colour2 = cv::Scalar(255,0,0);

    // draw point
    cv::circle(image, cvPos, 5, colour1, cv::FILLED, cv::LINE_AA);
    cv::circle(image, cvPos, int(6.0f+float(counts.at(*iter))/10.0f), colour2,
               int(1+counts.at(*iter))/5, cv::LINE_AA);
  }

  // current position always on top
  auto T_WS = graph.states_.rbegin()->second.pose->estimate();
  const Eigen::Vector3d pos = (T_WS.r()-centre)*scale;
  cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
  auto colour1 = cv::Scalar(127,127,127);
  auto colour2 = cv::Scalar(255,255,255);
  if(graph.states_.rbegin()->second.isKeyframe) {
    colour1 = cv::Scalar(255,255,0);
  }
  cv::circle(image, cvPos, 5, colour2, 2, cv::LINE_AA);
  cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);

  // current keyframe
  StateId kfId = currentKeyframeStateId();
  if(kfId.isInitialised()) {
    kinematics::Transformation T_WS_kf = graph.states_.at(kfId).pose->estimate();
    const Eigen::Vector3d pos = (T_WS_kf.r()-centre)*scale;
    cv::Point2d cvPos(pos[0]+image.cols*0.5, -pos[1]+image.rows*0.5);
    auto colour1 = cv::Scalar(255,255,127);
    auto colour2 = cv::Scalar(255,255,255);

    cv::circle(image, cvPos, 5, colour2, 2, cv::LINE_AA);
    cv::circle(image, cvPos, 3, colour1, cv::FILLED, cv::LINE_AA);
  }

  // some text:
  const SpeedAndBias speedAndBias = graph.states_.rbegin()->second.speedAndBias->estimate();
  std::stringstream postext;
  postext << "position = [" << T_WS.r()[0] << ", " << T_WS.r()[1] << ", " << T_WS.r()[2] << "]";
  cv::putText(image, postext.str(), cv::Point(15,15),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream veltext;
  veltext << "velocity = ["
          << speedAndBias[0] << ", " << speedAndBias[1] << ", " << speedAndBias[2] << "]";
  cv::putText(image, veltext.str(), cv::Point(15,35),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream gyroBiasText;
  gyroBiasText << "gyro bias = ["
               << speedAndBias[3] << ", " << speedAndBias[4] << ", " << speedAndBias[5] << "]";
  cv::putText(image, gyroBiasText.str(), cv::Point(15,55),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
  std::stringstream accBiasText;
  accBiasText << "acc bias = ["
              << speedAndBias[6] << ", " << speedAndBias[7] << ", " << speedAndBias[8] << "]";
  cv::putText(image, accBiasText.str(), cv::Point(15,75),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}

bool ViSlamBackend::getObservedIds(StateId id, std::set<StateId> &observedIds) const
{
  auto iter = realtimeGraph_.states_.find(id);
  if (iter == realtimeGraph_.states_.end()) {
    return false;
  }

  // insert from pose graph edges
  const auto &state = iter->second;
  for (const auto &link : state.twoPoseLinks) {
    observedIds.insert(link.second.state0);
    observedIds.insert(link.second.state1);
  }

  // insert currently co-visible
  if (keyFrames_.count(id) || imuFrames_.count(id) || loopClosureFrames_.count(id)) {
    std::set<LandmarkId> landmarks;
    for (const auto &obs : state.observations) {
      landmarks.insert(obs.second.landmarkId);
    }
    for (const auto &lmId : landmarks) {
      for (const auto &obs : realtimeGraph_.landmarks_.at(lmId).observations) {
        observedIds.insert(StateId(obs.first.frameId));
      }
    }
  }

  // insert frames connected by IMU (if available)
  auto next = iter;
  next++;
  if (next != realtimeGraph_.states_.end()) {
    observedIds.insert(next->first);
  }
  if (iter != realtimeGraph_.states_.begin()) {
    auto previous = iter;
    previous--;
    observedIds.insert(previous->first);
  }

  return true;
}

bool ViSlamBackend::isLoopClosureFrame(StateId frameId) const {
  return loopClosureFrames_.count(frameId) > 0;
}

bool ViSlamBackend::isRecentLoopClosureFrame(StateId frameId) const {
  for(auto id : keyFrames_) {
    if(auxiliaryStates_.at(id).recentLoopClosureFrames.count(frameId)) {
      return true;
    }
  }
  for(auto id : loopClosureFrames_) {
    if(auxiliaryStates_.at(id).recentLoopClosureFrames.count(frameId)) {
      return true;
    }
  }
  return false;
}

void ViSlamBackend::addLoopClosureFrame(StateId loopClosureFrameId,
                                        std::set<LandmarkId> & loopClosureLandmarks,
                                        bool skipFullGraphOptimisation)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_,
                    "trying to add loop closure while loop closing")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "trying to add loop closure while loop closing")

  // remember loop closure frame
  if(loopClosureFrames_.count(loopClosureFrameId)) {
    LOG(WARNING) << "trying to add previously added loop-closure frame";
    return;
  }
  if(keyFrames_.count(loopClosureFrameId)) {
    LOG(WARNING) << "trying to add current keyframe as loop-closure frame" << std::endl;
    return;
  }

  loopClosureFrames_.insert(loopClosureFrameId); // sort/remove later
  currentLoopClosureFrames_.insert(loopClosureFrameId);

  // undo pose graph errors into observations
  std::set<StateId> connectedStates;
  OKVIS_ASSERT_TRUE_DBG(Exception, auxiliaryStates_.at(loopClosureFrameId).isPoseGraphFrame,
                    "must be pose graph frame")
  std::vector<ceres::TwoPoseGraphError::Observation> allObservations;
  realtimeGraph_.convertToObservations(
        loopClosureFrameId, &loopClosureLandmarks, &connectedStates, &allObservations);
  for(auto lm : loopClosureLandmarks) {
    const auto & landmark = realtimeGraph_.landmarks_.at(lm);
    if(!fullGraph_.landmarkExists(lm)) {
      fullGraph_.addLandmark(lm, landmark.hPoint->estimate(), landmark.hPoint->initialized());
      fullGraph_.setLandmarkQuality(lm, landmark.quality);
    }
  }
  fullGraph_.removeTwoPoseConstLinks(loopClosureFrameId);
  for(const auto& obs : allObservations) {
    const bool useCauchy = obs.lossFunction != nullptr;
    LandmarkId landmarkId(obs.hPoint->id());
    fullGraph_.addExternalObservation(obs.reprojectionError, landmarkId,
                           obs.keypointIdentifier, useCauchy);
  }

  // add also connected frames to loop closure frames
  // flag no longer pose graph frames
  for(auto connectedState : connectedStates) {
    if(auxiliaryStates_.at(connectedState).isPoseGraphFrame
       && keyFrames_.count(connectedState)==0) {
      loopClosureFrames_.insert(connectedState);
      currentLoopClosureFrames_.insert(connectedState);
      auxiliaryStates_.at(connectedState).isPoseGraphFrame = false;
    } // else already in active window and considered
  }
  auxiliaryStates_.at(loopClosureFrameId).isPoseGraphFrame = false;

  // remember oldest Id / freeze / unfreeze
  if(!loopClosureFrames_.empty()) {
    StateId oldestId = loopClosureFrameId;
    if(fullGraph_.states_.count(oldestId)) {
      StateId oldestIdToSetVariable = auxiliaryStates_.at(oldestId).loopId;
      // also remember this throughout the created loop
      for(auto iter = auxiliaryStates_.find(oldestIdToSetVariable);
          iter != auxiliaryStates_.end(); ++iter) {
        iter->second.loopId = oldestIdToSetVariable;
      }

      // apply the freeze/unfreeze
      Time oldestT = fullGraph_.states_.at(oldestId).timestamp;
      int ctr = 0;
      for(auto iter = auxiliaryStates_.find(oldestIdToSetVariable); ; --iter) {
        if(ctr == numPoseGraphFrames || iter==auxiliaryStates_.begin()) {
          while((oldestT - fullGraph_.timestamp(iter->first)).toSec()<minDeltaT) {
            if(iter==auxiliaryStates_.begin()) {
              break;
            }
            --iter;
          }

          if (lastFreeze_.isInitialised()) {
            // ensure we go back at least as far as lastFreeze
            if (iter->first > lastFreeze_) {
              while (iter->first > lastFreeze_) {
                if (iter == auxiliaryStates_.begin()) {
                  break;
                }
                --iter;
              }
            }
          }

          fullGraph_.unfreezePosesFrom(iter->first);
          if(iter!=auxiliaryStates_.begin()) {
            fullGraph_.freezePosesUntil(iter->first);
          }
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          if(iter!=auxiliaryStates_.begin()) {
            fullGraph_.freezeSpeedAndBiasesUntil(iter->first);
          }
          break;
        }
        if(iter==auxiliaryStates_.begin()) {
          fullGraph_.unfreezePosesFrom(iter->first);
          fullGraph_.unfreezeSpeedAndBiasesFrom(iter->first);
          break;
        }
        ctr++;
      }
    }
  }

  // remember as recent loop closures
  for(auto id : keyFrames_) {
    auxiliaryStates_.at(id).recentLoopClosureFrames.insert(
          loopClosureFrames_.begin(), loopClosureFrames_.end());
  }
  for(auto id : loopClosureFrames_) {
    auxiliaryStates_.at(id).recentLoopClosureFrames.insert(
          loopClosureFrames_.begin(), loopClosureFrames_.end());
  }

  // signal we need optimisation
  if(!skipFullGraphOptimisation) {
    needsFullGraphOptimisation_ = true;
  }
}

bool ViSlamBackend::synchroniseRealtimeAndFullGraph(std::vector<StateId> &updatedStates)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "cannot synchronise while loop-closing")

  // first, we move the loop-closure frames into the set of regular keyframes...
  keyFrames_.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  loopClosureFrames_.clear();

  // remove landmarks in full graph that have disappeared
  auto landmarks = fullGraph_.landmarks_;
  for(auto iter = landmarks.begin(); iter != landmarks.end(); ++iter) {
    if(realtimeGraph_.landmarks_.count(iter->first) == 0) {
      fullGraph_.removeLandmark(iter->first);
    }
  }

  // compute pose change
  StateId oldFrameId;
  for(auto iter = fullGraph_.states_.crbegin(); iter != fullGraph_.states_.crend(); ++iter) {
    if(realtimeGraph_.states_.count(iter->first)) {
      oldFrameId = iter->first;
      break;
    }
  }

  const kinematics::Transformation T_WS_old = realtimeGraph_.pose(oldFrameId);
  const kinematics::Transformation T_WS_new = fullGraph_.pose(oldFrameId);
  const kinematics::Transformation T_Wnew_Wold = T_WS_new * T_WS_old.inverse();

  // process added new states
  for(const auto& addState : addStatesBacklog_) {
    bool asKeyframe = false;
    const bool existsInRealtimeGraph = realtimeGraph_.states_.count(addState.id) !=0;
    if(existsInRealtimeGraph) {
      if(realtimeGraph_.states_.at(addState.id).isKeyframe) {
        asKeyframe = true;
      }
    }
    fullGraph_.addStatesPropagate(addState.timestamp, addState.imuMeasurements, asKeyframe);
    kinematics::Transformation T_WS;
    SpeedAndBias speedAndBias;
    if(!existsInRealtimeGraph) {
      const auto& anystate = realtimeGraph_.anyState_.at(addState.id);
      const kinematics::Transformation T_Sk_S = anystate.T_Sk_S;
      const kinematics::Transformation T_WSk =
          T_Wnew_Wold*realtimeGraph_.states_.at(anystate.keyframeId).pose->estimate();
      T_WS = T_WSk*T_Sk_S;
      speedAndBias = fullGraph_.states_.at(addState.id).speedAndBias->estimate();
      speedAndBias.head<3>() = T_WSk.C() * anystate.v_Sk;
    } else {
      const kinematics::Transformation T_S0S1 =
          realtimeGraph_.states_.at(oldFrameId).pose->estimate().inverse()*
          realtimeGraph_.states_.at(addState.id).pose->estimate();
      T_WS = fullGraph_.states_.at(oldFrameId).pose->estimate()*T_S0S1;
      const SpeedAndBias speedAndBias_old =
          realtimeGraph_.states_.at(addState.id).speedAndBias->estimate();
      speedAndBias = fullGraph_.states_.at(addState.id).speedAndBias->estimate();
      const Eigen::Vector3d v_S =
          realtimeGraph_.states_.at(oldFrameId).pose->estimate().inverse().C()
          *speedAndBias_old.head<3>();
      speedAndBias.head<3>() = T_WS.C()*v_S;
    }
    fullGraph_.setPose(addState.id, T_WS);
    fullGraph_.setSpeedAndBias(addState.id, speedAndBias);
  }
  addStatesBacklog_.clear();

  for(const auto& eliminateState : eliminateStates_) {
    fullGraph_.removeAllObservations(eliminateState.first);
    /// \todo make more efficient (copy over)
    fullGraph_.eliminateStateByImuMerge(eliminateState.first, eliminateState.second);
    /// \todo make more efficient (copy over)
  }
  eliminateStates_.clear();

  for(const auto& eliminateState : eliminateStates_) {
    const auto iter = fullGraph_.states_.find(eliminateState.second);
    if(iter != fullGraph_.states_.end()) {
      if(iter->second.previousImuLink.errorTerm) {
        if (fullGraph_.imuParametersVec_.at(0).use) {
          std::static_pointer_cast<ceres::ImuError>(iter->second.previousImuLink.errorTerm)
            ->syncFrom(*std::static_pointer_cast<ceres::ImuError>(
              realtimeGraph_.states_.at(eliminateState.second).previousImuLink.errorTerm));
        } else {
          std::static_pointer_cast<ceres::PseudoImuError>(iter->second.previousImuLink.errorTerm)
          ->syncFrom(*std::static_pointer_cast<ceres::PseudoImuError>(
            realtimeGraph_.states_.at(eliminateState.second).previousImuLink.errorTerm));
        }
      }
    }
  }
  for(auto riter = fullGraph_.states_.crbegin(); riter != fullGraph_.states_.crend(); ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()
         && realtimeGraph_.states_.at(riter->first).pose->fixed()
         && realtimeGraph_.states_.at(riter->first).speedAndBias->fixed()) {
      break;
    }
    auto & errorTerm = realtimeGraph_.states_.at(riter->first).previousImuLink.errorTerm;
    if(errorTerm) {
      if (fullGraph_.imuParametersVec_.at(0).use) {
        std::static_pointer_cast<ceres::ImuError>(errorTerm)->syncFrom(
          *std::static_pointer_cast<ceres::ImuError>(riter->second.previousImuLink.errorTerm));
      } else {
        std::static_pointer_cast<ceres::PseudoImuError>(errorTerm)->syncFrom(
          *std::static_pointer_cast<ceres::PseudoImuError>(riter->second.previousImuLink.errorTerm));
      }
    }
  }

  // update new landmarks with pose change and insert into full graph
  for(auto iter = realtimeGraph_.landmarks_.begin(); iter != realtimeGraph_.landmarks_.end();
      ++iter) {
    if(fullGraph_.landmarks_.count(iter->first) == 0) {
      realtimeGraph_.setLandmark(iter->first, T_Wnew_Wold * iter->second.hPoint->estimate());
      fullGraph_.addLandmark(
            iter->first, iter->second.hPoint->estimate(), iter->second.hPoint->initialized());
      fullGraph_.setLandmarkQuality(iter->first, iter->second.quality);
    }
  }

  // copy the result over now
  for(auto riter = fullGraph_.states_.crbegin(); riter != fullGraph_.states_.crend(); ++riter) {
    if(riter->second.pose->fixed() && riter->second.speedAndBias->fixed()
         && realtimeGraph_.states_.at(riter->first).pose->fixed()
         && realtimeGraph_.states_.at(riter->first).speedAndBias->fixed()) {
      /// \todo remove these for safety
      OKVIS_ASSERT_TRUE(Exception, riter->second.pose->estimate().T()
                        ==realtimeGraph_.pose(riter->first).T(), "O-O")
      OKVIS_ASSERT_TRUE(Exception, riter->second.speedAndBias->estimate()
                        ==realtimeGraph_.speedAndBias(riter->first), "O-O")
      break;
    } else {
      updatedStates.push_back(riter->first);
    }
    if(realtimeGraph_.states_.count(riter->first) == 0) {
      OKVIS_THROW(Exception, "impossible: state not present")
      // new state not yet added
      continue;
    }
    realtimeGraph_.setPose(riter->first, riter->second.pose->estimate());
    realtimeGraph_.setSpeedAndBias(riter->first, riter->second.speedAndBias->estimate());
    for(size_t i = 0; i<riter->second.extrinsics.size(); ++i) {
      if(!riter->second.extrinsics.at(i)->fixed()) {
        realtimeGraph_.setExtrinsics(riter->first, uchar(i),
                                     riter->second.extrinsics.at(i)->estimate());
      }
    }
  }

  // update landmarks
  for(auto iter = fullGraph_.landmarks_.begin(); iter != fullGraph_.landmarks_.end(); ++iter) {
    OKVIS_ASSERT_TRUE_DBG(Exception, realtimeGraph_.landmarkExists(iter->first), "not allowed")
    realtimeGraph_.setLandmark(iter->first, iter->second.hPoint->estimate(),
                               iter->second.hPoint->initialized());
    realtimeGraph_.setLandmarkQuality(iter->first, iter->second.quality);
  }

  // process touched landmarks/observations
  OKVIS_ASSERT_TRUE(Exception, fullGraph_.landmarks_.size() == realtimeGraph_.landmarks_.size(),
                    "inconsistent landmark full vs realtime graph")
  for(auto lm : touchedLandmarks_) {
    if(!fullGraph_.landmarkExists(lm)) {
      continue;
    }
    // remove all
    auto observations = fullGraph_.landmarks_.at(lm).observations;
    for(const auto & obs : observations) {
      fullGraph_.removeObservation(obs.first);
    }
  }
  for(auto lm : touchedLandmarks_) {
    if(!fullGraph_.landmarkExists(lm)) {
      continue;
    }
    // copy over
    for(const auto & obs : realtimeGraph_.landmarks_.at(lm).observations) {
      fullGraph_.addExternalObservation(obs.second.errorTerm, lm, obs.first, true);
    }
  }

  // remove all other edges
  for(auto stateId : touchedStates_) {
    if(fullGraph_.states_.count(stateId) == 0) {
      continue; // later deleted
    }
    auto relativePoselinks = fullGraph_.states_.at(stateId).relativePoseLinks;
    for(const auto& link : relativePoselinks) {
      fullGraph_.removeRelativePoseConstraint(link.second.state0, link.second.state1);
    }
    auto twoPoselinks = fullGraph_.states_.at(stateId).twoPoseConstLinks;
    for(const auto& link : twoPoselinks) {
      fullGraph_.removeTwoPoseConstLink(link.second.state0, link.second.state1);
    }
  }

  // re-add
  std::set<::ceres::ResidualBlockId> addedRelativePoseLinks;
  std::set<::ceres::ResidualBlockId> addedTwoPoseLinks;
  for(auto stateId : touchedStates_) {
    if(fullGraph_.states_.count(stateId) == 0) {
      continue; // later deleted
    }
    for(const auto& relPoseLink : realtimeGraph_.states_.at(stateId).relativePoseLinks) {
      if(addedRelativePoseLinks.count(relPoseLink.second.residualBlockId)==0) {
        fullGraph_.addRelativePoseConstraint(
              relPoseLink.second.state0, relPoseLink.second.state1,
              relPoseLink.second.errorTerm->T_AB(),
              relPoseLink.second.errorTerm->information());
        addedRelativePoseLinks.insert(relPoseLink.second.residualBlockId);
      }
    }
    for(const auto& twoPoseLink : realtimeGraph_.states_.at(stateId).twoPoseLinks) {
      if(addedTwoPoseLinks.count(twoPoseLink.second.residualBlockId)==0) {
        fullGraph_.addExternalTwoPoseLink(
              twoPoseLink.second.errorTerm->cloneTwoPoseGraphErrorConst(),
              twoPoseLink.second.state0, twoPoseLink.second.state1);
        addedTwoPoseLinks.insert(twoPoseLink.second.residualBlockId);
      }
    }
  }
  touchedStates_.clear();
  touchedLandmarks_.clear();

  OKVIS_ASSERT_TRUE(Exception, fullGraph_.landmarks_.size() == realtimeGraph_.landmarks_.size(),
                    "inconsistent landmarks full vs realtime graph")
  OKVIS_ASSERT_TRUE(Exception, fullGraph_.states_.size() == realtimeGraph_.states_.size(),
                    "inconsistent states full vs realtime graph")

  /*if(!realtimeGraph_.isSynched(fullGraph_)) {
    std::cout << "not synched" << std::endl;
    OKVIS_THROW(Exception, "not synched")
  }*/

  // loop closure processed
  currentLoopClosureFrames_.clear();
  isLoopClosureAvailable_ = false;

  return true;
}

int ViSlamBackend::cleanUnobservedLandmarks() {
  // check consistency -- currently disabled
  //if (!isLoopClosing_ && !isLoopClosureAvailable_) {
  //  OKVIS_ASSERT_TRUE(Exception,
  //                    realtimeGraph_.isSynched(fullGraph_),
  //                    "trying to clean unobserved landmarks while loop closing")
  //}
  std::map<LandmarkId, std::set<KeypointIdentifier>> removed;
  int removed1 = realtimeGraph_.cleanUnobservedLandmarks(&removed);
  for(const auto & rem : removed) {
    for(const auto & obs : rem.second) {
      // note: it can happen (rarely) that a landmark gets cleaned but then re-added,
      // so the respective frame observations might be missed in the synchronisation.
      multiFrame(StateId(obs.frameId))->setLandmarkId(obs.cameraIndex, obs.keypointIndex, 0);
    }
  }
  if(isLoopClosing_ || isLoopClosureAvailable_) {
    for(const auto & rem : removed) {
      touchedLandmarks_.insert(rem.first);
      for(auto obs : rem.second) {
        // note: it can happen (rarely) that a landmark gets cleaned but then re-added,
        // so the respective frame observations might be missed in the synchronisation.
        touchedStates_.insert(StateId(obs.frameId));
      }
    }
    return removed1; /// \todo This can be done with some refactoring
  } else {
    int removed0 = fullGraph_.cleanUnobservedLandmarks();
    OKVIS_ASSERT_TRUE(Exception, removed0 == removed1, "landmarks cleaned inconsistent!")
    return removed1;
  }

}

bool ViSlamBackend::mergeLandmark(const LandmarkId &fromId, const LandmarkId &intoId)
{
  bool success = (realtimeGraph_.mergeLandmark(fromId, intoId, multiFrames_));
  // also reset associated keypoints
  auto observations = realtimeGraph_.landmarks_.at(intoId).observations;
  for (const auto &observation : observations) {
    multiFrames_.at(StateId(observation.first.frameId))
      ->setLandmarkId(observation.first.cameraIndex,
                      observation.first.keypointIndex,
                      intoId.value());
  }

  if (isLoopClosing_ || isLoopClosureAvailable_) {
    for (const auto &obs : observations) {
      touchedStates_.insert(StateId(obs.first.frameId));
    }
    touchedLandmarks_.insert(fromId);
    touchedLandmarks_.insert(intoId);
  } else {
    // now merge
    success &= fullGraph_.mergeLandmark(fromId, intoId, multiFrames_);
  }

  return success;
}

int ViSlamBackend::mergeLandmarks(std::vector<LandmarkId> fromIds, std::vector<LandmarkId> intoIds)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, fromIds.size() == intoIds.size(), "vectors must be same lengths")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "loop closure not finished, cannot merge landmarks")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "loop closure not finished, cannot merge landmarks")

  std::map<LandmarkId, LandmarkId> changes; // keep track of changes
  int ctr = 0;
  for(size_t i = 0; i < fromIds.size(); ++i) {
    // check if the fromId hasn't already been changed
    while(changes.count(fromIds.at(i))) {
      fromIds.at(i) = changes.at(fromIds.at(i));
    }
    // check if the intoId hasn't already been changed
    while(changes.count(intoIds.at(i))) {
      intoIds.at(i) = changes.at(intoIds.at(i));
    }
    // check if the change hasn't been indirectly applied already
    if(fromIds.at(i) == intoIds.at(i)) {
      continue; //this has already been done.
    }

    // now merge
    if(realtimeGraph_.mergeLandmark(fromIds.at(i), intoIds.at(i), multiFrames_)) {
      ctr++;
    }
    fullGraph_.mergeLandmark(fromIds.at(i), intoIds.at(i), multiFrames_);
    changes[fromIds.at(i)] = intoIds.at(i);

    // also reset associated keypoints
    auto observations = realtimeGraph_.landmarks_.at(intoIds.at(i)).observations;
    for(const auto & observation : observations) {
      multiFrames_.at(StateId(observation.first.frameId))->setLandmarkId(
            observation.first.cameraIndex, observation.first.keypointIndex, intoIds.at(i).value());
    }
  }

  return ctr;
}

void ViSlamBackend::optimiseFullGraph(int numIter, ::ceres::Solver::Summary &summary,
                                      int numThreads, bool verbose)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "can't optimise posegraph; already loop-closing")
  needsFullGraphOptimisation_ = false;
  isLoopClosing_ = true;
  if(!fullGraphRelativePoseConstraints_.empty()) {
    kinematics::Transformation T_SS_measured;
    StateId pose_i, pose_j;
    for(auto & item : fullGraphRelativePoseConstraints_) {
      T_SS_measured = item.T_Si_Sj;
      pose_i = item.pose_i;
      pose_j = item.pose_j;
      fullGraph_.addRelativePoseConstraint(item.pose_i, item.pose_j, item.T_Si_Sj,
                                           100*item.information);
    }
    fullGraph_.options().function_tolerance = 0.001;
    fullGraph_.optimise(numIter/3, numThreads, verbose);
    for(auto & item : fullGraphRelativePoseConstraints_) {
      fullGraph_.removeRelativePoseConstraint(item.pose_i, item.pose_j);
    }
    fullGraphRelativePoseConstraints_.clear();
  }
  fullGraph_.options().function_tolerance = 1e-6;

  fullGraph_.optimise(numIter, numThreads, verbose);
  summary = fullGraph_.summary();
  fullGraph_.updateLandmarks(); /// \todo check to do better

  isLoopClosureAvailable_ = true;
  isLoopClosing_ = false;
}

void ViSlamBackend::doFinalBa(
    int numIter, ::ceres::Solver::Summary &summary,
    std::set<StateId> & updatedStatesBa, double extrinsicsPositionUncertainty,
    double extrinsicsOrientationUncertainty, int numThreads, bool verbose)
{
  // convert all posegraph edges
  int i=0;
  for(const auto & state : fullGraph_.states_) {
    if(state.second.twoPoseConstLinks.size()>0) {
      if(keyFrames_.count(state.first) == 0 && loopClosureFrames_.count(state.first) == 0) {
        keyFrames_.insert(state.first);
      }
      std::cout << "\rConstructing VI-BA problem... "
                << std::round(10000.0*double(i)/double(fullGraph_.states_.size()))/100.0 << "%";
      expandKeyframe(state.first);
      i++;
    }
  }
  fullGraph_.cleanUnobservedLandmarks();
  std::cout << "\rConstructing VI-BA problem... " << "100.00%" << std::endl;

  // unfreeze
  fullGraph_.unfreezePosesFrom(StateId(1));
  fullGraph_.unfreezeSpeedAndBiasesFrom(StateId(1));

  // make sure IMU errors are reintegrated if needed
  ceres::ImuError::redoPropagationAlways = true;

  // optimise
  std::cout << "Running optimisation (verbose="
            << (verbose?"true)":"false)...") << std::endl;
  optimiseFullGraph(numIter, summary,numThreads, verbose);

  // remove speed and bias prior
  fullGraph_.removeSpeedAndBiasPrior(StateId(1));

  // remove extrinsics fixation
  if(extrinsicsPositionUncertainty > 0.0 && extrinsicsOrientationUncertainty > 0.0) {
    std::cout << "Running optimisation again without priors & fixation (verbose="
              << (verbose?"true)":"false)...") << std::endl;
    fullGraph_.setExtrinsicsVariable();
    fullGraph_.softConstrainExtrinsics(
          extrinsicsPositionUncertainty, extrinsicsOrientationUncertainty);
  } else {
    std::cout << "Running optimisation again without priors (verbose="
              << (verbose?"true)":"false)...") << std::endl;
  }

  // optimise
  optimiseFullGraph(numIter, summary,numThreads, verbose);

  cv::Mat img=cv::Mat::zeros(2000,2000,CV_8UC3);
  drawOverheadImage(img, 2);
  cv::imwrite("fullBa.png", img);

  // print
  const auto & state = fullGraph_.states_.rbegin()->second;
  for(size_t i = 0; i < state.extrinsics.size(); ++i) {
    std::cout << "extrinsics T_SC" << i << ":" << std::endl;
    std::cout << std::setprecision(15) << state.extrinsics.at(i)->estimate().T() << std::endl;
  }
  Eigen::Matrix<double,6,1> biases = Eigen::Matrix<double,6,1>::Zero();
  for(const auto & s : fullGraph_.states_) {
    biases += s.second.speedAndBias->estimate().tail<6>();
  }
  biases = (1.0/double(fullGraph_.states_.size()))*biases;
  std::cout << "biases: " << std::setprecision(15) << biases.transpose() << std::endl;
  Eigen::Matrix<double,6,1> biasesStd = Eigen::Matrix<double,6,1>::Zero();
  for(const auto & s : fullGraph_.states_) {
    Eigen::Matrix<double,6,1> diff = (s.second.speedAndBias->estimate().tail<6>()-biases);
    biasesStd += diff.cwiseAbs2();
  }
  std::cout << "biases stdev: " << biasesStd.cwiseSqrt().transpose() << std::endl;

  // some plotting (currently disabled)
  std::vector<cv::Mat> images;
  std::vector<cv::Mat> outliers;
  std::vector<int> inlierCtrs;
  std::vector<int> outlierCtrs;
  for(size_t im = 0; im<multiFrames_.rbegin()->second->numFrames(); ++im) {
    inlierCtrs.push_back(0);
    outlierCtrs.push_back(0);
    images.push_back(cv::Mat::zeros(
          int(multiFrames_.rbegin()->second->geometry(im)->imageHeight()),
          int(multiFrames_.rbegin()->second->geometry(im)->imageWidth()), CV_8UC3));
    outliers.push_back(cv::Mat::zeros(
          int(multiFrames_.rbegin()->second->geometry(im)->imageHeight()),
          int(multiFrames_.rbegin()->second->geometry(im)->imageWidth()), CV_8UC3));
  }

  for(const auto & obs : fullGraph_.observations_) {
    auto frame = multiFrames_.at(StateId(obs.first.frameId));
    cv::KeyPoint kpt;
    frame->getCvKeypoint(obs.first.cameraIndex, obs.first.keypointIndex, kpt);
    double* params[3];
    params[0] = fullGraph_.states_.at(StateId(obs.first.frameId)).pose->parameters();
    params[1] = fullGraph_.landmarks_.at(obs.second.landmarkId).hPoint->parameters();
    params[2] = fullGraph_.states_.at(StateId(obs.first.frameId)).extrinsics.at(
          obs.first.cameraIndex)->parameters();
    Eigen::Vector2d err;
    obs.second.errorTerm->Evaluate(params, err.data(), nullptr);
    double badness = sqrt(err.transpose()*err)/3.0; // Chi2 threshold 9
    if(badness>1.0) {
      cv::circle(outliers.at(obs.first.cameraIndex), kpt.pt, 1, cv::Scalar(255,0,0),
                 cv::FILLED, cv::LINE_AA);
      outlierCtrs.at(obs.first.cameraIndex)++;
    } else {
      cv::Scalar colour(0,std::max(0.0,255.0*(1.0-badness)),std::min(255.0,255.0*badness));
      cv::circle(images.at(obs.first.cameraIndex), kpt.pt, 1, colour, cv::FILLED, cv::LINE_AA);
      inlierCtrs.at(obs.first.cameraIndex)++;
    }

  }

  /*// show debug outlier image -- disabled
  for(size_t im = 0; im<multiFrames_.rbegin()->second->numFrames(); ++im) {
    std::stringstream namestr;
    namestr << "Reprojection error image " << im << " (" << inlierCtrs.at(im) << ")";
    cv::imshow(namestr.str(), images[im]);
    std::stringstream outlierstr;
    outlierstr << "Outliers image " << im << " (" << outlierCtrs.at(im) << ")";
    cv::imshow(outlierstr.str(), outliers[im]);
  }*/

  // synchronise and communicate that we used all states
  std::vector<StateId> updatedStates;
  synchroniseRealtimeAndFullGraph(updatedStates);
  for(const auto & state : fullGraph_.states_) {
    updatedStatesBa.insert(state.first);
  }

  // get rid of unobserved landmarks
  cleanUnobservedLandmarks();
}

bool ViSlamBackend::saveMap(std::string path)
{
  // save in g2o format
  std::string g2oPath = path.substr(0, path.size() - 4) + ".g2o";
  Component component(fullGraph_.imuParametersVec_[0],
                      multiFrames_.at(StateId(1))->cameraSystem(),
                      fullGraph_,
                      multiFrames_);
  component.save(g2oPath);

  // ... and as own format
  std::ofstream file(path);
  if(!file.good()) {
    return false;
  }
  
  std::ios init(nullptr);
  init.copyfmt(file);
  
  // first we write all the landmarks
  file << "landmarks:" << std::endl;
  for(const auto & landmark : fullGraph_.landmarks_) {
    if(landmark.second.quality > 0.001) {
      // ID
      file << landmark.first.value() << ",";
      // save position
      Eigen::Vector4d hposition = landmark.second.hPoint->estimate();
      Eigen::Vector3d position = hposition.head<3>()/hposition[3];
      file << position[0] << "," << position[1] << "," << position[2];
      file << std::endl;
    }
  }
  
  // next write the frames with keypoint descriptors
  fullGraph_.computeCovisibilities();
  for(const auto & state : fullGraph_.states_) {
    file << "frame: " << state.first.value() << ", covisibilities: ";
    std::vector<std::pair<StateId, int>> covisibilities;
    for(const auto & state2 : fullGraph_.states_) {
      int c = fullGraph_.covisibilities(state.first, state2.first);
      if(c > 0) {
        covisibilities.push_back(std::make_pair(state2.first, c));
      }
    }
    std::sort(covisibilities.begin(), covisibilities.end(), 
         [](const std::pair<StateId, int> & left, const std::pair<StateId, int> & right){
           return left.second > right.second;});
    for(const auto & covisibility : covisibilities) {
      file << covisibility.first.value() << " ";
    }
    file << std::endl;
    for(const auto & obs : state.second.observations) {
      const auto & landmark = fullGraph_.landmarks_.at(obs.second.landmarkId);
      if(landmark.quality > 0.001) {
        file << obs.first.keypointIndex << "," << obs.second.landmarkId.value() << ","; // kpt&lm ID
        Eigen::Vector4d hposition = landmark.hPoint->estimate();
        Eigen::Vector3d position = hposition.head<3>()/hposition[3];
        file << position[0] << "," << position[1] << "," << position[2] << ","; // lm 3D pos redund.
        // retrieve descriptor
        const unsigned char* descriptor =
            multiFrames_.at(state.first)->keypointDescriptor(
              obs.first.cameraIndex, obs.first.keypointIndex);
        for(size_t i=0; i<48; ++i) {
          file << std::setfill('0') << std::setw(2) << std::hex << uint32_t(descriptor[i]);
        }
        file.copyfmt(init); // reset formatting
        file << std::endl;
      }
    }
  }

  return true;
}

bool ViSlamBackend::writeFinalCsvTrajectory(const std::string &csvFileName, bool rpg) const
{
  std::fstream csvFile(csvFileName.c_str(), std::ios_base::out);
  bool success =  csvFile.good();
  if(!success) {
    return false;
  }

  // write description
  if(rpg) {
    csvFile << "# timestamp tx ty tz qx qy qz qw" << std::endl;
  } else {
    csvFile << "timestamp" << ", " << "p_WS_W_x" << ", " << "p_WS_W_y" << ", "
               << "p_WS_W_z" << ", " << "q_WS_x" << ", " << "q_WS_y" << ", "
               << "q_WS_z" << ", " << "q_WS_w" << ", " << "v_WS_W_x" << ", "
               << "v_WS_W_y" << ", " << "v_WS_W_z" << ", " << "b_g_x" << ", "
               << "b_g_y" << ", " << "b_g_z" << ", " << "b_a_x" << ", " << "b_a_y"
               << ", " << "b_a_z" << std::endl;
  }
  for(auto iter=realtimeGraph_.anyState_.begin(); iter!=realtimeGraph_.anyState_.end(); ++iter) {
    Eigen::Vector3d p_WS_W;
    Eigen::Quaterniond q_WS;
    SpeedAndBias speedAndBiases;
    std::stringstream time;
    if(iter->second.keyframeId.isInitialised()) {
      // reconstruct from close keyframe
      const ViGraph::State& keyframeState = realtimeGraph_.states_.at(iter->second.keyframeId);
      kinematics::Transformation T_WS = keyframeState.pose->estimate() * iter->second.T_Sk_S;
      p_WS_W = T_WS.r();
      q_WS = T_WS.q();
      speedAndBiases.head<3>() =  keyframeState.pose->estimate().C() * iter->second.v_Sk;
      speedAndBiases.tail<6>() = keyframeState.speedAndBias->estimate().tail<6>();
      time << iter->second.timestamp.sec << std::setw(9) << std::setfill('0')
           << iter->second.timestamp.nsec;
    } else {
      // read from state
      const ViGraph::State& state = realtimeGraph_.states_.at(iter->first);
      p_WS_W = state.pose->estimate().r();
      q_WS = state.pose->estimate().q();
      speedAndBiases = state.speedAndBias->estimate();
      time << state.timestamp.sec << std::setw(9) << std::setfill('0')
           << state.timestamp.nsec;
    }
    if(rpg) {
      csvFile << std::setprecision(19) << iter->second.timestamp.toSec() << " "
          << p_WS_W[0] << " " << p_WS_W[1] << " " << p_WS_W[2] << " "
          << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " " << q_WS.w() << std::endl;
    } else {
      csvFile << time.str() << ", " << std::scientific
          << std::setprecision(18) << p_WS_W[0] << ", " << p_WS_W[1] << ", "
          << p_WS_W[2] << ", " << q_WS.x() << ", " << q_WS.y() << ", "
          << q_WS.z() << ", " << q_WS.w() << ", " << speedAndBiases[0] << ", "
          << speedAndBiases[1] << ", " << speedAndBiases[2] << ", "
          << speedAndBiases[3] << ", " << speedAndBiases[4] << ", "
          << speedAndBiases[5] << ", " << speedAndBiases[6] << ", "
          << speedAndBiases[7] << ", " << speedAndBiases[8] << ", "
          << iter->second.keyframeId.value()<< std::endl;
    }
  }

  // done, close.
  csvFile.close();
  return success;
}

bool ViSlamBackend::attemptLoopClosure(StateId pose_i, StateId pose_j,
                                       const kinematics::Transformation& T_Si_Sj,
                                       const Eigen::Matrix<double, 6, 6>& information,
                                       bool & skipFullGraphOptimisation,
                                       double driftPercentageHeuristic)
{
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosing_, "Loop closure still running")
  OKVIS_ASSERT_TRUE(Exception, !isLoopClosureAvailable_,
                    "loop closure not finished, cannot merge landmarks")

  // check if poses exist, signal unsuccessful otherwise
  if(realtimeGraph_.states_.count(pose_i) == 0) {
    return false;
  }
  if(realtimeGraph_.states_.count(pose_j) == 0) {
    return false;
  }

  int numSteps = 0; // number of steps to distribute error over
  StateId lastLoopId = auxiliaryStates_.at(pose_i).loopId;
  for(auto iter = realtimeGraph_.states_.find(pose_i);
      iter != realtimeGraph_.states_.end(); ++iter) {
    const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
    if(lastLoopId!=loopId) {
      lastLoopId = loopId;
      numSteps++;
    }
  }
  StateId firstId = auxiliaryStates_.at(pose_i).loopId;
  lastLoopId = firstId;
  double distanceTravelled = 0.0;
  double distanceTravelled2 = 0.0; // just for the heuristic
  kinematics::Transformation T_WS_i = realtimeGraph_.states_.at(pose_i).pose->estimate();
  kinematics::Transformation T_WS_last = realtimeGraph_.states_.at(pose_i).pose->estimate();
  auto iter = realtimeGraph_.states_.find(pose_i);
  std::vector<double> distances;
  ++iter;
  Eigen::Vector3d distanceTravelledVec(0.0,0.0,0.0);
  if(pose_i != lastLoopId) {
    distanceTravelledVec +=
        (realtimeGraph_.states_.at(lastLoopId).pose->estimate().r()-T_WS_i.r());
    distanceTravelled2 += distanceTravelledVec.norm();

  }
  for(; iter != realtimeGraph_.states_.end(); ++iter) {
    const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
    kinematics::Transformation T_WS_j = iter->second.pose->estimate();
    T_WS_last = T_WS_j;
    if(lastLoopId!=loopId) {
      const Eigen::Vector3d dsVec = (T_WS_j.r()-T_WS_i.r());
      const double ds = dsVec.norm();
      lastLoopId = loopId;
      distances.push_back(ds);
      distanceTravelledVec += dsVec;
      distanceTravelled += ds;
      distanceTravelled2 += ds;
      T_WS_i = T_WS_j;
    }
  }

  skipFullGraphOptimisation = false;
  {

    // we don't optimise, but just re-align the new part to the old part.
    // we distribute the inconsistency along the loop (i.e. what should be variable)
    // according to changing loopIds (i.e. leaving previously closed loops rigid)

    const kinematics::Transformation T_WSi = realtimeGraph_.pose(pose_i);
    const kinematics::Transformation T_WSj_old= realtimeGraph_.pose(pose_j);
    const kinematics::Transformation T_Si_Sj_old = T_WSi.inverse() * T_WSj_old;
    const kinematics::Transformation T_WSj_new = T_WSi * T_Si_Sj;
    kinematics::Transformation T_Wnew_Wold_final = T_WSj_new * T_WSj_old.inverse();

    // compute rotation increment for the later averaging
    Eigen::AngleAxisd aa(T_Wnew_Wold_final.q());
    aa.angle() = aa.angle() * (1.0 / double(numSteps));
    const kinematics::Transformation T_WW(Eigen::Vector3d(0.0,0.0,0.0), Eigen::Quaterniond(aa));

    // compute rotation adjustments
    kinematics::Transformation T_WS_prev = realtimeGraph_.pose(pose_i);
    kinematics::Transformation T_WS = realtimeGraph_.pose(pose_i);
    auto iter = realtimeGraph_.states_.find(pose_i);
    lastLoopId = firstId;
    iter++;
    for( ; iter != realtimeGraph_.states_.end(); ++iter) {
      const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
      const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
      T_WS_prev = T_WSk_old;
      const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
      if(lastLoopId!=loopId) {
        T_WS = T_WW * T_WS * T_SS;
        lastLoopId = loopId;
      } else {
        T_WS = T_WS * T_SS;
      }
    }

    // compute translation error to be distributed later
    const Eigen::Vector3d dr_W = T_WSj_new.r()-T_WS.r();

    // heuristic verification: check relative trajectory errors
    const double relPositionError = dr_W.norm()/(distanceTravelled2);
    const double relOrientationError =
        T_WSj_new.q().angularDistance(T_WSj_old.q())/double(numSteps);
    const double relPositionErrorBudget = // [m/m]
            driftPercentageHeuristic/100.0 + // 1.35% position bias default
            0.02*distanceTravelledVec.norm()/distanceTravelled2 + // 2% scale error
            0.08/sqrt(numSteps); // position noise, 8% stdev per step
    const double relOrientationErrorBudget =
        0.0004 + 0.004/sqrt(numSteps); // bias and noise, in rad/step
    if(relPositionError > relPositionErrorBudget
        || relOrientationError > relOrientationErrorBudget
        || numSteps < 1) {

      LOG(INFO) << pose_j.value() << "->" << pose_i.value() << " : "
                << "Skip loop closure (heuristic consistency).";
      LOG(INFO) << "Rel. pos. err. " << relPositionError << " vs budget "
          << relPositionErrorBudget << " m/m, rel. or. err. "
          << relOrientationError << " vs budget "
          << relOrientationErrorBudget << " rad/kf";
      LOG(INFO) << "dist. travelled " << distanceTravelled2 << " m, no. steps " << numSteps;

      return false;
    }

    // do not accept uncertain loop closures
    Eigen::Matrix<double, 6, 6> P;
    okvis::PseudoInverse::symm(information, P);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(P.topLeftCorner<3, 3>());
    Eigen::Vector3d eigenvalues = saes.eigenvalues();
    double sigma = sqrt(eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
    if (sigma > 0.1 && 3.0 * sigma > relPositionErrorBudget * distanceTravelled2) {
      LOG(INFO) << pose_j.value() << "->" << pose_i.value() << " : "
                << "Skip loop closure (reloc. 3-sigma = " << 3.0*sigma << " m vs budget "
                   << relPositionErrorBudget*distanceTravelled2 << ") ";
      return false;
    }

    // compute full adjustments
    T_WS_prev = realtimeGraph_.pose(pose_i);
    T_WS = realtimeGraph_.pose(pose_i);
    iter = realtimeGraph_.states_.find(pose_i);
    lastLoopId = firstId;
    iter++;
    int ctr = 0;
    double r = 0.0;
    for( ; iter != realtimeGraph_.states_.end(); ++iter) {
      const kinematics::Transformation T_WSk_old = iter->second.pose->estimate();
      const kinematics::Transformation T_SS = T_WS_prev.inverse()*T_WSk_old;
      T_WS_prev = T_WSk_old;
      const StateId loopId = auxiliaryStates_.at(iter->first).loopId;
      if(lastLoopId!=loopId) {
        // we weight distance adjustments by distance travelled, and rotation uniformly.
        r +=  distances.at(size_t(ctr)) / distanceTravelled;
        T_WS = T_WW *  T_WS * T_SS;
        lastLoopId = loopId;
        ++ctr;
      } else {
        T_WS = T_WS * T_SS;
      }
      const kinematics::Transformation T_WS_set(T_WS.r() + r*dr_W, T_WS.q());
      const kinematics::Transformation T_Wnew_Wold = T_WS_set * T_WSk_old.inverse();
      SpeedAndBias speedAndBias = iter->second.speedAndBias->estimate();
      const Eigen::Vector3d v_Wold = speedAndBias.head<3>();
      speedAndBias.head<3>() = T_Wnew_Wold.C() * v_Wold;
      realtimeGraph_.setPose(iter->first, T_WS_set);
      fullGraph_.setPose(iter->first, T_WS_set);
      realtimeGraph_.setSpeedAndBias(iter->first, speedAndBias);
      fullGraph_.setSpeedAndBias(iter->first, speedAndBias);
      updatedStatesLoopClosureAttempt_.insert(iter->first);
    }

    // update landmarks
    for(auto iter = realtimeGraph_.landmarks_.begin();
        iter != realtimeGraph_.landmarks_.end(); ++iter) {
      // TODO: check if this is always right!
      Eigen::Vector4d hPointNew = T_Wnew_Wold_final * iter->second.hPoint->estimate();
      realtimeGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
      fullGraph_.setLandmark(iter->first, hPointNew, iter->second.hPoint->initialized());
    }
  }

  // remember this frame closed a loop
  auxiliaryStates_.at(pose_j).closedLoop = true;

  fullGraphRelativePoseConstraints_.push_back(RelPoseInfo{T_Si_Sj, information, pose_i, pose_j});

  //const kinematics::Transformation T_SiSj_estimated = (pose(pose_i).inverse() * pose(pose_j));
  //OKVIS_ASSERT_TRUE(Exception,
  //                  (T_SiSj_estimated.inverse() * T_Si_Sj).T().isIdentity(1.0e-6),
  //                  "We are f**ked: T_SiSj_estimated = \n" << T_SiSj_estimated.T() << std::endl <<
  //                  "T_Si_Sj = \n" << T_Si_Sj.T())

  return true;
}

StateId ViSlamBackend::currentKeyframeStateId(bool considerLoopClosureFrames) const
{
  StateId currentFrame = currentStateId();
  return mostOverlappedStateId(currentFrame, considerLoopClosureFrames);
}

StateId ViSlamBackend::mostOverlappedStateId(StateId frame, bool considerLoopClosureFrames) const
{
  std::set<StateId> allFrames;
  allFrames.insert(keyFrames_.begin(), keyFrames_.end());
  allFrames.insert(imuFrames_.begin(), imuFrames_.end());
  allFrames.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  if(!considerLoopClosureFrames) {
    for(const auto & id : currentLoopClosureFrames_)
      allFrames.erase(id);
  }
  StateId returnId;
  double overlap = 0.0;
  for(auto id : allFrames) {
    if(id==frame) {
      continue;
    }
    if(!realtimeGraph_.states_.at(id).isKeyframe) {
      continue;
    }
    double thisOverlap = overlapFraction(multiFrames_.at(id), multiFrames_.at(frame));
    if(thisOverlap >= overlap) {
      returnId = id;
      overlap = thisOverlap;
    }
  }
  if(returnId==frame) {
    OKVIS_THROW(Exception, "most overlapped frame with itself: makes no sense")
  }
  return returnId;
}

StateId ViSlamBackend::currentLoopclosureStateId() const
{
  StateId currentFrame = currentStateId();
  StateId returnId;
  double overlap = 0.0;
  for(auto id : loopClosureFrames_) {
    if(id==currentFrame) {
      continue;
    }
    if(!realtimeGraph_.states_.at(id).isKeyframe) {
      continue;
    }
    double thisOverlap = overlapFraction(multiFrames_.at(id), multiFrames_.at(currentFrame));
    if(thisOverlap >= overlap) {
      returnId = id;
      overlap = thisOverlap;
    }
  }
  if(returnId==currentFrame) {
    OKVIS_THROW(Exception, "current frame is loopclosure frame: makes no sense")
  }
  if(overlap>0.5) {
    return returnId;
  }
  return StateId();
}

int ViSlamBackend::prunePlaceRecognitionFrames() {
  realtimeGraph_.computeCovisibilities();
  std::set<StateId> allFrames;
  allFrames.insert(keyFrames_.begin(), keyFrames_.end());
  allFrames.insert(loopClosureFrames_.begin(), loopClosureFrames_.end());
  int ctr=0;
  if(keyFrames_.size() == 0) {
    return 0;
  }
  StateId id0 = *keyFrames_.rbegin();
  for(auto id1 : allFrames) {
    if(id0.value()<=id1.value()) {
      continue;
    }
    if(!auxiliaryStates_.at(id0).isPlaceRecognitionFrame) {
      continue;
    }
    if(!auxiliaryStates_.at(id1).isPlaceRecognitionFrame) {
      continue;
    }
    if(realtimeGraph_.covisibilities(id0, id1) < 10) {
      continue;
    }
    const double overlap = overlapFraction(multiFrames_.at(id0), multiFrames_.at(id1));
    if(overlap > 0.6) {
      ctr++;
      // prune the newer frame
      auxiliaryStates_.at(id0).isPlaceRecognitionFrame = false;
      break;
    }
  }
  return ctr;
}

void ViSlamBackend::clear()
{
  // clear underlying graphs
  realtimeGraph_.clear();
  fullGraph_.clear();

  multiFrames_.clear();

  auxiliaryStates_.clear(); // Store information about states.
  currentComponentIdx_ = 0; // The index of the current component.

  loopClosureFrames_.clear(); // All the current loop closure frames.

  imuFrames_.clear(); // All the current IMU frames.
  keyFrames_.clear(); // All the current keyframes.

  needsFullGraphOptimisation_ = false;
  isLoopClosing_ = false;
  isLoopClosureAvailable_ = false;
  components_.resize(1);

  addStatesBacklog_.clear(); // Backlog of states to add to fullGraph_.
  eliminateStates_.clear(); // States eliminated in realtimeGraph_.
  touchedStates_.clear(); // States modified in realtimeGraph_.
  touchedLandmarks_.clear(); // Landmarks modified in realtimeGraph_.

  fullGraphRelativePoseConstraints_.clear(); // Relative pose constraints.

  lastFreeze_ = StateId(); // Store up to where the realtimeGraph_ states were fixed.
}

double ViSlamBackend::overlapFraction(const MultiFramePtr frameA,
                                      const MultiFramePtr frameB) const {

  OKVIS_ASSERT_TRUE(Exception, frameA->numFrames() == frameB->numFrames(),
                    "must be same number of frames")
  const size_t numFrames = frameA->numFrames();
  const MultiFramePtr frames[2] = {frameA, frameB};

  std::set<LandmarkId> landmarks[2];
  std::vector<cv::Mat> detectionsImg[2];
  detectionsImg[0].resize(numFrames);
  detectionsImg[1].resize(numFrames);
  std::vector<cv::Mat> matchesImg[2];
  matchesImg[0].resize(numFrames);
  matchesImg[1].resize(numFrames);

  // paint detection images and remember matched points
  for(size_t f=0; f<2; ++f) {
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      const int rows = frames[f]->image(im).rows/10;
      const int cols = frames[f]->image(im).cols/10;
      const double radius = double(std::min(rows,cols))*kptradius_;
      detectionsImg[f].at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
      matchesImg[f].at(im) = cv::Mat::zeros(rows, cols, CV_8UC1);
      const size_t num = frames[f]->numKeypoints(im);
      cv::KeyPoint keypoint;
      for (size_t k = 0; k < num; ++k) {
        frames[f]->getCvKeypoint(im, k, keypoint);
        cv::circle(detectionsImg[f].at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                   cv::FILLED);
        uint64_t lmId = frames[f]->landmarkId(im, k);
        if (lmId != 0) {
          landmarks[f].insert(LandmarkId(lmId));
        }
      }
    }
  }

  // find matches
  std::set<LandmarkId> matches;
  std::set_intersection(
      landmarks[0].begin(), landmarks[0].end(),landmarks[1].begin(),
      landmarks[1].end(), std::inserter(matches,matches.begin()));

  // without matches there will be no overlap
  if(matches.size() == 0) {
    return 0.0;
  }

  // draw match images
  for(size_t f=0; f<2; ++f) {
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      cv::KeyPoint keypoint;
      const size_t num = frames[f]->numKeypoints(im);
      const int rows = frames[f]->image(im).rows/10;
      const int cols = frames[f]->image(im).cols/10;
      const double radius = double(std::min(rows,cols))*kptradius_;
      for (size_t k = 0; k < num; ++k) {
        frames[f]->getCvKeypoint(im, k, keypoint);
        if (matches.count(LandmarkId(frames[f]->landmarkId(im, k)))) {
          cv::circle(matchesImg[f].at(im), keypoint.pt*0.1, int(radius), cv::Scalar(255),
                     cv::FILLED);
        }
      }
    }
  }

  // IoU
  double overlap[2];
  for(size_t f=0; f<2; ++f) {
    int intersectionCount = 0;
    int unionCount = 0;
    for (size_t im = 0; im < frames[f]->numFrames(); ++im) {
      if(frames[f]->image(im).empty()) continue;
      cv::Mat intersectionMask, unionMask;
      cv::bitwise_and(matchesImg[f].at(im), detectionsImg[f].at(im), intersectionMask);
      cv::bitwise_or(matchesImg[f].at(im), detectionsImg[f].at(im), unionMask);
      intersectionCount += cv::countNonZero(intersectionMask);
      unionCount += cv::countNonZero(unionMask);
    }
    overlap[f] = double(intersectionCount)/double(unionCount);
  }

  return std::min(overlap[0], overlap[1]);
}

}  // namespace okvis
