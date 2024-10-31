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
 * @file ViGraph.cpp
 * @brief Source file for the ViGraph class.
 * @author Stefan Leutenegger
 */

#include <okvis/ViGraph.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Add a camera to the configuration. Sensors can only be added and never removed.
ViGraph::ViGraph()
{
  cauchyLossFunctionPtr_.reset(new ::ceres::CauchyLoss(1.0));
  ::ceres::Problem::Options problemOptions;
  problemOptions.manifold_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership =
      ::ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.enable_fast_removal = true;
  problem_.reset(new ::ceres::Problem(problemOptions));
  options_.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
  options_.trust_region_strategy_type = ::ceres::DOGLEG;
  //options_.dense_linear_algebra_library_type = ::ceres::LAPACK; // somehow slow on Jetson...
}

int ViGraph::addCamera(const CameraParameters& cameraParameters) {
  cameraParametersVec_.push_back(cameraParameters);
  return static_cast<int>(cameraParametersVec_.size()) - 1;
}

// Add an IMU to the configuration.
int ViGraph::addImu(const ImuParameters& imuParameters) {
  if (imuParametersVec_.size() > 1) {
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.push_back(imuParameters);
  return static_cast<int>(imuParametersVec_.size()) - 1;
}

StateId ViGraph::addStatesInitialise(
    const Time &timestamp, const ImuMeasurementDeque &imuMeasurements,
    const cameras::NCameraSystem & nCameraSystem)
{
  State state;
  state.timestamp = timestamp;
  state.isKeyframe = true; // first one must be keyframe.
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.size()==0, "states added before...")
  StateId id(1);

  // set translation to zero, unit rotation
  kinematics::Transformation T_WS;
  T_WS.setIdentity();

  if (imuParametersVec_.at(0).use) {
    OKVIS_ASSERT_TRUE_DBG(Exception, imuMeasurements.size() > 0, "no IMU measurements passed")

    // acceleration vector
    Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
    for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
         it < imuMeasurements.end();
         ++it) {
      acc_B += it->measurement.accelerometers;
    }
    acc_B /= double(imuMeasurements.size());
    Eigen::Vector3d e_acc = acc_B.normalized();

    // align with ez_W:
    Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
    Eigen::Matrix<double, 6, 1> poseIncrement;
    poseIncrement.head<3>() = Eigen::Vector3d::Zero();
    poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
    double angle = std::acos(ez_W.transpose() * e_acc);
    poseIncrement.tail<3>() *= angle;
    T_WS.oplus(-poseIncrement);
  } else {
    // otherwise we assume the camera is vertical & upright
    kinematics::Transformation T_WC;
    T_WC.set(T_WC.r(), T_WC.q() * Eigen::Quaterniond(-sqrt(2), sqrt(2), 0, 0));
    T_WS = T_WC * nCameraSystem.T_SC(0)->inverse();
  }

  // now set/add states
  state.pose.reset(new ceres::PoseParameterBlock(T_WS, id.value(), timestamp));
  SpeedAndBias speedAndBias = SpeedAndBias::Zero();
  speedAndBias.tail<3>() = imuParametersVec_.at(0).a0;
  speedAndBias.segment<3>(3) = imuParametersVec_.at(0).g0;
  state.speedAndBias.reset(
        new ceres::SpeedAndBiasParameterBlock(speedAndBias, id.value(), timestamp));
  problem_->AddParameterBlock(state.pose->parameters(), 7, &poseManifold_);
  state.pose->setLocalParameterizationPtr(&poseManifold_);
  problem_->AddParameterBlock(state.speedAndBias->parameters(), 9);
  for(size_t i = 0; i<cameraParametersVec_.size(); ++i) {
    const kinematics::Transformation T_SC = *nCameraSystem.T_SC(i);
    state.extrinsics.push_back(std::shared_ptr<ceres::PoseParameterBlock>(
                                 new ceres::PoseParameterBlock(T_SC, id.value(), timestamp)));
    problem_->AddParameterBlock(state.extrinsics.back()->parameters(), 7,
                                &poseManifold_);
    state.extrinsics.back()->setLocalParameterizationPtr(&poseManifold_);
  }

  // add the priors
  Eigen::Matrix<double, 6, 1> informationDiag = Eigen::Matrix<double, 6, 1>::Ones();
  informationDiag[0] = 1.0e8;
  informationDiag[1] = 1.0e8;
  informationDiag[2] = 1.0e8;
  if(imuParametersVec_.at(0).use) {
    // yaw and pitch not fixed
    informationDiag[3] = 0.0;
    informationDiag[4] = 0.0;
  } else {
    // yaw and pitch fixed
    informationDiag[3] = 1.0e2;
    informationDiag[4] = 1.0e2;
  }
  informationDiag[5] = 1.0e2;
  state.posePrior.errorTerm.reset(new ceres::PoseError(T_WS, informationDiag));
  const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
  const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
  state.speedAndBiasPrior.errorTerm.reset(
        new ceres::SpeedAndBiasError(speedAndBias, 0.1, sigma_bg * sigma_bg, sigma_ba * sigma_ba));
  state.posePrior.residualBlockId = problem_->AddResidualBlock(
        state.posePrior.errorTerm.get(), nullptr, state.pose->parameters());
  state.speedAndBiasPrior.residualBlockId = problem_->AddResidualBlock(
        state.speedAndBiasPrior.errorTerm.get(), nullptr, state.speedAndBias->parameters());
  for(size_t i = 0; i<cameraParametersVec_.size(); ++i) {
    if(cameraParametersVec_.at(i).online_calibration.do_extrinsics) {
      // add a pose prior
      PosePrior extrinsicsPrior;
      const double sigma_r = cameraParametersVec_.at(i).online_calibration.sigma_r;
      const double sigma_alpha = cameraParametersVec_.at(i).online_calibration.sigma_alpha;
      extrinsicsPrior.errorTerm.reset(
            new ceres::PoseError(
              state.extrinsics.at(i)->estimate(), sigma_r*sigma_r, sigma_alpha*sigma_alpha));
      extrinsicsPrior.residualBlockId = problem_->AddResidualBlock(
                extrinsicsPrior.errorTerm.get(), nullptr, state.extrinsics.at(i)->parameters());
      state.extrinsicsPriors.push_back(extrinsicsPrior);
    } else {
      // simply fix
      problem_->SetParameterBlockConstant(state.extrinsics.at(i)->parameters());
      state.extrinsics.at(i)->setFixed(true);
    }
  }

  states_[id] = state; // actually add...
  AnyState anyState;
  anyState.timestamp = state.timestamp;
  anyState.T_Sk_S = kinematics::Transformation::Identity();
  anyState.v_Sk = Eigen::Vector3d::Zero();
  anyState_[id] = anyState; // add, never remove.

  return id;
}

StateId ViGraph::addStatesPropagate(const Time &timestamp,
                                     const ImuMeasurementDeque &imuMeasurements, bool isKeyframe)
{
  // propagate state
  State state;
  state.timestamp = timestamp;
  state.isKeyframe = isKeyframe;
  StateId id(states_.rbegin()->first+1);
  State & lastState = states_.rbegin()->second;
  kinematics::Transformation T_WS = lastState.pose->estimate();
  SpeedAndBias speedAndBias = lastState.speedAndBias->estimate();
  if(imuParametersVec_.at(0).use) {
    ceres::ImuError::propagation(imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,
                                 lastState.timestamp, timestamp);
  } else {
    // try and apply constant velocity model
    auto iter = states_.rbegin();
    ++iter;
    if (iter != states_.rend()) {
      const double r = (timestamp - lastState.timestamp).toSec()
                       / (lastState.timestamp - iter->second.timestamp).toSec();
      kinematics::Transformation T_WS_m1 = iter->second.pose->estimate();
      Eigen::Vector3d dr = r * (T_WS.r() - T_WS_m1.r());
      Eigen::AngleAxisd daa(T_WS.q() * T_WS_m1.q().inverse());
      daa.angle() *= r;
      T_WS.set(T_WS.r() + dr, T_WS.q() * Eigen::Quaterniond(daa));
      //LOG(WARNING) << "---\n" << T_WS_m1.T() << "\n" << T_WS.T() << "\n===";
      //LOG(INFO) << dr.norm() << " : " << 2.0 * acos(dq.w());
    }
  }
  state.pose.reset(new ceres::PoseParameterBlock(T_WS, id.value(), timestamp));
  problem_->AddParameterBlock(state.pose->parameters(), 7, &poseManifold_);
  state.pose->setLocalParameterizationPtr(&poseManifold_);
  state.speedAndBias.reset(new ceres::SpeedAndBiasParameterBlock(
                             speedAndBias, id.value(), timestamp));
  problem_->AddParameterBlock(state.speedAndBias->parameters(), 9);

  // create IMU link
  ImuLink imuLink;
  if(imuParametersVec_.at(0).use) {
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          lastState.timestamp > imuMeasurements.front().timeStamp,
                          "Cannot propagate IMU: " << lastState.timestamp << ">"
                                                   << imuMeasurements.front().timeStamp)
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          timestamp < imuMeasurements.back().timeStamp,
                          "Cannot propagate IMU: " << timestamp << "<"
                                                   << imuMeasurements.back().timeStamp)
    imuLink.errorTerm.reset(new ceres::ImuError(imuMeasurements, imuParametersVec_.at(0),
                                                lastState.timestamp, timestamp));
  } else {
    imuLink.errorTerm.reset(new ceres::PseudoImuError(lastState.timestamp, timestamp));
  }
  // add to ceres
  imuLink.residualBlockId = problem_->AddResidualBlock(
    imuLink.errorTerm.get(), nullptr,
    lastState.pose->parameters(), lastState.speedAndBias->parameters(),
    state.pose->parameters(), state.speedAndBias->parameters());

  //OKVIS_ASSERT_TRUE(
  //  Exception, okvis::ceres::jacobiansCorrect(problem_.get(), imuLink.residualBlockId),
  //  "Jacobian verification failed")

  // store IMU link
  lastState.nextImuLink = imuLink;
  state.previousImuLink = imuLink;

  // propagate extrinsics (if needed)
  for(size_t i = 0; i<cameraParametersVec_.size(); ++i) {
    // re-use same extrinsics
    state.extrinsics.push_back(lastState.extrinsics.at(i));
  }

  states_[id] = state; // actually add...
  AnyState anyState;
  anyState.timestamp = state.timestamp;
  anyState.T_Sk_S = kinematics::Transformation::Identity();
  anyState.v_Sk = Eigen::Vector3d::Zero();
  anyState_[id] = anyState; // add, never remove.

  return id;
}

bool ViGraph::addStatesFromOther(StateId stateId, const ViGraph &other)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, other.states_.count(stateId), "stateId not found")

  State state;
  const State& otherState = other.states_.at(stateId);
  state.timestamp = otherState.timestamp;
  state.isKeyframe = otherState.isKeyframe;

  // clone states
  const size_t numCameras = otherState.extrinsics.size();
  state.extrinsics.resize(numCameras);

  // create all new from otherState
  state.pose.reset(new ceres::PoseParameterBlock(
                     otherState.pose->estimate(), stateId.value(), otherState.timestamp));
  state.speedAndBias.reset(new ceres::SpeedAndBiasParameterBlock(
                     otherState.speedAndBias->estimate(), stateId.value(), otherState.timestamp));
  for(size_t i = 0; i < numCameras; ++i) {
    state.extrinsics.at(i).reset(
          new ceres::PoseParameterBlock(otherState.extrinsics.at(i)->estimate(),
                                        stateId.value(), otherState.timestamp));
  }

  // handle priors, if any
  if(otherState.posePrior.errorTerm) {
    PosePrior& posePrior = state.posePrior;
    posePrior.errorTerm.reset(new ceres::PoseError(otherState.posePrior.errorTerm->measurement(),
                                                   otherState.posePrior.errorTerm->information()));
    posePrior.residualBlockId = problem_->AddResidualBlock(posePrior.errorTerm.get(),
                                                           nullptr, state.pose->parameters());
  }
  if(otherState.speedAndBiasPrior.errorTerm) {
    SpeedAndBiasPrior& speedAndBiasPrior = state.speedAndBiasPrior;
    speedAndBiasPrior.errorTerm.reset(
          new ceres::SpeedAndBiasError(otherState.speedAndBiasPrior.errorTerm->measurement(),
                                       otherState.speedAndBiasPrior.errorTerm->information()));
    speedAndBiasPrior.residualBlockId = problem_->AddResidualBlock(
          speedAndBiasPrior.errorTerm.get(), nullptr, state.speedAndBias->parameters());
  }
  for(size_t i=0; i<otherState.extrinsicsPriors.size(); ++i) {
    PosePrior extrinsicsPrior;
    const PosePrior& otherExtrinsicsPrior = state.extrinsicsPriors.at(i);
    extrinsicsPrior.errorTerm.reset(
          new ceres::PoseError(otherExtrinsicsPrior.errorTerm->measurement(),
                               otherExtrinsicsPrior.errorTerm->information()));
    extrinsicsPrior.residualBlockId = problem_->AddResidualBlock(
          extrinsicsPrior.errorTerm.get(), nullptr, state.extrinsics.at(i)->parameters());
    state.extrinsicsPriors.push_back(extrinsicsPrior);
  }

  // handle links, if any
  if(otherState.previousImuLink.errorTerm) {
    auto otherIter = other.states_.find(stateId);
    OKVIS_ASSERT_TRUE_DBG(Exception, otherIter != other.states_.begin(), "no previous state")
    otherIter--;
    const State& otherPreviousState = otherIter->second;
    const Time t_0 = otherPreviousState.timestamp;
    const Time t_1 = otherState.timestamp;
    State& previousState = states_.rbegin()->second;
    OKVIS_ASSERT_TRUE_DBG(Exception, otherIter->first == states_.rbegin()->first,
                      "different previous states")
    OKVIS_ASSERT_TRUE_DBG(Exception, t_0 == previousState.timestamp, "inconsistent previous times")
    ImuLink imuLink;
    imuLink.errorTerm = otherState.previousImuLink.errorTerm->clone();
    imuLink.residualBlockId = problem_->AddResidualBlock(
          imuLink.errorTerm.get(), nullptr,
          previousState.pose->parameters(), previousState.speedAndBias->parameters(),
          state.pose->parameters(), state.speedAndBias->parameters());
    state.previousImuLink = imuLink;
    previousState.nextImuLink = imuLink;

    // and possibly relative camera poses
    for(size_t i=0; i<otherState.previousExtrinsicsLink.size(); ++i) {
      ExtrinsicsLink extrinsicsLink;
      const ceres::RelativePoseError& otherRelativePoseError =
          *otherState.previousExtrinsicsLink.at(i).errorTerm;
      extrinsicsLink.errorTerm.reset(
            new ceres::RelativePoseError(otherRelativePoseError.information()));
      extrinsicsLink.residualBlockId = problem_->AddResidualBlock(
            extrinsicsLink.errorTerm.get(), nullptr, previousState.extrinsics.at(i)->parameters(),
            state.extrinsics.at(i)->parameters());
      state.previousExtrinsicsLink.push_back(extrinsicsLink);
      previousState.nextExtrinsicsLink.push_back(extrinsicsLink);
    }
  }

  states_[stateId] = state; // actually add...
  AnyState anyState;
  anyState.timestamp = state.timestamp;
  anyState.T_Sk_S = kinematics::Transformation::Identity();
  anyState.v_Sk = Eigen::Vector3d::Zero();
  anyState_[stateId] = anyState; // add, never remove.

  return true;
}

bool ViGraph::addLandmark(LandmarkId landmarkId, const Eigen::Vector4d &homogeneousPoint,
                           bool initialised)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarkId.isInitialised(), "landmark ID invalid")
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId) == 0, "landmark already exists")
  Landmark landmark;
  landmark.hPoint.reset(new ceres::HomogeneousPointParameterBlock(
                          homogeneousPoint, landmarkId.value(), initialised));
  problem_->AddParameterBlock(landmark.hPoint->parameters(), 4,
                              &homogeneousPointManifold_);
  landmark.hPoint->setLocalParameterizationPtr(&homogeneousPointManifold_);
  landmarks_[landmarkId] = landmark;
  return true;
}

LandmarkId ViGraph::addLandmark(const Eigen::Vector4d &homogeneousPoint, bool initialised)
{
  const LandmarkId landmarkId = landmarks_.empty() ?
        LandmarkId(1) : LandmarkId(landmarks_.rbegin()->first+1); // always increase highest ID by 1
  Landmark landmark;
  landmark.hPoint.reset(new ceres::HomogeneousPointParameterBlock(
                          homogeneousPoint, landmarkId.value(), initialised));
  problem_->AddParameterBlock(landmark.hPoint->parameters(), 4,
                              &homogeneousPointManifold_);
  landmark.hPoint->setLocalParameterizationPtr(&homogeneousPointManifold_);
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId) == 0, "Bug: landmark not added")
  landmarks_[landmarkId] = landmark;
  return landmarkId;
}

bool ViGraph::removeLandmark(LandmarkId landmarkId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  Landmark& landmark = landmarks_.at(landmarkId);

  // remove all observations
  for(auto observation : landmark.observations) {
    problem_->RemoveResidualBlock(observation.second.residualBlockId);
    if (observation.second.depthError.residualBlockId) {
      problem_->RemoveResidualBlock(observation.second.depthError.residualBlockId);
    }
    observations_.erase(observation.first);
    // also remove it in the state
    StateId stateId(observation.first.frameId);
    states_.at(stateId).observations.erase(observation.first);
  }

  // now remove the landmark itself
  problem_->RemoveParameterBlock(landmark.hPoint->parameters());
  landmarks_.erase(landmarkId);
  return true;
}

bool ViGraph::setLandmarkInitialised(LandmarkId landmarkId, bool initialised)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  landmarks_.at(landmarkId).hPoint->setInitialized(initialised);
  return true;
}

bool ViGraph::isLandmarkInitialised(LandmarkId landmarkId) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  return landmarks_.at(landmarkId).hPoint->initialized();
}

bool ViGraph::isLandmarkAdded(LandmarkId landmarkId) const
{
  return landmarks_.count(landmarkId)>0;
}

void ViGraph::checkObservations() const {

  // check by overall observations
  for(auto obs : observations_) {
    OKVIS_ASSERT_TRUE(Exception,
                      landmarks_.at(obs.second.landmarkId).observations.count(obs.first),
                      "observations check failed")
    OKVIS_ASSERT_TRUE(Exception,
                      states_.at(StateId(obs.first.frameId)).observations.count(obs.first),
                      "observations check failed")
  }

  // check by states
  for(auto state : states_) {
    for(auto obs : state.second.observations) {
      OKVIS_ASSERT_TRUE(Exception,
                        landmarks_.at(obs.second.landmarkId).observations.count(obs.first),
                        "observations check failed")
      OKVIS_ASSERT_TRUE(Exception, observations_.count(obs.first), "observations check failed")
    }
  }

  // check by landmarks
  for(auto lm : landmarks_) {
    for(auto obs : lm.second.observations) {
      OKVIS_ASSERT_TRUE(Exception, observations_.count(obs.first), "observations check failed")
      OKVIS_ASSERT_TRUE(Exception,
                        states_.at(StateId(obs.first.frameId)).observations.count(obs.first),
                        "observations check failed");
    }
  }

}

bool ViGraph::removeObservation(KeypointIdentifier keypointId)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, observations_.count(keypointId), "observation does not exists")
  Observation observation = observations_.at(keypointId);

  // remove in ceres
  problem_->RemoveResidualBlock(observation.residualBlockId);
  if (observation.depthError.residualBlockId) {
    problem_->RemoveResidualBlock(observation.depthError.residualBlockId);
  }

  // remove everywhere in bookkeeping
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(observation.landmarkId),
                        "landmark does not exists")
  Landmark& landmark = landmarks_.at(observation.landmarkId);
  OKVIS_ASSERT_TRUE_DBG(Exception, landmark.observations.count(keypointId),
                    "observation does not exists")
  landmark.observations.erase(keypointId);
  StateId stateId(keypointId.frameId);
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(stateId),
                    "state does not exists")
  State& state = states_.at(stateId);
  OKVIS_ASSERT_TRUE_DBG(Exception, state.observations.count(keypointId),
                    "observation does not exists")
  state.observations.erase(keypointId);
  observations_.erase(keypointId);

  // covisibilities invalid
  covisibilitiesComputed_ = false;

  return true;
}

bool ViGraph::computeCovisibilities()
{
  if(covisibilitiesComputed_) {
    return true; // already done previously
  }
  coObservationCounts_.clear();
  visibleFrames_.clear();
  for(auto iter=landmarks_.begin(); iter!=landmarks_.end(); ++iter) {
    if(iter->second.classification == 10 || iter->second.classification == 11) {
      continue;
    }
    auto obs = iter->second.observations;
    std::set<uint64> covisibilities;
    for(auto obsiter=obs.begin(); obsiter!=obs.end(); ++obsiter) {
      covisibilities.insert(obsiter->first.frameId);
      visibleFrames_.insert(StateId(obsiter->first.frameId));
    }
    for(auto i0=covisibilities.begin(); i0!=covisibilities.end(); ++i0) {
      for(auto i1=covisibilities.begin(); i1!=covisibilities.end(); ++i1) {
        if(*i1>=*i0) {
          continue;
        }
        if(coObservationCounts_.find(*i0)==coObservationCounts_.end()) {
          coObservationCounts_[*i0][*i1] = 1;
        } else {
          if (coObservationCounts_.at(*i0).find(*i1)==coObservationCounts_.at(*i0).end()) {
            coObservationCounts_.at(*i0)[*i1] = 1;
          } else {
            coObservationCounts_.at(*i0).at(*i1)++;
          }
        }
      }
    }
  }
  covisibilitiesComputed_ = true;
  return coObservationCounts_.size()>0;
}



int ViGraph::covisibilities(StateId pose_i, StateId pose_j) const
{
  OKVIS_ASSERT_TRUE(Exception, covisibilitiesComputed_, "covisibilities not yet computed")
  if(pose_i==pose_j) {
    return 0;
  }
  size_t a=pose_i.value();
  size_t b=pose_j.value();
  if(pose_i < pose_j) {
    b=pose_i.value();
    a=pose_j.value();
  }
  if(coObservationCounts_.count(a)) {
    if(coObservationCounts_.at(a).count(b)) {
      return coObservationCounts_.at(a).at(b);
    }
  }
  return 0;
}

bool ViGraph::addRelativePoseConstraint(StateId poseId0, StateId poseId1,
                                         const kinematics::Transformation &T_S0S1,
                                         const Eigen::Matrix<double, 6, 6> &information)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId0), "stateId not found")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId1), "stateId not found")
  State & state0 = states_.at(poseId0);
  State & state1 = states_.at(poseId1);
  OKVIS_ASSERT_TRUE_DBG(Exception, state0.relativePoseLinks.count(poseId1)==0,
                    "relative pose error already exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, state1.relativePoseLinks.count(poseId0)==0,
                    "relative pose error already exists")
  RelativePoseLink relativePoseLink;
  relativePoseLink.errorTerm.reset(new ceres::RelativePoseError(information, T_S0S1));
  relativePoseLink.residualBlockId = problem_->AddResidualBlock(relativePoseLink.errorTerm.get(),
                                                                nullptr, state0.pose->parameters(),
                                                                state1.pose->parameters());
  relativePoseLink.state0 = poseId0;
  relativePoseLink.state1 = poseId1;
  // add to book-keeping
  state0.relativePoseLinks[poseId1] = relativePoseLink;
  state1.relativePoseLinks[poseId0] = relativePoseLink;

  return true;
}

bool ViGraph::removeRelativePoseConstraint(StateId poseId0, StateId poseId1)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId0),
                        "stateId " << poseId0.value() << " not found")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(poseId1),
                        "stateId " << poseId1.value() << "not found")
  State & state0 = states_.at(poseId0);
  State & state1 = states_.at(poseId1);
  OKVIS_ASSERT_TRUE_DBG(Exception, state0.relativePoseLinks.count(poseId1)==1,
                    "relative pose error does not exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, state1.relativePoseLinks.count(poseId0)==1,
                    "relative pose error does not exists")
  RelativePoseLink & relativePoseLink = state0.relativePoseLinks.at(poseId1);
  problem_->RemoveResidualBlock(relativePoseLink.residualBlockId);

  // add to book-keeping
  state0.relativePoseLinks.erase(poseId1);
  state1.relativePoseLinks.erase(poseId0);

  return true;
}

const Eigen::Vector4d &ViGraph::landmark(LandmarkId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(id), "landmark does not exists")
  return landmarks_.at(id).hPoint->estimate();
}

bool ViGraph::getLandmark(LandmarkId landmarkId, MapPoint2& mapPoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(landmarkId), "landmark does not exists")
  std::set<KeypointIdentifier> observations;
  const Landmark & landmark = landmarks_.at(landmarkId);
  for(auto observation : landmark.observations) {
    observations.insert(observation.first);
  }
  mapPoint = MapPoint2{landmarkId, landmark.hPoint->estimate(), observations,
      landmark.hPoint->initialized(), landmark.quality, landmark.classification};
  return true;
}

size_t ViGraph::getLandmarks(MapPoints &landmarks) const
{
  for(auto& landmark : landmarks_) {
    std::set<KeypointIdentifier> observations;
    for(auto observation : landmark.second.observations) {
      observations.insert(observation.first);
    }
    landmarks[landmark.first] = MapPoint2{
        landmark.first, landmark.second.hPoint->estimate(), observations,
        landmark.second.hPoint->initialized(), landmark.second.quality,
        landmark.second.classification};
  }
  return landmarks.size();
}

bool ViGraph::landmarkExists(LandmarkId landmarkId) const
{
  return landmarks_.count(landmarkId) != 0;
}

bool ViGraph::setLandmark(LandmarkId id, const Eigen::Vector4d &homogeneousPoint, bool initialised)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(id), "landmark does not exists")
  auto& landmark = landmarks_.at(id).hPoint;
  landmark->setEstimate(homogeneousPoint);
  landmark->setInitialized(initialised);
  return true;
}
bool ViGraph::setLandmark(LandmarkId id, const Eigen::Vector4d &homogeneousPoint)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, landmarks_.count(id), "landmark does not exists")
  auto& landmark = landmarks_.at(id).hPoint;
  landmark->setEstimate(homogeneousPoint);
  return true;
}

bool ViGraph::areLandmarksInFrontOfCameras() const
{
  for (const auto &obs : observations_) {
    Eigen::Vector4d hp_W = landmark(obs.second.landmarkId);
    kinematics::Transformation T_WS, T_SCi;
    const StateId stateId(obs.first.frameId);
    const State &state = states_.at(stateId);
    T_WS = state.pose->estimate();
    T_SCi = state.extrinsics.at(obs.first.cameraIndex)->estimate();
    kinematics::Transformation T_WCi = T_WS * T_SCi;
    Eigen::Vector4d pos_Ci = T_WCi.inverse() * hp_W;
    if (pos_Ci[2] * pos_Ci[3] < 0.0) {
      return false;
    }
    if (fabs(pos_Ci[3]) > 1.0e-16) {
      if (pos_Ci[2] / pos_Ci[3] < 0.05) {
        return false;
      }
    }
  }
  return true;
}

const kinematics::TransformationCacheless & ViGraph::pose(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).pose->estimate();
}

const SpeedAndBias &ViGraph::speedAndBias(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).speedAndBias->estimate();
}

const kinematics::TransformationCacheless &ViGraph::extrinsics(StateId id, uchar camIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(id).extrinsics.at(camIdx), "extrinsics do not exists")
      return states_.at(id).extrinsics.at(camIdx)->estimate();
}

bool ViGraph::isKeyframe(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).isKeyframe;
}


Time ViGraph::timestamp(StateId id) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  return states_.at(id).timestamp;
}

StateId ViGraph::currentStateId() const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, !states_.empty(), "no states exist")
  return states_.rbegin()->first;
}

StateId ViGraph::stateIdByAge(size_t age) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.size()>age, "too few states exist")
  auto iter = states_.rbegin();
  for(size_t k=0; k<age; k++) {
    iter++;
  }
  return iter->first;
}

bool ViGraph::setKeyframe(StateId id, bool isKeyframe)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  states_.at(id).isKeyframe = isKeyframe;
  return true;
}

bool ViGraph::setPose(StateId id, const kinematics::TransformationCacheless &pose)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  states_.at(id).pose->setEstimate(pose);
  return true;
}

bool ViGraph::setSpeedAndBias(StateId id, const SpeedAndBias &speedAndBias)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  states_.at(id).speedAndBias->setEstimate(speedAndBias);
  return true;
}

bool ViGraph::setExtrinsics(StateId id, uchar camIdx,
                            const kinematics::TransformationCacheless &extrinsics) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.count(id), "state does not exists")
  OKVIS_ASSERT_TRUE_DBG(Exception, states_.at(id).extrinsics.at(camIdx), "extrinsics do not exists")
  states_.at(id).extrinsics.at(camIdx)->setEstimate(extrinsics);
  return true;
}

bool ViGraph::setExtrinsicsVariable()
{
  for(size_t i=0; i<cameraParametersVec_.size(); ++i) {
    problem_->SetParameterBlockVariable(states_.begin()->second.extrinsics.at(i)->parameters());
  }
  return true;
}

bool ViGraph::softConstrainExtrinsics(double posStd, double rotStd)
{
  if(states_.begin()->second.extrinsicsPriors.size() == cameraParametersVec_.size())
  for(size_t i=0; i<cameraParametersVec_.size(); ++i) {
    problem_->RemoveResidualBlock(states_.begin()->second.extrinsicsPriors.at(i).residualBlockId);
  }
  states_.begin()->second.extrinsicsPriors.resize(cameraParametersVec_.size());
  for(size_t i=0; i<cameraParametersVec_.size(); ++i) {
    // add a pose prior
    PosePrior& extrinsicsPrior = states_.begin()->second.extrinsicsPriors.at(i);
    extrinsicsPrior.errorTerm.reset(
          new ceres::PoseError(
            states_.begin()->second.extrinsics.at(i)->estimate(), posStd*posStd, rotStd*rotStd));
    extrinsicsPrior.residualBlockId = problem_->AddResidualBlock(
              extrinsicsPrior.errorTerm.get(), nullptr,
          states_.begin()->second.extrinsics.at(i)->parameters());

  }
  return true;
}

//static int behindCtr = 0;

void ViGraph::updateLandmarks()
{
  for (auto it = landmarks_.begin(); it != landmarks_.end(); ++it) {
    Eigen::Vector4d hp_W = it->second.hPoint->estimate();
    const size_t num = it->second.observations.size();
    bool isInitialised = false;
    double quality = 0.0;
    bool behind = false;
    double best_err = 1.0e12;
    Eigen::Vector4d best_pos(0.0, 0.0, 0.0, 0.0);
    double minD = 1.0e12;
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    if (num > 0) {
      for (const auto &observation : it->second.observations) {
        kinematics::Transformation T_WS, T_SCi;
        const StateId stateId(observation.first.frameId);
        const State &state = states_.at(stateId);
        T_WS = state.pose->estimate();
        T_SCi = state.extrinsics.at(observation.first.cameraIndex)->estimate();
        kinematics::Transformation T_WCi = T_WS * T_SCi;
        Eigen::Vector4d pos_Ci = T_WCi.inverse() * hp_W;
        Eigen::Vector3d dir_W = (T_WCi.C() * pos_Ci.head<3>()).normalized();

        if (fabs(pos_Ci[3]) > 1.0e-12) {
          pos_Ci = pos_Ci / pos_Ci[3];
        }
        if (pos_Ci[2] < 0.1) {
          behind = true;
        }
        if (pos_Ci[2] < 0.0) {
          dir_W = -dir_W; // reverse!!
        }

        // consider only small reprojection errors
        Eigen::Vector2d err;
        double *params[3];
        params[0] = state.pose->parameters();
        params[1] = it->second.hPoint->parameters();
        params[2] = state.extrinsics.at(observation.first.cameraIndex)->parameters();
        double *jacobians[3];
        double *jacobians_minimal[3];
        Eigen::Matrix<double, 2, 4> J1;
        Eigen::Matrix<double, 2, 3> J1_minimal;
        jacobians[0] = nullptr;
        jacobians[1] = J1.data();
        jacobians[2] = nullptr;
        jacobians_minimal[0] = nullptr;
        jacobians_minimal[1] = J1_minimal.data();
        jacobians_minimal[2] = nullptr;
        observation.second.errorTerm->EvaluateWithMinimalJacobians(params,
                                                                   err.data(),
                                                                   jacobians,
                                                                   jacobians_minimal);
        const double err_norm = err.norm();

        if (err_norm > 2.5) {
          //bad = true;
          continue;
        }

        if (fabs(pos_Ci[3]) > 1.0e-12) {
          if (pos_Ci[2] < minD) {
            minD = pos_Ci[2];
          }
        }

        if (err_norm < best_err && pos_Ci.norm() > 0.0001) {
          // remember best fit
          // if it was far away, leave it far away; but make sure it's in front
          // of the camera and at least at 10 cm...
          const double dist = std::max(0.1, pos_Ci.norm());
          best_pos.head<3>() = T_WCi.r() + dist * dir_W;
          best_pos[3] = 1.0;
          best_err = err_norm;
        }

        H += J1_minimal.transpose() * J1_minimal;
      }

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
      es.compute(H);
      const double s0 = sqrt(std::max(1.0e-12, es.eigenvalues()[0]));
      quality = (minD - 6.0 / s0) / fabs(std::max(1.0e-12, minD));
      if (behind) {
        //std::cout << "@@@@@@@@@@@@@ " << ++behindCtr << std::endl;
        quality = 0.0;
      }
      if(quality > 0.1) {
        isInitialised = true;
      } else {
        if(behind && best_pos.norm()>1.0e-12) {
          // reset along best ray
          it->second.hPoint->setEstimate(best_pos);
        }
      }
    }
    // update initialisation
    it->second.hPoint->setInitialized(isInitialised);
    it->second.quality = std::max(0.0, quality);
  }
}

/*void ViGraph::updateLandmarks()
{
  for (auto it = landmarks_.begin(); it != landmarks_.end(); ++it) {
    Eigen::Vector4d hp_W = it->second.hPoint->estimate();
    const size_t num = it->second.observations.size();
    bool isInitialised = false;
    double quality = 0.0;
    bool behind = false;
    double best_err = 1.0e12;
    Eigen::Vector4d best_pos(0.0,0.0,0.0,0.0);
    if(num>0){
      Eigen::Array<double, 3, Eigen::Dynamic> dirs(3,num);
      int o=0;
      //bool bad = false;
      for(const auto& observation : it->second.observations) {
        kinematics::Transformation T_WS, T_SCi;
        const StateId stateId(observation.first.frameId);
        const State & state = states_.at(stateId);
        T_WS = state.pose->estimate();
        T_SCi = state.extrinsics.at(observation.first.cameraIndex)->estimate();
        kinematics::Transformation T_WCi = T_WS*T_SCi;
        Eigen::Vector4d pos_Ci = T_WCi.inverse()*hp_W;

        if(fabs(pos_Ci[3])>1.0e-12) {
          pos_Ci = pos_Ci/pos_Ci[3];
        }
        Eigen::Vector3d dir_W = (T_WCi.C()*pos_Ci.head<3>()).normalized();
        if(pos_Ci[2]<0.1) {
          behind = true;
        }
        if(pos_Ci[2] < 0.0) {
          dir_W = -dir_W; // reverse!!
        }

        // consider only small reprojection errors
        Eigen::Vector2d err;
        double* params[3];
        params[0] = state.pose->parameters();
        params[1] = it->second.hPoint->parameters();
        params[2] = state.extrinsics.at(observation.first.cameraIndex)->parameters();
        observation.second.errorTerm->Evaluate(params, err.data(), nullptr);
        const double err_norm = err.norm();

        if(err_norm>2.5) {
          //bad = true;
          continue;
        }

        if(err_norm < best_err && pos_Ci.norm()>0.0001) {
          // remember best fit
          // if it was far away, leave it far away; but make sure it's in front
          // of the camera and at least at 10 cm...
          const double dist = std::max(0.1,pos_Ci.norm());
          best_pos.head<3>() = T_WCi.r() + dist*dir_W;
          best_pos[3]=1.0;
          best_err = err_norm;
        }

        dirs.col(o) = dir_W;
        ++o;
      }
      Eigen::Array<double, 3, Eigen::Dynamic> dirso(3,o);
      dirso = dirs.topLeftCorner(3,o);
      Eigen::Vector3d std_dev =
          ((dirso.colwise() - dirso.rowwise().mean()).square().rowwise().sum()).sqrt();
      quality = std_dev.norm();
      if (behind) {
        //std::cout << "@@@@@@@@@@@@@ " << ++behindCtr << std::endl;
        quality = 0.0;
      }
      if(quality > 0.04) {
        isInitialised = true;
      } else {
        if(behind && best_pos.norm()>1.0e-12) {
          // reset along best ray
          it->second.hPoint->setEstimate(best_pos);
        }
      }
    }
    // update initialisation
    it->second.hPoint->setInitialized(isInitialised);
    it->second.quality = quality;
  }
}*/

#ifdef USE_OPENMP
void ViGraph::optimise(int maxIterations, int numThreads, bool verbose)
#else
// avoid warning since numThreads unused
void ViGraph::optimise(int maxIterations, int /*numThreads*/, bool verbose)
#warning openmp not detected, your system may be slower than expected
#endif
{
  // assemble options

#ifdef USE_OPENMP
  options_.num_threads = int(numThreads);
#endif
  options_.max_num_iterations = int(maxIterations);

  if (verbose) {
    options_.minimizer_progress_to_stdout = true;
  } else {
    options_.minimizer_progress_to_stdout = false;
  }

  // call solver
  ::ceres::Solve(options_, problem_.get(), &summary_);

  // summary output
  if (verbose) {
    LOG(INFO) << summary_.FullReport();
  }
}

bool ViGraph::setOptimisationTimeLimit(double timeLimit, int minIterations)
{
  if (ceresCallback_ != nullptr) {
    if (timeLimit < 0.0) {
      // no time limit => set minimum iterations to maximum iterations
      ceresCallback_->setMinimumIterations(options_.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  } else if (timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(
          new okvis::ceres::CeresIterationCallback(timeLimit, minIterations));
    options_.callbacks.push_back(ceresCallback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

int ViGraph::cleanUnobservedLandmarks(std::map<LandmarkId, std::set<KeypointIdentifier> > *removed)
{
  int ctr = 0;
  for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
    const auto& lm = it->second;
    if(lm.observations.size()<=1) {
      if(removed) {
        (*removed)[it->first] = std::set<KeypointIdentifier>();
      }
      if(lm.observations.size()==1) {
        if(removed) {
          removed->at(it->first).insert(lm.observations.begin()->first);
        }
        removeObservation(lm.observations.begin()->first);
      }
      problem_->RemoveParameterBlock(lm.hPoint->parameters());
      it = landmarks_.erase(it);
      ctr++;
    }
    else {
      ++it;
    }
  }
  return ctr;
}

}  // namespace okvis
