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
 * @file Component.cpp
 * @brief Source file for the Component class.
 * @author Stefan Leutenegger
 */


#include <fstream>
#include <iomanip>
#include <iostream>

#include <glog/logging.h>

#include "okvis/cameras/EquidistantDistortion.hpp"
#include "okvis/cameras/PinholeCamera.hpp"
#include "okvis/cameras/RadialTangentialDistortion.hpp"
#include "okvis/cameras/RadialTangentialDistortion8.hpp"

#include <okvis/Component.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

Component::Component(const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem,
                     ViGraphEstimator &fullGraph,
                     std::map<StateId, MultiFramePtr> multiFrames)
    : imuParameters_(imuParameters)
    , nCameraSystem_(nCameraSystem)
    , fullGraph_(&fullGraph)
    , multiFrames_(multiFrames)
{}

Component::Component(const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem)
    : imuParameters_(imuParameters)
    , nCameraSystem_(nCameraSystem)
{}

bool Component::load(const std::string &path)
{
  std::ifstream file(path);
  OKVIS_ASSERT_TRUE(Exception, file, "Could not open " << path << " for reading.");
  OKVIS_ASSERT_TRUE(Exception, !fullGraph_, "Graph not empty.");
  fullGraphOwn_.reset(new ViGraphEstimator());
  fullGraph_ = fullGraphOwn_.get();

  // find distortion type
  cameras::NCameraSystem::DistortionType distType = nCameraSystem_.distortionType(0);
  for (size_t i = 1; i < nCameraSystem_.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distType == nCameraSystem_.distortionType(i),
                      "mixed frame types are not supported yet")
  }

  // keep track of added extrinsics (bit hacky, should be part of ViGraph)
  std::vector<std::map<uint64_t, std::shared_ptr<ceres::PoseParameterBlock>>>
    extrinsicsParameterBlocks(nCameraSystem_.numCameras());

  // start reading file line by line
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream line_ss;
    line_ss << line;
    std::string type;
    line_ss >> type;

    // parse different type tags
    if ((type == "VERTEX_SE3:QUAT_TIME") || (type == "VERTEX_R3:VEL")
        || (type == "VERTEX_R3:ACCBIAS") || (type == "VERTEX_R3:GYRBIAS")) {
      uint64_t id64 = 0;
      line_ss >> id64;
      StateId id(id64);
      if (!fullGraph_->states_.count(id)) {
        fullGraph_->states_[id] = ViGraph::State();
      }
      ViGraph::State &state = fullGraph_->states_.at(id);

      if (type == "VERTEX_SE3:QUAT_TIME") {
        // parse pose
        double rx, ry, rz, qx, qy, qz, qw;
        line_ss >> rx;
        line_ss >> ry;
        line_ss >> rz;
        line_ss >> qx;
        line_ss >> qy;
        line_ss >> qz;
        line_ss >> qw;
        kinematics::Transformation T_WS(Eigen::Vector3d(rx, ry, rz),
                                        Eigen::Quaterniond(qw, qx, qy, qz));
        // parse timestamp
        uint64_t timestamp64;
        line_ss >> timestamp64;
        okvis::Time timestamp;
        timestamp.fromNSec(timestamp64);
        state.timestamp = timestamp;

        // set pose parameter block
        state.pose.reset(new ceres::PoseParameterBlock(T_WS, id64, timestamp));

        fullGraph_->problem_->AddParameterBlock(state.pose->parameters(),
                                                7,
                                                &fullGraph_->poseManifold_);
        state.pose->setLocalParameterizationPtr(&fullGraph_->poseManifold_);

        /// \todo Prior
        /// \todo Extrinsics

      } else {
        double vx, vy, vz;
        line_ss >> vx;
        line_ss >> vy;
        line_ss >> vz;
        // initialise speed and bias
        if (!state.speedAndBias) {
          state.speedAndBias.reset(new ceres::SpeedAndBiasParameterBlock(SpeedAndBias::Zero(),
                                                                         id64,
                                                                         state.pose->timestamp()));
          fullGraph_->problem_->AddParameterBlock(state.speedAndBias->parameters(), 9);
        }
        int offset = 0;
        if (type == "VERTEX_R3:VEL") {
          // parse velocity
          offset = 0;
        } else if (type == "VERTEX_R3:ACCBIAS") {
          // parse accelerometer bias
          offset = 6;
        } else if (type == "VERTEX_R3:GYRBIAS") {
          // parse gyro bias
          offset = 3;
        }
        state.speedAndBias->parameters()[offset] = vx;
        state.speedAndBias->parameters()[offset + 1] = vy;
        state.speedAndBias->parameters()[offset + 2] = vz;
      }
    } else if (type == "VERTEX_TRACKXYZ") {
      // parse landmark
      uint64_t id64;
      line_ss >> id64;
      LandmarkId id(id64);
      double rx, ry, rz;
      line_ss >> rx;
      line_ss >> ry;
      line_ss >> rz;
      Eigen::Vector4d r(rx, ry, rz, 1.0);
      double quality;
      line_ss >> quality;

      // create and add
      fullGraph_->addLandmark(id, r, quality > 0.001);
      fullGraph_->setLandmarkQuality(id, quality);
    } else if (type == "FRAME") {
      // parse frame info
      //stateId cameraIdx extrId [3 values extr. pos] [4 value extr. quat] [nsec timestamp]
      uint64_t stateId64, cameraIdx, extrId64;
      line_ss >> stateId64 >> cameraIdx >> extrId64;
      double rx, ry, rz, qx, qy, qz, qw;
      line_ss >> rx >> ry >> rz >> qx >> qy >> qz >> qw;
      kinematics::Transformation T_SC(Eigen::Vector3d(rx, ry, rz),
                                      Eigen::Quaterniond(qw, qx, qy, qz));
      uint64_t timestamp64;
      line_ss >> timestamp64;
      Time timestamp;
      timestamp.fromNSec(timestamp64);

      // create frame
      bool firstFrame = false;
      MultiFramePtr multiFrame;
      auto iter = multiFrames_.find(StateId(stateId64));
      if (iter != multiFrames_.end()) {
        multiFrame = iter->second;
      } else {
        multiFrame.reset(new MultiFrame(nCameraSystem_, timestamp, stateId64));
        multiFrames_[StateId(stateId64)] = multiFrame;
        firstFrame = true;
      }

      // create extrinsics, if needed
      ViGraph::State &state = fullGraph_->states_.at(StateId(stateId64));
      if (firstFrame) {
        state.extrinsics.resize(multiFrame->numFrames());
      }
      if (extrinsicsParameterBlocks.at(cameraIdx).count(extrId64)) {
        state.extrinsics.at(cameraIdx) = extrinsicsParameterBlocks.at(cameraIdx).at(extrId64);
      } else {
        state.extrinsics.at(cameraIdx).reset(
          new ceres::PoseParameterBlock(T_SC, extrId64, timestamp));
        extrinsicsParameterBlocks.at(cameraIdx)[extrId64] = state.extrinsics.at(cameraIdx);
        fullGraph_->problem_->AddParameterBlock(state.extrinsics.at(cameraIdx)->parameters(),
                                                7,
                                                &fullGraph_->poseManifold_);
      }

      // parse all keypoints
      std::vector<cv::KeyPoint> keypoints;
      keypoints.reserve(1000);
      std::vector<cv::Mat> descriptors;
      descriptors.reserve(1000);
      std::streampos oldpos = file.tellg(); // stores the position
      while (std::getline(file, line)) {
        std::stringstream line_ss;
        line_ss << line;
        std::string type;
        line_ss >> type;
        if (type == "FRAME:KEYPOINT") {
          // parse keypoint
          cv::KeyPoint kpt;
          uint64_t stateId64_2, i_2;
          line_ss >> stateId64_2 >> i_2 >> kpt.pt.x >> kpt.pt.y >> kpt.size;
          std::string descriptorType;
          line_ss >> descriptorType;
          OKVIS_ASSERT_TRUE(Exception,
                            descriptorType == "BRISK2",
                            "descriptor " << descriptorType << " not supported, only BRISK 2")
          OKVIS_ASSERT_TRUE(Exception, stateId64_2 == stateId64, "mismatching keypoint stateId")
          OKVIS_ASSERT_TRUE(Exception, cameraIdx == i_2, "mismatching keypoint stateId")
          keypoints.push_back(kpt);
          // Get descriptor
          std::string descriptorstring;
          line_ss >> descriptorstring;
          cv::Mat descriptor(1, 48, CV_8UC1);
          for (int col = 0; col < 48; ++col) {
            uint32_t byte;
            std::stringstream(descriptorstring.substr(2 * col, 2)) >> std::hex >> byte;
            descriptor.at<uchar>(0, col) = byte;
          }
          descriptors.push_back(descriptor);
        } else {
          // assemble and insert into multiframe
          multiFrame->resetKeypoints(cameraIdx, keypoints);
          cv::Mat allDescriptors(keypoints.size(), 48, CV_8UC1);
          for (uint64_t k = 0; k < keypoints.size(); ++k) {
            std::memcpy(allDescriptors.data + 48 * k, descriptors.at(k).data, 48);
          }
          multiFrame->resetDescriptors(cameraIdx, allDescriptors);
          multiFrame->computeBackProjections(cameraIdx);

          // finished sub-reading
          file.seekg(oldpos); // get back to the position
          break;
        }
        oldpos = file.tellg();
      }
    } else if (type == "EDGE_IMU") {
      // parse whole imu edge with measurements
      // EDGE_IMU prev_state_id state_id
      uint64_t prev_state_id, state_id;
      line_ss >> prev_state_id >> state_id;
      StateId prevStateId(prev_state_id);
      StateId stateId(state_id);

      // parse each measurement
      ImuMeasurementDeque imuMeasurements;
      std::streampos oldpos = file.tellg(); // stores the position
      while (std::getline(file, line)) {
        std::stringstream line_ss;
        line_ss << line;
        std::string type;
        line_ss >> type;
        if (type == "EDGE_IMU:MEASUREMENTS") {
          // parse measurement
          // EDGE_IMU:MEASUREMENTS [3 values acc m/s^2] [3 values gyr rad/s] [timestamp nsec]
          Eigen::Vector3d acc, gyr;
          line_ss >> acc[0] >> acc[1] >> acc[2] >> gyr[0] >> gyr[1] >> gyr[2];
          uint64_t timestamp64;
          Time timestamp;
          line_ss >> timestamp64;
          timestamp.fromNSec(timestamp64);
          ImuMeasurement imuMeasurement(timestamp, ImuSensorReadings(gyr, acc));
          imuMeasurements.push_back(imuMeasurement);
        } else {
          // assemble and insert IMU error term
          OKVIS_ASSERT_TRUE(Exception,
                            fullGraph_->states_.count(stateId),
                            "State " << state_id << " does not exist.");
          OKVIS_ASSERT_TRUE(Exception,
                            fullGraph_->states_.count(prevStateId),
                            "State " << prev_state_id << " does not exist.");
          ViGraph::State &state0 = fullGraph_->states_.at(prevStateId);
          ViGraph::State &state1 = fullGraph_->states_.at(stateId);
          state0.nextImuLink.errorTerm.reset(new ceres::ImuError(imuMeasurements,
                                                                 imuParameters_,
                                                                 state0.timestamp,
                                                                 state1.timestamp));
          state1.previousImuLink.errorTerm = state0.nextImuLink.errorTerm;
          state0.nextImuLink.residualBlockId
            = fullGraph_->problem_->AddResidualBlock(state0.nextImuLink.errorTerm.get(),
                                                     nullptr,
                                                     state0.pose->parameters(),
                                                     state0.speedAndBias->parameters(),
                                                     state1.pose->parameters(),
                                                     state1.speedAndBias->parameters());
          state1.previousImuLink.residualBlockId = state0.nextImuLink.residualBlockId;

          // finished sub-reading
          file.seekg(oldpos); // get back to the position
          break;
        }
        oldpos = file.tellg();
      }
    } else if (type == "EDGE_OBS") {
      // parse observation
      uint64_t stateId64, lmId64;
      int cameraIdx, keypointIdx;
      Eigen::Vector2d meas;
      Eigen::Matrix2d info;
      line_ss >> stateId64 >> cameraIdx >> keypointIdx >> lmId64 >> meas[0] >> meas[1] >> info(0, 0)
        >> info(0, 1) >> info(1, 0) >> info(1, 1);
      StateId stateId(stateId64);
      LandmarkId lmId(lmId64);

      // insert new multiFrame if needed
      MultiFramePtr multiFrame;
      if (!multiFrames_.count(stateId)) {
        OKVIS_THROW(Exception, "Observation to non-existant multi-frame")
      } else {
        multiFrame = multiFrames_.at(stateId);
      }

      // now construct observation
      multiFrame->setLandmarkId(cameraIdx, keypointIdx, lmId64);
      KeypointIdentifier kid(stateId64, cameraIdx, keypointIdx);
      switch (distType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        fullGraph_
          ->addObservation<cameras::PinholeCamera<cameras::RadialTangentialDistortion>>(*multiFrame,
                                                                                        lmId,
                                                                                        kid,
                                                                                        true);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        fullGraph_
          ->addObservation<cameras::PinholeCamera<cameras::EquidistantDistortion>>(*multiFrame,
                                                                                   lmId,
                                                                                   kid,
                                                                                   true);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        fullGraph_->addObservation<cameras::PinholeCamera<cameras::RadialTangentialDistortion8>>(
          *multiFrame, lmId, kid, true);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
      }

      // also store the 3D point
      Eigen::Vector4d hPoint_W = fullGraph_->landmark(lmId);
      kinematics::Transformation T_WS = fullGraph_->pose(stateId);
      Eigen::Vector4d hPoint_S = T_WS.inverse() * hPoint_W;
      multiFrame->setLandmark(cameraIdx,
                              keypointIdx,
                              hPoint_S,
                              fullGraph_->landmarks_.at(lmId).quality > 0.05);

    } else {
      file.close();
      OKVIS_THROW(Exception, "Error parsing " << path << ", unknown type tag " << type)
      return false;
    }
  }

  file.close();

  LOG(INFO) << "--- loaded component: ---";
  LOG(INFO) << "no. states: " << fullGraph_->states_.size();
  LOG(INFO) << "no. landmarks: " << fullGraph_->landmarks_.size();

  return true;
}

bool Component::save(const std::string &path)
{
  std::ios init(nullptr); // default format
  std::ofstream file(path);
  file << std::setprecision(17);
  OKVIS_ASSERT_TRUE(Exception, file, "Could not open " << path << " for writing.");
  init.copyfmt(file); // default format

  std::set<LandmarkId> writtenLandmarks; // only save landmarks once

  // go through state after state and write
  for (auto iter = fullGraph_->states_.begin(); iter != fullGraph_->states_.end(); ++iter) {
    StateId id = iter->first;
    const ViGraph::State &state = iter->second;
    // write pose vertex
    // VERTEX_SE3:QUAT_TIME id [3 values pos] [4 value quat] [nsec timestamp]
    const kinematics::Transformation &T_WS = state.pose->estimate();
    file << "VERTEX_SE3:QUAT_TIME " << id.value() << " " << T_WS.r()[0] << " " << T_WS.r()[1] << " "
         << T_WS.r()[2] << " " << T_WS.q().x() << " " << T_WS.q().y() << " " << T_WS.q().z() << " "
         << T_WS.q().w() << " " << state.timestamp.toNSec() << std::endl;

    // write speed and biases
    const SpeedAndBias &sb = state.speedAndBias->estimate();
    // VERTEX_R3:VEL id [3 value velocity m/s in World]
    file << "VERTEX_R3:VEL " << id.value() << " " << sb[0] << " " << sb[1] << " " << sb[2]
         << std::endl;
    // VERTEX_R3:ACCBIAS id [3 values accelerometer bias m/s^2]
    file << "VERTEX_R3:ACCBIAS " << id.value() << " " << sb[6] << " " << sb[7] << " " << sb[8]
         << std::endl;
    // VERTEX_R3:GYRBIAS id [3 values gyro bias rad/s]
    file << "VERTEX_R3:GYRBIAS " << id.value() << " " << sb[3] << " " << sb[4] << " " << sb[5]
         << std::endl;

    // write frame
    // FRAME stateId cameraIdx extrId [3 values extr. pos] [4 value extr. quat] [nsec timestamp]
    MultiFramePtr multiFrame = multiFrames_.at(id);
    for (uint64_t i = 0; i < multiFrame->numFrames(); ++i) {
      kinematics::Transformation T_SC = state.extrinsics.at(i)->estimate();
      file << "FRAME " << id.value() << " " << i << " " << state.extrinsics.at(i)->id() << " "
           << T_SC.r()[0] << " " << T_SC.r()[1] << " " << T_SC.r()[2] << " " << T_SC.q().x() << " "
           << T_SC.q().y() << " " << T_SC.q().z() << " " << T_SC.q().w() << " "
           << multiFrame->timestamp().toNSec() << std::endl;
      // write keypoints & descriptors
      // FRAME:KEYPOINT stateId cameraIdx [2 values pixels] [size] [descriptor as <KIND:values>]
      for (uint64_t k = 0; k < multiFrame->numKeypoints(i); ++k) {
        cv::KeyPoint cvKeypoint;
        multiFrame->getCvKeypoint(i, k, cvKeypoint);
        file << "FRAME:KEYPOINT " << id.value() << " " << i << " " << cvKeypoint.pt.x << " ";
        file << cvKeypoint.pt.y << " " << cvKeypoint.size << " BRISK2 ";
        const unsigned char *descriptor = multiFrames_.at(id)->keypointDescriptor(i, k);
        for (size_t i = 0; i < 48; ++i) {
          file << std::setfill('0') << std::setw(2) << std::hex << uint32_t(descriptor[i]);
        }
        file.copyfmt(init); // reset formatting
        file << std::endl;
      }
    }

    // write imu error term
    if (state.previousImuLink.errorTerm) {
      const auto imuError = std::dynamic_pointer_cast<ceres::ImuError>(
        state.previousImuLink.errorTerm);
      if(imuError) {
        auto iterPrev = iter;
        iterPrev--;

        // EDGE_IMU prev_state_id state_id
        ///a_max g_max a_max g_max sigma_g_c sigma_bg sigma_a_c sigma_ba sigma_gw_c sigma_aw_c g0 a0 g
        //const ImuParameters &pars = imuError.imuParameters();
        file << "EDGE_IMU " << iterPrev->first.value() << " " << id.value() << std::endl;
        //<< " " << pars.a_max
        //<< " " << pars.g_max << " " << pars.a_max << " " << pars.g_max << " " << pars.sigma_g_c
        //<< " " << pars.sigma_bg << " " << pars.sigma_a_c << " " << pars.sigma_ba << " "
        //<< pars.sigma_gw_c << " " << pars.sigma_aw_c << " " << pars.g0[0] << " " << pars.g0[1]
        //<< " " << pars.g0[2] << " " << pars.a0[0] << " " << pars.a0[1] << " " << pars.a0[2]
        //<< " " << pars.g << std::endl;

        // all IMU measurements
        const okvis::ImuMeasurementDeque imuMeasurements = imuError->imuMeasurements();
        for (auto imuIter = imuMeasurements.begin(); imuIter != imuMeasurements.end(); ++imuIter) {
          // EDGE_IMU:MEASUREMENTS [3 values acc m/s^2] [3 values gyr rad/s] [timestamp nsec]
          const Eigen::Vector3d acc = imuIter->measurement.accelerometers;
          const Eigen::Vector3d gyr = imuIter->measurement.gyroscopes;
          file << "EDGE_IMU:MEASUREMENTS " << acc[0] << " " << acc[1] << " " << acc[2] << " "
               << gyr[0] << " " << gyr[1] << " " << gyr[2] << " " << imuIter->timeStamp.toNSec()
               << std::endl;
        }
      }
    }

    // write observations and landmark
    for (const auto &obs : state.observations) {
      // landmark
      const LandmarkId lmId = obs.second.landmarkId;
      if (!writtenLandmarks.count(lmId)) { // only write every landmark once!
        const ViGraph::Landmark &landmark = fullGraph_->landmarks_.at(lmId);
        const Eigen::Vector4d hPt = landmark.hPoint->estimate();
        // VERTEX_TRACKXYZ id [3 value position] quality isInitialised
        file << "VERTEX_TRACKXYZ " << lmId.value() << " ";
        file << hPt[0] / hPt[3] << " " << hPt[1] / hPt[3] << " " << hPt[2] / hPt[3] << " ";
        file << landmark.quality << std::endl;
        writtenLandmarks.insert(obs.second.landmarkId);
      }

      // observation
      // EDGE_OBS state_id camera_idx keypoint_idx lm_id [2 values pixel coord.]
      // [9 values info matrix] [optional descriptor as <KIND:values>]
      const Eigen::Matrix2d info = obs.second.errorTerm->information();
      const Eigen::Vector2d coord = obs.second.errorTerm->measurement();
      file << "EDGE_OBS " << id.value() << " " << obs.first.cameraIndex << " "
           << obs.first.keypointIndex << " " << lmId.value() << " " << coord[0] << " " << coord[1]
           << " " << info(0, 0) << " " << info(0, 1) << " " << info(1, 0) << " " << info(1, 1)
           << std::endl;
    }
  }

  file.close();

  return true;
}

}  // namespace okvis
