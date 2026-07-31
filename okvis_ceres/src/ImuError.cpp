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
 * @file ImuError.cpp
 * @brief Source file for the ImuError class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>

#include <okvis/kinematics/operators.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ode/ode.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/PseudoInverse.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

std::atomic_bool ImuError::redoPropagationAlways(false);

// Construct with measurements and parameters.
ImuError::ImuError(const okvis::ImuMeasurementDeque & imuMeasurements,
                   const okvis::ImuParameters & imuParameters,
                   const okvis::Time& t_0, const okvis::Time& t_1) {
  
  reset();

  setImuMeasurements(imuMeasurements);
  setImuParameters(imuParameters);

  setT0(t_0);
  setT1(t_1);

  OKVIS_ASSERT_TRUE_DBG(
    Exception, t_0 >= imuMeasurements.front().timeStamp,
    "First IMU measurement included in ImuError is not old enough! "
      << t_0 << " !>= " << imuMeasurements.front().timeStamp)
  OKVIS_ASSERT_TRUE_DBG(
    Exception, t_1 <= imuMeasurements.back().timeStamp,
    "Last IMU measurement included in ImuError is not new enough!"
      << t_1 << " !<= " << imuMeasurements.back().timeStamp)
}

void ImuError::reset() const
{
  preintegrated_ = PreintegratedMeasurements::Identity();
  helpers_ = HelperMeasurements::Zero();

  P_delta_ = Eigen::MatrixXd::Zero(kNumResiduals, kNumResiduals);

  dPdsigma_.resize(4);
  for (size_t j = 0; j < 4; ++j) {
    dPdsigma_.at(j).setZero();
  }
}

void ImuError::setImuParameters(const ImuParameters &imuParameters) {

  imuParameters_ = imuParameters;

  // adjust covariance/information etc.
  if(dPdsigma_.size() > 0) {
    P_delta_ = dPdsigma_.at(0)*imuParameters_.sigma_g_c*imuParameters_.sigma_g_c;
    P_delta_ += dPdsigma_.at(1)*imuParameters_.sigma_a_c*imuParameters_.sigma_a_c;
    P_delta_ += dPdsigma_.at(2)*imuParameters_.sigma_gw_c*imuParameters_.sigma_gw_c;
    P_delta_ += dPdsigma_.at(3)*imuParameters_.sigma_aw_c*imuParameters_.sigma_aw_c;

    // calculate inverse and sqrt
    PseudoInverse::symmSqrtU(P_delta_, squareRootInformation_);
    information_ = squareRootInformation_.transpose()*squareRootInformation_;
  }
}

ImuError::PreintegratedMeasurements ImuError::computePreintegratedIncrements(
  const PreintegratedMeasurements & preintegrated, const ImuSensorReadings & imu, double dt)
{
  const auto & w = imu.gyroscopes;
  const auto & a = imu.accelerometers;

  PreintegratedMeasurements increments;

  increments.Delta_q = kinematics::deltaQ(w * dt);

  // Midpoint preintegrated rotation
  const Eigen::Matrix3d C =
    (preintegrated.Delta_q * kinematics::deltaQ(0.5 * w * dt)).toRotationMatrix();

  increments.Delta_p = preintegrated.Delta_v * dt + 0.5 * C * a * dt * dt;
  increments.Delta_v = C * a * dt;

  return increments;
}

ImuError::HelperMeasurements ImuError::computeNextHelperMeasurements(
  const HelperMeasurements & helpers, const ImuSensorReadings & imu, double dt,
  const Eigen::Matrix3d & C_0, const Eigen::Matrix3d & C_1, const Eigen::Quaterniond & dq)
{
  const auto & w = imu.gyroscopes;
  const auto & a = imu.accelerometers;

  const Eigen::Matrix3d a_wedge = kinematics::crossMx(a * dt);

  // Midpoint preintegrated rotation
  const Eigen::Matrix3d C = C_0 * kinematics::deltaQ(0.5 * w * dt).toRotationMatrix();

  HelperMeasurements next_helpers = HelperMeasurements::Zero();

  // Sum_{k = i}^{j - 1} Delta_R_ik * dt
  next_helpers.C_integral = helpers.C_integral + C * dt;

  // Sum_{k = i}^{j - 1} (Sum_{m = i}^{k - 1} Delta_R_im * dt) * dt + 0.5 * Delta_R_ik * dt * dt
  next_helpers.C_doubleintegral =
    helpers.C_doubleintegral + helpers.C_integral * dt + 0.5 * C * dt * dt;

  // d Delta_R_ij / d b_g = Sum_{k = i}^{j - 1} Delta_R_ik+1 * JR((w_k - b_w_k) * dt) * dt
  next_helpers.dalpha_db_g = helpers.dalpha_db_g + C_1 * kinematics::rightJacobian(w * dt) * dt;

  // d Delta_R_ij / d b_g = Sum_{k = i}^{j - 1} Delta_R_k+1j^T JR((w_k - b_w_k) * dt) * dt
  next_helpers.dq_db_g =
    dq.inverse().toRotationMatrix() * helpers.dq_db_g + kinematics::rightJacobian(w * dt) * dt;

  // Midpoint of dv_db_g * dq_db_g^-1
  next_helpers.ddv_db_g =
    0.5 * (C_0 * a_wedge * helpers.dq_db_g + C_1 * a_wedge * next_helpers.dq_db_g);

  // d Delta_v_ij / d b_g = Sum_{k = i}^{j - 1} Delta_R_ik * (a_k - b_a_k)^ * d Delta_R_ik / d b_g * dt
  next_helpers.dv_db_g = helpers.dv_db_g + next_helpers.ddv_db_g * dt;

  // dp_db_g_ = d Delta_p_ij / d b_g = Sum_{k = i}^{j - 1} Delta_v_ik * dt + 0.5 * Delta_R_ik * (a_k - b_a_k)^ * d Delta_R_ik / d b_g * dt * dt
  next_helpers.dp_db_g =
    helpers.dp_db_g + dt * helpers.dv_db_g + 0.5 * next_helpers.ddv_db_g * dt * dt;

  return next_helpers;
}

int ImuError::doPropagation(
  PreintegratedMeasurements & preintegrated, HelperMeasurements & helpers,
  Eigen::Matrix<double, kNumResiduals, kNumResiduals> & covariance,
  const ImuParameters & imuParameters, const ImuMeasurementDeque & imuMeasurements,
  const SpeedAndBias & speedAndBiases, const Time & t0, const Time & t1)
{
  Time time = t0;
  const Time & end = t1;

  bool hasStarted = false;
  int cnt = 0;
  for (ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it != imuMeasurements.end() - 1; ++it) {
    auto imu_0 = it->measurement;
    auto imu_1 = (it + 1)->measurement;

    Time current_time = it->timeStamp;
    Time next_time = (it + 1)->timeStamp;
    double dt = (next_time - time).toSec();

    if (dt <= 0.0) {
      continue;
    }

    if (end < next_time) {
      double interval = (next_time - it->timeStamp).toSec();
      next_time = end;
      dt = (next_time - time).toSec();
      const double r = dt / interval;
      imu_1.gyroscopes = ((1.0 - r) * imu_0.gyroscopes + r * imu_1.gyroscopes).eval();
      imu_1.accelerometers = ((1.0 - r) * imu_0.accelerometers + r * imu_1.accelerometers).eval();
    }

    if (!hasStarted) {
      hasStarted = true;
      if (current_time < time) {
        const double r = dt / (next_time - it->timeStamp).toSec();
        imu_0.gyroscopes = (r * imu_0.gyroscopes + (1.0 - r) * imu_1.gyroscopes).eval();
        imu_0.accelerometers = (r * imu_0.accelerometers + (1.0 - r) * imu_1.accelerometers).eval();
      }
    }

    double sigma_g_c = imuParameters.sigma_g_c;
    double sigma_a_c = imuParameters.sigma_a_c;
    const double & sigma_gw_c = imuParameters.sigma_gw_c;
    const double & sigma_aw_c = imuParameters.sigma_aw_c;

    if (
      imu_0.gyroscopes.cwiseAbs().maxCoeff() > imuParameters.g_max ||
      imu_1.gyroscopes.cwiseAbs().maxCoeff() > imuParameters.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING) << "Gyr saturation - scaling sigma";
    }

    if (
      imu_0.accelerometers.cwiseAbs().maxCoeff() > imuParameters.a_max ||
      imu_1.accelerometers.cwiseAbs().maxCoeff() > imuParameters.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "Acc saturation - scaling sigma";
    }

    ImuSensorReadings imu{
      0.5 * (imu_0.gyroscopes + imu_1.gyroscopes) - speedAndBiases.segment<3>(3),
      0.5 * (imu_0.accelerometers + imu_1.accelerometers) - speedAndBiases.segment<3>(6)};

    const auto & w = imu.gyroscopes;
    const auto & a = imu.accelerometers;

    auto preintegrated_increments = computePreintegratedIncrements(preintegrated, imu, dt);
    PreintegratedMeasurements next_preintegrated = preintegrated * preintegrated_increments;

    HelperMeasurements next_helpers = computeNextHelperMeasurements(
      helpers, imu, dt, preintegrated.Delta_q.toRotationMatrix(),
      next_preintegrated.Delta_q.toRotationMatrix(), preintegrated_increments.Delta_q);

    // TODO(foal): Potential improvement. We compute the same midpoint preintegrated rotation matrix three times.
    const Eigen::Matrix3d C =
      (preintegrated.Delta_q * kinematics::deltaQ(0.5 * w * dt)).toRotationMatrix();

    Eigen::Matrix<double, kNumResiduals, kNumResiduals> F_delta =
      Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Identity();
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> Q =
      Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Zero();

    F_delta.block<3, 3>(0, 3) =
      -kinematics::crossMx(preintegrated.Delta_v * dt + 0.5 * C * a * dt * dt);
    F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
    F_delta.block<3, 3>(0, 9) = dt * helpers.dv_db_g + 0.5 * next_helpers.ddv_db_g * dt * dt;
    F_delta.block<3, 3>(0, 12) = -helpers.C_integral * dt + 0.5 * C * dt * dt;
    F_delta.block<3, 3>(3, 9) = -dt * next_preintegrated.Delta_q.toRotationMatrix();
    F_delta.block<3, 3>(6, 3) = -kinematics::crossMx(preintegrated_increments.Delta_v);
    F_delta.block<3, 3>(6, 9) = next_helpers.ddv_db_g * dt;
    F_delta.block<3, 3>(6, 12) = -C * dt;

    Q.block<3, 3>(3, 3) = sigma_g_c * sigma_g_c * dt * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(6, 6) = sigma_a_c * sigma_a_c * dt * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(0, 0) = 0.5 * Q.block<3, 3>(6, 6) * dt * dt;
    Q.block<3, 3>(9, 9) = sigma_gw_c * sigma_gw_c * dt * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(12, 12) = sigma_aw_c * sigma_aw_c * dt * Eigen::Matrix3d::Identity();

    // Q = K * sigma_sq
    covariance = F_delta * covariance * F_delta.transpose() + Q;

    // memory shift
    preintegrated = next_preintegrated;
    helpers = next_helpers;
    time = next_time;

    ++cnt;

    if (next_time == end) {
      break;
    }
  }

  covariance = 0.5 * (covariance + covariance.transpose().eval());

  return cnt;
}

int ImuError::append(
  const okvis::kinematics::Transformation & /*T_WS*/, const okvis::SpeedAndBias & speedAndBiases,
  const okvis::ImuMeasurementDeque & imuMeasurements, const okvis::Time & t_1)
{
  OKVIS_ASSERT_TRUE_DBG(
    Exception, t1_ >= imuMeasurements.front().timeStamp,
    "First IMU measurement included in ImuError is not old enough!")

  OKVIS_ASSERT_TRUE_DBG(
    Exception, t_1 <= imuMeasurements.back().timeStamp,
    "Last IMU measurement included in ImuError is not new enough!")

  okvis::Time time = t1_;
  okvis::Time end = t_1;

  if (imuMeasurements.back().timeStamp < end) {
    return -1;
  }

  okvis::ImuMeasurementDeque::const_iterator iter = imuMeasurements.begin();
  while (iter != imuMeasurements.end()) {
    if (iter->timeStamp > imuMeasurements_.back().timeStamp) {
      break;
    }
    ++iter;
  }
  imuMeasurements_.insert(imuMeasurements_.end(), iter, imuMeasurements.end());

  setT1(t_1);

  const int cnt = doPropagation(
    preintegrated_, helpers_, P_delta_, imuParameters_, imuMeasurements, speedAndBiases, time, end);

  PseudoInverse::symmSqrtU(P_delta_, squareRootInformation_);
  information_ = squareRootInformation_.transpose() * squareRootInformation_;

  return cnt;
}

// Propagates pose, speeds and biases with given IMU measurements.
int ImuError::redoPreintegration(
  const okvis::kinematics::Transformation & /*T_WS*/,
  const okvis::SpeedAndBias & speedAndBiases) const
{
  // ensure unique access outside this function...

  okvis::Time time = t0_;
  okvis::Time end = t1_;

  OKVIS_ASSERT_TRUE_DBG(
    Exception, time >= imuMeasurements_.front().timeStamp,
    "First IMU measurement included in ImuError is not old enough!")

  OKVIS_ASSERT_TRUE_DBG(
    Exception, end <= imuMeasurements_.back().timeStamp,
    "Last IMU measurement included in ImuError is not new enough!")

  if (!(imuMeasurements_.back().timeStamp >= end)) {
    return -1;
  }

  reset();

  const int cnt = doPropagation(
    preintegrated_, helpers_, P_delta_, imuParameters_, imuMeasurements_, speedAndBiases, time,
    end);

  PseudoInverse::symmSqrtU(P_delta_, squareRootInformation_);
  information_ = squareRootInformation_.transpose() * squareRootInformation_;

  // store the reference (linearisation) point
  speedAndBiases_ref_ = speedAndBiases;

  return cnt;
}

void ImuError::syncFrom(const ImuError & other)
{
  imuParameters_ = other.imuParameters_;
  imuMeasurements_ = other.imuMeasurements_;
  t0_ = other.t0_;
  t1_ = other.t1_;
  preintegrated_ = other.preintegrated_;
  helpers_ = other.helpers_;
  P_delta_ = other.P_delta_;
  speedAndBiases_ref_ = other.speedAndBiases_ref_;
  redo_ = other.redo_;
  redoCounter_ = other.redoCounter_;
  information_ = other.information_;
  squareRootInformation_ = other.squareRootInformation_;
  dPdsigma_ = other.dPdsigma_;
}

std::shared_ptr<ImuErrorBase> ImuError::clone() const
{
  std::shared_ptr<ImuError> clone(new ImuError());
  clone->imuParameters_ = imuParameters_;
  clone->imuMeasurements_ = imuMeasurements_;
  clone->t0_ = t0_;
  clone->t1_ = t1_;
  clone->preintegrated_ = preintegrated_;
  clone->helpers_ = helpers_;
  clone->P_delta_ = P_delta_;
  clone->speedAndBiases_ref_ = speedAndBiases_ref_;
  clone->redo_ = redo_;
  clone->redoCounter_ = redoCounter_;
  clone->information_ = information_;
  clone->squareRootInformation_ = squareRootInformation_;
  clone->dPdsigma_ = dPdsigma_;
  return clone;
}

// Propagates pose, speeds and biases with given IMU measurements.
int ImuError::propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                          const okvis::ImuParameters & imuParams,
                          okvis::kinematics::Transformation& T_WS,
                          okvis::SpeedAndBias & speedAndBiases,
                          const okvis::Time & t_start,
                          const okvis::Time & t_end, covariance_t* covariance,
                          jacobian_t* jacobian) {

  // now the propagation
  okvis::Time time = t_start;
  okvis::Time end = t_end;

  // sanity check:
  OKVIS_ASSERT_TRUE(Exception, imuMeasurements.front().timeStamp<=time,
                        imuMeasurements.front().timeStamp << " !<= " << time)
  if (!(imuMeasurements.back().timeStamp >= end))
    return -1;  // nothing to do...

  // initial condition
  Eigen::Vector3d r_0 = T_WS.r();
  Eigen::Quaterniond q_WS_0 = T_WS.q();
  Eigen::Matrix3d C_WS_0 = T_WS.C();

  // increments (initialise with identity)
  Eigen::Quaterniond Delta_q(1,0,0,0);
  Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
  Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  Eigen::Matrix<double, kNumResiduals, kNumResiduals> P_delta =
    Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Zero();

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
        it != imuMeasurements.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();


    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParams.sigma_g_c;
    double sigma_a_c = imuParams.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParams.g_max
        || fabs(omega_S_0[1]) > imuParams.g_max
        || fabs(omega_S_0[2]) > imuParams.g_max
        || fabs(omega_S_1[0]) > imuParams.g_max
        || fabs(omega_S_1[1]) > imuParams.g_max
        || fabs(omega_S_1[2]) > imuParams.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING) << "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParams.a_max || fabs(acc_S_0[1]) > imuParams.a_max
        || fabs(acc_S_0[2]) > imuParams.a_max
        || fabs(acc_S_1[0]) > imuParams.a_max
        || fabs(acc_S_1[1]) > imuParams.a_max
        || fabs(acc_S_1[2]) > imuParams.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5*(omega_S_0+omega_S_1) - speedAndBiases.segment<3>(3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = ode::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5*(acc_S_0+acc_S_1) - speedAndBiases.segment<3>(6));
    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5*(C + C_1)*dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral + 0.5*(C + C_1)*acc_S_true*dt;
    // rotation matrix double integral:
    C_doubleintegral += C_integral*dt + 0.25*(C + C_1)*dt*dt;
    acc_doubleintegral += acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

    // Jacobian parts
    dalpha_db_g += dt*C_1;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix()*cross +
        okvis::kinematics::rightJacobian(omega_S_true*dt)*dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g + 0.5*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);
    dp_db_g += dt*dv_db_g + 0.25*dt*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);

    // covariance propagation
    if (covariance) {
      Eigen::Matrix<double, kNumResiduals, kNumResiduals> F_delta =
        Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Identity();
      // transform
      F_delta.block<3,3>(0,3) = -okvis::kinematics::crossMx(
            acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt);
      F_delta.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*dt;
      F_delta.block<3,3>(0,9) = dt*dv_db_g + 0.25*dt*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);
      F_delta.block<3,3>(0,12) = -C_integral*dt + 0.25*(C + C_1)*dt*dt;
      F_delta.block<3,3>(3,9) = -dt*C_1;
      F_delta.block<3,3>(6,3) = -okvis::kinematics::crossMx(0.5*(C + C_1)*acc_S_true*dt);
      F_delta.block<3,3>(6,9) = 0.5*dt*(C*acc_S_x*cross + C_1*acc_S_x*cross_1);
      F_delta.block<3,3>(6,12) = -0.5*(C + C_1)*dt;

      P_delta = F_delta*P_delta*F_delta.transpose();
      // add noise. Note that transformations with rotation matrices can be ignored, since the noise
      // is isotropic.
      //F_tot = F_delta*F_tot;
      const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
      P_delta(3,3) += sigma2_dalpha;
      P_delta(4,4) += sigma2_dalpha;
      P_delta(5,5) += sigma2_dalpha;
      const double sigma2_v = dt * sigma_a_c * imuParams.sigma_a_c;
      P_delta(6,6) += sigma2_v;
      P_delta(7,7) += sigma2_v;
      P_delta(8,8) += sigma2_v;
      const double sigma2_p = 0.5*dt*dt*sigma2_v;
      P_delta(0,0) += sigma2_p;
      P_delta(1,1) += sigma2_p;
      P_delta(2,2) += sigma2_p;
      const double sigma2_b_g = dt * imuParams.sigma_gw_c * imuParams.sigma_gw_c;
      P_delta(9,9)   += sigma2_b_g;
      P_delta(10,10) += sigma2_b_g;
      P_delta(11,11) += sigma2_b_g;
      const double sigma2_b_a = dt * imuParams.sigma_aw_c * imuParams.sigma_aw_c;
      P_delta(12,12) += sigma2_b_a;
      P_delta(13,13) += sigma2_b_a;
      P_delta(14,14) += sigma2_b_a;
    }

    // memory shift
    Delta_q = Delta_q_1;
    C_integral = C_integral_1;
    acc_integral = acc_integral_1;
    cross = cross_1;
    dv_db_g = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t_end)
      break;

  }

  // actual propagation output:
  const Eigen::Vector3d g_W = imuParams.g * Eigen::Vector3d(0, 0, 6371009).normalized();
  T_WS.set(r_0+speedAndBiases.head<3>()*Delta_t
             + C_WS_0*(acc_doubleintegral/*-C_doubleintegral*speedAndBiases.segment<3>(6)*/)
             - 0.5*g_W*Delta_t*Delta_t,
             q_WS_0*Delta_q);
  speedAndBiases.head<3>() += C_WS_0*(acc_integral/*-C_integral*speedAndBiases.segment<3>(6)*/)
      -g_W*Delta_t;

  // assign Jacobian, if requested
  if (jacobian) {
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> & F = *jacobian;
    F.setIdentity(); // holds for all states, including d/dalpha, d/db_g, d/db_a
    F.block<3,3>(0,3) = -okvis::kinematics::crossMx(C_WS_0*acc_doubleintegral);
    F.block<3,3>(0,6) = Eigen::Matrix3d::Identity()*Delta_t;
    F.block<3,3>(0,9) = C_WS_0*dp_db_g;
    F.block<3,3>(0,12) = -C_WS_0*C_doubleintegral;
    F.block<3,3>(3,9) = -C_WS_0*dalpha_db_g;
    F.block<3,3>(6,3) = -okvis::kinematics::crossMx(C_WS_0*acc_integral);
    F.block<3,3>(6,9) = C_WS_0*dv_db_g;
    F.block<3,3>(6,12) = -C_WS_0*C_integral;
  }

  // overall covariance, if requested
  if (covariance) {
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> & P = *covariance;
    // transform from local increments to actual states
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> T =
      Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Identity();
    T.topLeftCorner<3,3>() = C_WS_0;
    T.block<3,3>(3,3) = C_WS_0;
    T.block<3,3>(6,6) = C_WS_0;
    P = T * P_delta * T.transpose();
  }
  return i;
}

bool ImuError::initPose(const ImuMeasurementDeque &imuMeasurements,
                        kinematics::Transformation &T_WS) {
  // set translation to zero, unit rotation
  T_WS.setIdentity();

  if (imuMeasurements.size() == 0) return false;

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it < imuMeasurements.end(); ++it) {
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

  return true;
}

// This evaluates the error term and additionally computes the Jacobians.
bool ImuError::Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool ImuError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const {
  bool success = true;

  Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1> > weighted_error(residuals);

  // get poses
  const okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(
        parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]).normalized());

  const okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(
        parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]).normalized());

  // get speed and bias
  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (int i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  // call the propagation
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;
  // ensure unique access
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = speedAndBiases_0.tail<6>()
          - speedAndBiases_ref_.tail<6>();
    redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);
    if ((redo_ && ((imuMeasurements_.size() < 50) || redoPropagationAlways)) || redoCounter_==0) {
      const int steps = redoPreintegration(T_WS_0, speedAndBiases_0);
      if(steps<=0) {
        for(const auto & m : imuMeasurements_)
          std::cout << m.timeStamp << std::endl;
      }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            steps > 0,
                            "IMU error term computation failed: 0 steps integrated: "
                              << imuMeasurements_.front().timeStamp << " - "
                              << imuMeasurements_.back().timeStamp << " , "
                              << imuMeasurements_.size())
      if (steps == 0) {
        // hack it away
        LOG(WARNING) << "IMU error term computation failed: 0 steps integrated -- disable.";
        success = false;
      }

      redoCounter_++;
      Delta_b.setZero();
      redo_ = false;
    }
  }

  // actual propagation output:
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    // the above is a bit stupid, but shared read-locks only come in C++14
    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    // assign Jacobian w.r.t. x0
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> F0 =
      Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Identity();  // holds for d/db_g, d/db_a
    const Eigen::Vector3d delta_p_est_W = T_WS_0.r() - T_WS_1.r() +
                                          speedAndBiases_0.head<3>() * Delta_t -
                                          0.5 * g_W * Delta_t * Delta_t;
    const Eigen::Vector3d delta_v_est_W =
      speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() - g_W * Delta_t;
    const Eigen::Quaterniond Dq =
      okvis::kinematics::deltaQ(-helpers_.dalpha_db_g * Delta_b.head<3>()) * preintegrated_.Delta_q;
    F0.block<3, 3>(0, 0) = C_S0_W;
    F0.block<3, 3>(0, 3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
    F0.block<3, 3>(0, 6) = C_S0_W * Eigen::Matrix3d::Identity() * Delta_t;
    F0.block<3, 3>(0, 9) = helpers_.dp_db_g;
    F0.block<3, 3>(0, 12) = -helpers_.C_doubleintegral;
    F0.block<3, 3>(3, 3) =
      (okvis::kinematics::plus(Dq * T_WS_1.q().inverse()) * okvis::kinematics::oplus(T_WS_0.q()))
        .topLeftCorner<3, 3>();
    F0.block<3, 3>(3, 9) =
      (okvis::kinematics::oplus(T_WS_1.q().inverse() * T_WS_0.q()) * okvis::kinematics::oplus(Dq))
        .topLeftCorner<3, 3>() *
      (-helpers_.dalpha_db_g);
    F0.block<3, 3>(6, 3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
    F0.block<3, 3>(6, 6) = C_S0_W;
    F0.block<3, 3>(6, 9) = helpers_.dv_db_g;
    F0.block<3, 3>(6, 12) = -helpers_.C_integral;

    // assign Jacobian w.r.t. x1
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> F1 =
      -Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Identity();  // holds for the biases
    F1.block<3, 3>(0, 0) = -C_S0_W;
    F1.block<3, 3>(3, 3) = -(okvis::kinematics::plus(Dq) * okvis::kinematics::oplus(T_WS_0.q()) *
                             okvis::kinematics::plus(T_WS_1.q().inverse()))
                              .topLeftCorner<3, 3>();
    F1.block<3, 3>(6, 6) = -C_S0_W;

    // the overall error vector
    Eigen::Matrix<double, kNumResiduals, 1> error;
    error.segment<3>(0) =
      C_S0_W * delta_p_est_W + preintegrated_.Delta_p + F0.block<3, 6>(0, 9) * Delta_b;
    error.segment<3>(3) = 2 * (Dq * (T_WS_1.q().inverse() * T_WS_0.q())).vec();
    error.segment<3>(6) =
      C_S0_W * delta_v_est_W + preintegrated_.Delta_v + F0.block<3, 6>(6, 9) * Delta_b;
    error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();
    if (!success) {
      error.setZero();  // disable
    }

    // error weighting
    weighted_error = squareRootInformation_ * error;

    // get the Jacobians
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, kNumResiduals, 6> J0_minimal =
          squareRootInformation_ * F0.block<kNumResiduals, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[0], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> J0(jacobians[0]);
        J0 = J0_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>> J0_minimal_mapped(
              jacobiansMinimal[0]);
            J0_minimal_mapped = J0_minimal;
            if (!success) {
              J0_minimal_mapped.setZero();
            }
          }
        }
      }
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> J1(jacobians[1]);
        J1 = squareRootInformation_ * F0.block<kNumResiduals, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[1] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> J1_minimal_mapped(
              jacobiansMinimal[1]);
            J1_minimal_mapped = J1;
            if (!success) {
              J1_minimal_mapped.setZero();
            }
          }
        }
      }
      if (jacobians[2] != nullptr) {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, kNumResiduals, 6> J2_minimal =
          squareRootInformation_ * F1.block<kNumResiduals, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseManifold::minusJacobian(parameters[2], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> J2(jacobians[2]);
        J2 = J2_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[2] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>> J2_minimal_mapped(
              jacobiansMinimal[2]);
            J2_minimal_mapped = J2_minimal;
            if (!success) {
              J2_minimal_mapped.setZero();
            }
          }
        }
      }
      if (jacobians[3] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> J3(jacobians[3]);
        J3 = squareRootInformation_ * F1.block<kNumResiduals, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != nullptr) {
          if (jacobiansMinimal[3] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> J3_minimal_mapped(
              jacobiansMinimal[3]);
            J3_minimal_mapped = J3;
            if (!success) {
              J3_minimal_mapped.setZero();
            }
          }
        }
      }
    }
  }
  return true;
}

bool ImuError::EvaluateWithSigmaGradientAndHessian(const double * const *parameters, double *cost,
                                                   double *gradient, double *hessian) const
{
  // obtain residuals
  // get poses
  const okvis::kinematics::Transformation T_WS_0(
        Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

  const okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]));

  // get speed and bias
  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (int i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  // call the propagation
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;
  // ensure unique access
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = speedAndBiases_0.tail<6>()
          - speedAndBiases_ref_.tail<6>();
  }

  // actual propagation output:
  Eigen::Matrix<double, kNumResiduals, 1> error;
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    // the above is a bit stupid, but shared read-locks only come in C++14
    const Eigen::Vector3d g_W = imuParameters_.g * Eigen::Vector3d(0, 0, 6371009).normalized();

    // assign Jacobian w.r.t. x0
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> F0 =
      Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Identity();  // holds for d/db_g, d/db_a
    const Eigen::Vector3d delta_p_est_W = T_WS_0.r() - T_WS_1.r() +
                                          speedAndBiases_0.head<3>() * Delta_t -
                                          0.5 * g_W * Delta_t * Delta_t;
    const Eigen::Vector3d delta_v_est_W =
      speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() - g_W * Delta_t;
    const Eigen::Quaterniond Dq =
      okvis::kinematics::deltaQ(-helpers_.dalpha_db_g * Delta_b.head<3>()) * preintegrated_.Delta_q;
    F0.block<3, 3>(0, 0) = C_S0_W;
    F0.block<3, 3>(0, 3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
    F0.block<3, 3>(0, 6) = C_S0_W * Eigen::Matrix3d::Identity() * Delta_t;
    F0.block<3, 3>(0, 9) = helpers_.dp_db_g;
    F0.block<3, 3>(0, 12) = -helpers_.C_doubleintegral;
    F0.block<3, 3>(3, 3) =
      (okvis::kinematics::plus(Dq * T_WS_1.q().inverse()) * okvis::kinematics::oplus(T_WS_0.q()))
        .topLeftCorner<3, 3>();
    F0.block<3, 3>(3, 9) =
      (okvis::kinematics::oplus(T_WS_1.q().inverse() * T_WS_0.q()) * okvis::kinematics::oplus(Dq))
        .topLeftCorner<3, 3>() *
      (-helpers_.dalpha_db_g);
    F0.block<3, 3>(6, 3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
    F0.block<3, 3>(6, 6) = C_S0_W;
    F0.block<3, 3>(6, 9) = helpers_.dv_db_g;
    F0.block<3, 3>(6, 12) = -helpers_.C_integral;

    // assign Jacobian w.r.t. x1
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> F1 =
      -Eigen::Matrix<double, kNumResiduals, kNumResiduals>::Identity();  // holds for the biases
    F1.block<3, 3>(0, 0) = -C_S0_W;
    F1.block<3, 3>(3, 3) = -(okvis::kinematics::plus(Dq) * okvis::kinematics::oplus(T_WS_0.q()) *
                             okvis::kinematics::plus(T_WS_1.q().inverse()))
                              .topLeftCorner<3, 3>();
    F1.block<3, 3>(6, 6) = -C_S0_W;

    // the overall error vector
    error.segment<3>(0) =
      C_S0_W * delta_p_est_W + preintegrated_.Delta_p + F0.block<3, 6>(0, 9) * Delta_b;
    error.segment<3>(3) = 2 * (Dq * (T_WS_1.q().inverse() * T_WS_0.q())).vec();
    error.segment<3>(6) =
      C_S0_W * delta_v_est_W + preintegrated_.Delta_v + F0.block<3, 6>(6, 9) * Delta_b;
    error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();
  }

  // evaluate cost
  *cost = 0.5 * (error.transpose() * information_ * error + log(P_delta_.determinant()));

  // evaluate gradient and Hessian
  // sigma = [sigma_g_c, sigma_a_c, sigma_gw_c, sigma_aw_c]
  // sigma_g_c: gyro noise density [rad/s/sqrt(Hz)]
  // sigma_a_c: accelerometer noise density [m/s^2/sqrt(Hz)]
  // sigma_gw_c: gyro drift noise density [rad/s^s/sqrt(Hz)]
  // sigma_aw_c: accelerometer drift noise density [m/s^2/sqrt(Hz)]
  Eigen::Map<Eigen::Matrix<double, 4, 1>> gradientVec(gradient);
  Eigen::Map<Eigen::Matrix<double, 4, 4>> hessianMat(hessian);
  for (size_t j = 0; j < 4; ++j) {
    Eigen::Matrix<double, kNumResiduals, 1> inf_e = information_ * error;
    //Eigen::Matrix<double, kNumResiduals, kNumResiduals> Z
    //  = information_*dPdsigma_.at(j)*information_;
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> Y = information_ * dPdsigma_.at(j);
    gradientVec[int(j)] = 0.5 * (-inf_e.transpose() * dPdsigma_.at(j) * inf_e + Y.trace());
    // evaluate Hessian
    // diagonal first: dcost/dsigma_j^2
    //Eigen::Matrix<double, kNumResiduals, kNumResiduals> Zdash
    // = -Z*dPdsigma_.at(j)*information_ - information_*dPdsigma_.at(j)*Z;
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> Ydash = -Y * information_ * dPdsigma_.at(j);
    hessianMat(int(j), int(j)) =
      0.5 * (2 * (inf_e.transpose() * dPdsigma_.at(j)) * information_ * (dPdsigma_.at(j) * inf_e) +
             Ydash.trace());
    for (size_t k = 0; k < j; ++k) {
      // off-diagonal entries
      //Zdash = -Z*dPdsigma_.at(j)*information_ - information_*dPdsigma_.at(k)*Z;
      Ydash = -Y * information_ * dPdsigma_.at(k);
      const double t0 =
        ((inf_e.transpose() * dPdsigma_.at(j)) * information_ * (dPdsigma_.at(k) * inf_e));
      const double t1 =
        ((inf_e.transpose() * dPdsigma_.at(k)) * information_ * (dPdsigma_.at(j) * inf_e));
      hessianMat(int(j), int(k)) = 0.5 * (t0 + t1 + Ydash.trace());
      hessianMat(int(k), int(j)) = hessianMat(int(j), int(k));
    }
  }

  return true;
}

PseudoImuError::PseudoImuError(const Time &t_0, const Time &t_1)
{
  t0_ = t_0; // The start time (i.e. time of the first set of states).
  t1_ = t_1; // The end time (i.e. time of the sedond set of states).
}

void PseudoImuError::syncFrom(const PseudoImuError &other)
{
  t0_ = other.t0_; // The start time (i.e. time of the first set of states).
  t1_ = other.t1_; // The end time (i.e. time of the sedond set of states).
}

std::shared_ptr<ImuErrorBase> PseudoImuError::clone() const
{
  std::shared_ptr<PseudoImuError> clone(new PseudoImuError());
  //clone.
  clone->t0_ = t0_; // The start time (i.e. time of the first set of states).
  clone->t1_ = t1_; // The end time (i.e. time of the sedond set of states).
  return clone;
}

// This evaluates the error term and additionally computes the Jacobians.
bool PseudoImuError::Evaluate(double const *const *parameters,
                              double *residuals,
                              double **jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool PseudoImuError::EvaluateWithMinimalJacobians(double const *const *parameters,
                                                  double *residuals,
                                                  double **jacobians,
                                                  double **jacobiansMinimal) const
{
  // get poses
  const okvis::kinematics::Transformation
    T_WS_0(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
           Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5])
             .normalized());

  const okvis::kinematics::Transformation
    T_WS_1(Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
           Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5])
             .normalized());

  // get speed and bias
  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (int i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  // do the propagation
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Vector3d dr = speedAndBiases_0.head<3>() * Delta_t;
  kinematics::Transformation T_WS_1_predicted(T_WS_0.r() + dr, T_WS_0.q());

  // compute the residual
  Eigen::Matrix<double, kNumResiduals, 1> error;
  error.head<3>() = C_S0_W * (T_WS_1_predicted.r() - T_WS_1.r());
  error.segment<3>(3) = 2.0 * ((T_WS_1.q().inverse() * T_WS_0.q()).coeffs().head<3>());
  error.segment<3>(6) = C_S0_W * (speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>());
  error.segment<6>(9) = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();
  Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> weighted_error(residuals);
  information_t squareRootInformation = information_t::Identity(); /// \todo
  squareRootInformation.block<3, 3>(0, 0) *= 1.0 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(3, 3) *= 1.0 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(6, 6) *= 0.10 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(9, 9) *= 1.0 / sqrt(Delta_t);
  squareRootInformation.block<3, 3>(12, 12) *= 1.0 / sqrt(Delta_t);
  weighted_error = squareRootInformation * error;

  // get the Jacobians
  if (jacobians != nullptr) {
    if (jacobians[0] != nullptr) {
      // Jacobian w.r.t. minimal perturbance
      Eigen::Matrix<double, kNumResiduals, 6> Jp0 = Eigen::Matrix<double, kNumResiduals, 6>::Zero();
      Jp0.block<3, 3>(0, 0) = C_S0_W;
      Jp0.block<3, 3>(0, 3) = C_S0_W * kinematics::crossMx(T_WS_1_predicted.r() - T_WS_1.r());
      Jp0.block<3, 3>(3, 3)
        = kinematics::plus(T_WS_1.q().inverse() * T_WS_0.q()).topLeftCorner<3, 3>() * C_S0_W;
      Jp0.block<3, 3>(6, 3) = C_S0_W
                              * kinematics::crossMx(speedAndBiases_0.head<3>()
                                                    - speedAndBiases_1.head<3>());
      Eigen::Matrix<double, kNumResiduals, 6> J0_minimal = squareRootInformation * Jp0;

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[0], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0 = J0_minimal * J_lift;

      // if requested, provide minimal Jacobians
      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[0] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>> J0_minimal_mapped(
            jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> J1(jacobians[1]);
      Eigen::Matrix<double, kNumResiduals, 9> Jsb0 =
        Eigen::Matrix<double, kNumResiduals, 9>::Zero();
      Jsb0.block<3, 3>(0, 0) = Delta_t * C_S0_W;
      Jsb0.block<3, 3>(6, 0) = C_S0_W;
      Jsb0.bottomRightCorner<6, 6>() = Eigen::Matrix<double, 6, 6>::Identity();
      J1 = squareRootInformation * Jsb0;

      // if requested, provide minimal Jacobians
      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[1] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> J1_minimal_mapped(
            jacobiansMinimal[1]);
          J1_minimal_mapped = J1;
        }
      }
    }
    if (jacobians[2] != nullptr) {
      // Jacobian w.r.t. minimal perturbance
      Eigen::Matrix<double, kNumResiduals, 6> Jp1 = Eigen::Matrix<double, kNumResiduals, 6>::Zero();
      Jp1.block<3, 3>(0, 0) = -C_S0_W;
      Jp1.block<3, 3>(3, 3)
        = -kinematics::plus(T_WS_1.q().inverse() * T_WS_0.q()).topLeftCorner<3, 3>() * C_S0_W;
      Eigen::Matrix<double, kNumResiduals, 6> J2_minimal = squareRootInformation * Jp1;

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseManifold::minusJacobian(parameters[2], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor> > J2(
        jacobians[2]);
      J2 = J2_minimal * J_lift;

      // if requested, provide minimal Jacobians
      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[2] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor> > J2_minimal_mapped(
            jacobiansMinimal[2]);
          J2_minimal_mapped = J2_minimal;
        }
      }
    }
    if (jacobians[3] != nullptr) {
      Eigen::Matrix<double, kNumResiduals, 9> Jsb1 =
        Eigen::Matrix<double, kNumResiduals, 9>::Zero();
      Jsb1.block<3, 3>(6, 0) = -C_S0_W;
      Jsb1.bottomRightCorner<6, 6>() = -Eigen::Matrix<double, 6, 6>::Identity();
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor> > J3(jacobians[3]);
      J3 = squareRootInformation * Jsb1;

      // if requested, provide minimal Jacobians
      if (jacobiansMinimal != nullptr) {
        if (jacobiansMinimal[3] != nullptr) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor> > J3_minimal_mapped(
            jacobiansMinimal[3]);
          J3_minimal_mapped = J3;
        }
      }
    }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis
