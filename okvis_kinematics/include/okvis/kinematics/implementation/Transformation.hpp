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
 * @file implementation/Transformation.hpp
 * @brief Header implementation file for the TransformationT class.
 * @author Stefan Leutenegger
 */

#pragma once

#include <okvis/kinematics/Transformation.hpp>
#include "okvis/kinematics/operators.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief kinematics Namespace for kinematics functionality, i.e. TransformationTs and stuff.
namespace kinematics {

__inline__ double sinc(double x) {
  if (fabs(x) > 1.0e-6) {
    return sin(x) / x;
  } else {
    static const double c_2 = 1.0 / 6.0;
    static const double c_4 = 1.0 / 120.0;
    static const double c_6 = 1.0 / 5040.0;
    const double x_2 = x * x;
    const double x_4 = x_2 * x_2;
    const double x_6 = x_2 * x_2 * x_2;
    return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
  }
}

__inline__ Eigen::Quaterniond deltaQ(const Eigen::Vector3d& dAlpha)
{
  Eigen::Vector4d dq;
  const double halfnorm = 0.5 * dAlpha.norm();
  dq.template head<3>() = sinc(halfnorm) * 0.5 * dAlpha;
  dq[3] = cos(halfnorm);
  return Eigen::Quaterniond(dq);
}

// Right Jacobian, see Forster et al. RSS 2015 eqn. (8)
__inline__ Eigen::Matrix3d rightJacobian(const Eigen::Vector3d & PhiVec) {
  const double Phi = PhiVec.norm();
  Eigen::Matrix3d retMat = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d Phi_x = okvis::kinematics::crossMx(PhiVec);
  const Eigen::Matrix3d Phi_x2 = Phi_x*Phi_x;
  if(Phi < 1.0e-4) {
    retMat += -0.5*Phi_x + 1.0/6.0*Phi_x2;
  } else {
    const double Phi2 = Phi*Phi;
    const double Phi3 = Phi2*Phi;
    retMat += -(1.0-cos(Phi))/(Phi2)*Phi_x + (Phi-sin(Phi))/Phi3*Phi_x2;
  }
  return retMat;
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const TransformationT<CACHE_C> &other)
    : parameters_(other.parameters_),
      r_(&parameters_[0]),
      q_(&parameters_[3]),
      C_(other.C_) {
}
template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const TransformationT<!CACHE_C> & other)
    : parameters_(other.parameters_),
      r_(&parameters_[0]),
      q_(&parameters_[3]),
      C_(other.C_) {
  if(CACHE_C) C_ = q_.toRotationMatrix();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(TransformationT<CACHE_C> &&other)
    : parameters_(std::move(other.parameters_)),
      r_(&parameters_[0]),
      q_(&parameters_[3]),
      C_(std::move(other.C_)) {
}
template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(TransformationT<!CACHE_C> && other)
    : parameters_(std::move(other.parameters_)),
      r_(&parameters_[0]),
      q_(&parameters_[3]),
      C_(std::move(other.C_)) {
  if(CACHE_C) C_ = q_.toRotationMatrix();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT()
    : r_(&parameters_[0]),
      q_(&parameters_[3]),
      C_(Eigen::Matrix3d::Identity()) {
  r_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  q_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const Eigen::Vector3d & r_AB,
                                      const Eigen::Quaterniond& q_AB)
    : r_(&parameters_[0]),
      q_(&parameters_[3]) {
  r_ = r_AB;
  q_ = q_AB.normalized();
  if(CACHE_C) updateC();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::TransformationT(const Eigen::Matrix4d & T_AB)
    : r_(&parameters_[0]),
      q_(&parameters_[3]),
      C_(T_AB.topLeftCorner<3, 3>()) {
  r_ = (T_AB.topRightCorner<3, 1>());
  q_ = (T_AB.topLeftCorner<3, 3>());
  //assert(fabs(T_AB(3, 0)) < 1.0e-12);
  //assert(fabs(T_AB(3, 1)) < 1.0e-12);
  //assert(fabs(T_AB(3, 2)) < 1.0e-12);
  //assert(fabs(T_AB(3, 3) - 1.0) < 1.0e-12);
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>::~TransformationT() {

}

template<bool CACHE_C>
template<typename Derived_coeffs>
inline bool TransformationT<CACHE_C>::setCoeffs(
    const Eigen::MatrixBase<Derived_coeffs> & coeffs) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_coeffs, 7);
  parameters_ = coeffs;
  if(CACHE_C) updateC();
  return true;
}

// The underlying TransformationT
template<bool CACHE_C>
inline Eigen::Matrix4d TransformationT<CACHE_C>::T() const {
  Eigen::Matrix4d T_ret;
  T_ret.topLeftCorner<3, 3>() = CACHE_C ? C_ : q_.toRotationMatrix();
  T_ret.topRightCorner<3, 1>() = r_;
  T_ret.bottomLeftCorner<1, 3>().setZero();
  T_ret(3, 3) = 1.0;
  return T_ret;
}

// return the rotation matrix
template<bool CACHE_C>
inline Eigen::Matrix3d TransformationT<CACHE_C>::C() const {
  if(!CACHE_C) return q_.toRotationMatrix();
  return C_;
}

// return the translation vector
template<bool CACHE_C>
inline const Eigen::Map<Eigen::Vector3d> & TransformationT<CACHE_C>::r() const {
  return r_;
}

template<bool CACHE_C>
inline const Eigen::Map<Eigen::Quaterniond> & TransformationT<CACHE_C>::q() const {
  return q_;
}

template<bool CACHE_C>
inline Eigen::Matrix<double, 3, 4> TransformationT<CACHE_C>::T3x4() const {
  Eigen::Matrix<double, 3, 4> T3x4_ret;
  T3x4_ret.topLeftCorner<3, 3>() = CACHE_C ? C_ : q_.toRotationMatrix();
  T3x4_ret.topRightCorner<3, 1>() = r_;
  return T3x4_ret;
}

// Return a copy of the TransformationT inverted.
template<bool CACHE_C>
inline TransformationT<CACHE_C> TransformationT<CACHE_C>::inverse() const {
  return CACHE_C ? TransformationT(-(C_.transpose() * r_), q_.inverse()) :
                   TransformationT(-(q_.toRotationMatrix().transpose() * r_), q_.inverse());
}

// Set this to a random TransformationT.
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::setRandom() {
  setRandom(1.0, M_PI);
}
// Set this to a random TransformationT with bounded rotation and translation.
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::setRandom(double translationMaxMeters,
                                      double rotationMaxRadians) {
  // Create a random unit-length axis.
  Eigen::Vector3d axis = rotationMaxRadians * Eigen::Vector3d::Random();
  // Create a random rotation angle in radians.
  Eigen::Vector3d r = translationMaxMeters * Eigen::Vector3d::Random();
  r_ = r;
  q_ = Eigen::AngleAxisd(axis.norm(), axis.normalized());
  if(CACHE_C) updateC();
}

// Setters
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::set(const Eigen::Matrix4d & T_AB) {
  r_ = (T_AB.topRightCorner<3, 1>());
  q_ = (T_AB.topLeftCorner<3, 3>());
  if(CACHE_C) updateC();
}

template<bool CACHE_C>
inline void TransformationT<CACHE_C>::set(const Eigen::Vector3d & r_AB,
                                const Eigen::Quaternion<double> & q_AB) {
  r_ = r_AB;
  q_ = q_AB.normalized();
  if(CACHE_C) updateC();
}

// Set this TransformationT to identity
template<bool CACHE_C>
inline void TransformationT<CACHE_C>::setIdentity() {
  q_.setIdentity();
  r_.setZero();
  if(CACHE_C) C_.setIdentity();
}

template<bool CACHE_C>
inline TransformationT<CACHE_C> TransformationT<CACHE_C>::Identity() {
  return TransformationT();
}

// operator*
template<bool CACHE_C>
inline TransformationT<CACHE_C> TransformationT<CACHE_C>::operator*(
    const TransformationT<CACHE_C> & rhs) const {
  return TransformationT((CACHE_C ? C_ : q_.toRotationMatrix()) * rhs.r_ + r_, q_ * rhs.q_);
}
template<bool CACHE_C>
inline Eigen::Vector3d TransformationT<CACHE_C>::operator*(
    const Eigen::Vector3d & rhs) const {
  return (CACHE_C ? C_ : q_.toRotationMatrix()) * rhs;
}
template<bool CACHE_C>
inline Eigen::Vector4d TransformationT<CACHE_C>::operator*(
    const Eigen::Vector4d & rhs) const {
  const double s = rhs[3];
  Eigen::Vector4d retVec;
  retVec.head<3>() = (CACHE_C ? C_ : q_.toRotationMatrix()) * rhs.head<3>() + r_ * s;
  retVec[3] = s;
  return retVec;
}

template<bool CACHE_C>
inline TransformationT<CACHE_C>& TransformationT<CACHE_C>::operator=(const TransformationT<CACHE_C> & rhs) {
  parameters_ = rhs.parameters_;
  if(CACHE_C) C_ = rhs.C_;
  r_ = Eigen::Map<Eigen::Vector3d>(&parameters_[0]);
  q_ = Eigen::Map<Eigen::Quaterniond>(&parameters_[3]);
  return *this;
}
template<bool CACHE_C>
inline TransformationT<CACHE_C>& TransformationT<CACHE_C>::operator=(const TransformationT<!CACHE_C> & rhs) {
  parameters_ = rhs.parameters_;
  if(CACHE_C) C_ = q_.toRotationMatrix();
  r_ = Eigen::Map<Eigen::Vector3d>(&parameters_[0]);
  q_ = Eigen::Map<Eigen::Quaterniond>(&parameters_[3]);
  return *this;
}

template<bool CACHE_C>
inline void TransformationT<CACHE_C>::updateC() {
  if(CACHE_C) C_ = q_.toRotationMatrix();
  if(!CACHE_C) throw(std::runtime_error("bug"));
}

// apply small update:
template<bool CACHE_C>
template<typename Derived_delta>
inline bool TransformationT<CACHE_C>::oplus(
    const Eigen::MatrixBase<Derived_delta> & delta) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
  r_ += delta.template head<3>();
  const Eigen::Vector3d dalpha = delta.template tail<3>();
  Eigen::Vector4d dq;
  const double halfnorm = 0.5 * dalpha.norm();
  dq.template head<3>() = sinc(halfnorm) * 0.5 * dalpha;
  dq[3] = cos(halfnorm);
  q_ = (Eigen::Quaterniond(dq) * q_);
  q_.normalize();
  if(CACHE_C) updateC();
  return true;
}

template<bool CACHE_C>
template<typename Derived_delta, typename Derived_jacobian>
inline bool TransformationT<CACHE_C>::oplus(
    const Eigen::MatrixBase<Derived_delta> & delta,
    const Eigen::MatrixBase<Derived_jacobian> & jacobian) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived_delta, 6);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
  if (!oplus(delta)) {
    return false;
  }
  return oplusJacobian(jacobian);
}

template<bool CACHE_C>
template<typename Derived_jacobian>
inline bool TransformationT<CACHE_C>::oplusJacobian(
    const Eigen::MatrixBase<Derived_jacobian> & jacobian) const {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 7, 6);
  Eigen::Matrix<double, 4, 3> S = Eigen::Matrix<double, 4, 3>::Zero();
  const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
  const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
      .template topLeftCorner<3, 3>().setIdentity();
  S(0, 0) = 0.5;
  S(1, 1) = 0.5;
  S(2, 2) = 0.5;
  const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian)
      .template bottomRightCorner<4, 3>() = okvis::kinematics::oplus(q_) * S;
  return true;
}

template<bool CACHE_C>
template <typename Derived_jacobian>
inline bool TransformationT<CACHE_C>::liftJacobian(
  const Eigen::MatrixBase<Derived_jacobian> & jacobian) const
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived_jacobian, 6, 7);
  const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).setZero();
  const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template topLeftCorner<3,3>()
      = Eigen::Matrix3d::Identity();
  const_cast<Eigen::MatrixBase<Derived_jacobian>&>(jacobian).template bottomRightCorner<3,4>()
      = 2*okvis::kinematics::oplus(q_.inverse()).template topLeftCorner<3,4>();
  return true;
}

}  // namespace kinematics
}  // namespace okvis
