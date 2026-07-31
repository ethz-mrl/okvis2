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
 * @file ImuError.hpp
 * @brief Header file for the ImuError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_IMUERROR_HPP_
#define INCLUDE_OKVIS_CERES_IMUERROR_HPP_

#include <vector>
#include <mutex>
#include <atomic>

#include <ceres/sized_cost_function.h>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Implements a nonlinear IMU factor.
class ImuErrorBase : public ::ceres::SizedCostFunction<
                       15 /* number of residuals */,
                       7 /* size of first parameter (PoseParameterBlock k) */,
                       9 /* size of second parameter (SpeedAndBiasParameterBlock k) */,
                       7 /* size of third parameter (PoseParameterBlock k+1) */,
                       9 /* size of fourth parameter (SpeedAndBiasParameterBlock k+1) */>,
                     public ErrorInterface
{
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<15, 7, 9, 7, 9> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 15;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, kNumResiduals, kNumResiduals> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, kNumResiduals, kNumResiduals> jacobian_t;

  /// \brief The type of the Jacobian w.r.t. poses --
  /// \warning This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, kNumResiduals, 7> jacobian0_t;

  /// \brief The type of Jacobian w.r.t. Speed and biases
  typedef Eigen::Matrix<double, kNumResiduals, 9> jacobian1_t;

  /// \brief Trivial destructor.
  virtual ~ImuErrorBase() override = default;

  /// \brief (Re)set the start time.
  /// \@param[in] t_0 Start time.
  void setT0(const okvis::Time &t_0) { t0_ = t_0; }

  /// \brief (Re)set the start time.
  /// \@param[in] t_1 End time.
  void setT1(const okvis::Time &t_1) { t1_ = t_1; }

  // getters

  /// \brief Get the start time.
  okvis::Time t0() const { return t0_; }

  /// \brief Get the end time.
  okvis::Time t1() const { return t1_; }

  /// \brief Get a clone of this.
  /// \return A clone as std::shared_ptr.
  virtual std::shared_ptr<ImuErrorBase> clone() const = 0;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const override final {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  virtual int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  int parameterBlockDim(int parameterBlockId) const override final {
    return int(base_t::parameter_block_sizes().at(size_t(parameterBlockId)));
  }

  protected:
  // times
  okvis::Time t0_; ///< The start time (i.e. time of the first set of states).
  okvis::Time t1_; ///< The end time (i.e. time of the sedond set of states).
};

/// \brief Implements a nonlinear IMU factor.
class ImuError :
    public ImuErrorBase {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief If switched to true, all error terms will be always re-doing propagation (very slow).
  static std::atomic_bool redoPropagationAlways;

  /// \brief Default constructor -- assumes information recomputation.
  ImuError() = default;

  /// \brief Trivial destructor.
  virtual ~ImuError() override = default;

  /// \brief Construct with measurements and parameters.
  /// \@param[in] imuMeasurements All the IMU measurements.
  /// \@param[in] imuParameters The parameters to be used.
  /// \@param[in] t_0 Start time.
  /// \@param[in] t_1 End time.
  ImuError(const okvis::ImuMeasurementDeque & imuMeasurements,
           const okvis::ImuParameters & imuParameters, const okvis::Time& t_0,
           const okvis::Time& t_1);

  /// \brief Append with measurements and parameters.
  /// \@param[in] imuMeasurements All the IMU measurements.
  /// \@param[in] t_1 End time.
  int append(const okvis::kinematics::Transformation& T_WS,
             const okvis::SpeedAndBias & speedAndBiases,
             const okvis::ImuMeasurementDeque & imuMeasurements,
             const okvis::Time& t_1);

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @remark This can be used externally to perform propagation
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return Number of integration steps.
   */
  static int propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                         const okvis::ImuParameters & imuParams,
                         okvis::kinematics::Transformation& T_WS,
                         okvis::SpeedAndBias & speedAndBiases,
                         const okvis::Time& t_start, const okvis::Time& t_end,
                         covariance_t* covariance = nullptr,
                         jacobian_t* jacobian = nullptr);

  /// \brief Initialise pose from IMU measurements (assuming non-accelerating).
  /// @param[in] imuMeasurements IMU measurements.
  /// @param[out] T_WS The pose output.
  static bool initPose(const okvis::ImuMeasurementDeque& imuMeasurements,
                       okvis::kinematics::Transformation& T_WS);

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @warning This is not actually const, since the re-propagation must somehow be stored...
   * @param[in] T_WS Start pose.
   * @param[in] speedAndBiases Start speed and biases.
   * @return Number of integration steps.
   */
  int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                         const okvis::SpeedAndBias & speedAndBiases) const;

  /// \brief Synchronise from other IMU error.
  /// \param other The other IMU error.
  void syncFrom(const ImuError & other);

  // setters

  /// \brief (Re)set the parameters.
  /// \@param[in] imuParameters The parameters to be used.
  void setImuParameters(const okvis::ImuParameters& imuParameters);

  /// \brief (Re)set the measurements
  /// \@param[in] imuMeasurements All the IMU measurements.
  void setImuMeasurements(const okvis::ImuMeasurementDeque& imuMeasurements) {
    imuMeasurements_ = imuMeasurements;
  }

  // getters
  /// \brief Get a clone of this.
  /// \return A clone as std::shared_ptr.
  virtual std::shared_ptr<ImuErrorBase> clone() const override final;

  /// \brief Get the IMU Parameters.
  /// \return the IMU parameters.
  const okvis::ImuParameters& imuParameters() const {
    return imuParameters_;
  }

  /// \brief Get the IMU measurements.
  const okvis::ImuMeasurementDeque& imuMeasurements() const {
    return imuMeasurements_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  /**
   * @brief This evaluates the cost and additionally computes
   *        the gradient and Hessian in the minimal internal representation.
   * @param[in] parameters Pointer to the parameters (see ceres)
   * @param[out] cost Pointer to the scalar cost.
   * @param[out] gradient Pointer to the gradient.
   * @param[out] hessian Pointer to the Hessian.
   * @return Success of the evaluation.
   */
  bool EvaluateWithSigmaGradientAndHessian(double const* const* parameters,
                                            double* cost, double* gradient, double* hessian) const;

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const override final {
    return "ImuError";
  }

 protected:

  struct PreintegratedMeasurements
  {
    Eigen::Quaterniond Delta_q;
    Eigen::Vector3d Delta_v;
    Eigen::Vector3d Delta_p;

    PreintegratedMeasurements operator*(const PreintegratedMeasurements & other) const
    {
      PreintegratedMeasurements result;
      result.Delta_q = Delta_q * other.Delta_q;
      result.Delta_p = Delta_p + other.Delta_p;
      result.Delta_v = Delta_v + other.Delta_v;
      return result;
    }

    static PreintegratedMeasurements Identity()
    {
      PreintegratedMeasurements id;
      id.Delta_q = Eigen::Quaterniond(1, 0, 0, 0);
      id.Delta_v = Eigen::Vector3d::Zero();
      id.Delta_p = Eigen::Vector3d::Zero();
      return id;
    }
  };

  struct HelperMeasurements
  {
    Eigen::Matrix3d C_integral;
    Eigen::Matrix3d C_doubleintegral;
    Eigen::Matrix3d dq_db_g;
    Eigen::Matrix3d dalpha_db_g;
    Eigen::Matrix3d dv_db_g;
    Eigen::Matrix3d dp_db_g;
    Eigen::Matrix3d ddv_db_g;

    static HelperMeasurements Zero()
    {
      HelperMeasurements h;
      h.C_integral = Eigen::Matrix3d::Zero();
      h.C_doubleintegral = Eigen::Matrix3d::Zero();
      h.dq_db_g = Eigen::Matrix3d::Zero();
      h.dalpha_db_g = Eigen::Matrix3d::Zero();
      h.dv_db_g = Eigen::Matrix3d::Zero();
      h.dp_db_g = Eigen::Matrix3d::Zero();
      h.ddv_db_g = Eigen::Matrix3d::Zero();
      return h;
    }
  };

  /**
   * @brief Reset preintegrated measurements to identity, helper measurements to zero and preintegration covariance to zero.
   */
  void reset() const;

  /**
   * @brief Compute preintegrated measurement Delta_x_ik+1 from given measurement and time delta.
   * @param[in] imu The imu measurement at time k.
   * @param[in] dt Time delta tk+1 - tk.
   * @return The preintegrated measurement.
   */
  static PreintegratedMeasurements computePreintegratedIncrements(
    const PreintegratedMeasurements & preintegrated, const ImuSensorReadings & imu, double dt);

  /**
   * @brief Compute helper measurement at time k up to time k+1.
   * @param[in] imu The imu measurement at time k.
   * @param[in] dt Time delta tk+1 - tk.
   * @param[in] C_0 Preintegrated rotation at time k Delta_R_ik.
   * @param[in] C_1 Preintegrated rotation at time k+1 Delta_R_ik+1.
   * @param[in] dq Preintegrated rotation increment at time k.
   * @return The helper measurement increments.
   */
  static HelperMeasurements computeNextHelperMeasurements(
    const HelperMeasurements & helpers, const ImuSensorReadings & imu, double dt,
    const Eigen::Matrix3d & C_0, const Eigen::Matrix3d & C_1, const Eigen::Quaterniond & dq);

  /**
   * @brief Do propagation from t0_ to t1_ with given start states.
   * @param[in] speedAndBiases Start speed and biases.
   * @param[in] imuMeasurements The IMU measurements to be used. Must span t0_ - t1_.
   * @param[in] t0 Start time.
   * @param[in] t1 End time.
   * @param[in] reset If true, reset preintegrated measurements, helpers and covariance.
   * @return Number of integration steps.
   */
  static int doPropagation(
    PreintegratedMeasurements & preintegrated, HelperMeasurements & helpers,
    Eigen::Matrix<double, kNumResiduals, kNumResiduals> & covariance,
    const ImuParameters & imu_parameters, const ImuMeasurementDeque & imuMeasurements,
    const SpeedAndBias & speedAndBiases, const Time & t0, const Time & t1);

  // parameters
  okvis::ImuParameters imuParameters_; ///< The IMU parameters.

  // measurements
  okvis::ImuMeasurementDeque imuMeasurements_; ///< The IMU measurements. used. Must span t0_ - t1_.

  // preintegration stuff. the mutable is a TERRIBLE HACK, but what can I do.
  mutable std::mutex preintegrationMutex_; ///< Protect access of intermediate results.

  /// \brief The preintegrated measurements.
  mutable PreintegratedMeasurements preintegrated_;
  
  /// \brief Helper variables for the preintegration.
  mutable HelperMeasurements helpers_;

  /// \brief The Jacobian of the increment (w/o biases).
  mutable Eigen::Matrix<double, kNumResiduals, kNumResiduals> P_delta_
  ;

  /// \brief Reference biases that are updated when called redoPreintegration.
  mutable SpeedAndBias speedAndBiases_ref_ = SpeedAndBias::Zero();

  mutable bool redo_ = true; ///< Keeps track of whether or not redoPreintegration() is needed.
  mutable int redoCounter_ = 0; ///< Counts the number of preintegrations for statistics.

  // information matrix and its square root
  mutable information_t information_; ///< The information matrix for this error term.
  mutable information_t squareRootInformation_; ///< The square root information of this error term.

  /// \brief For gradient/hessian w.r.t. the sigmas.
  mutable AlignedVector<Eigen::Matrix<double, kNumResiduals, kNumResiduals>> dPdsigma_;

};

/// \brief Implements a linear pseudo-IMU factor as a constant velocity / orientation prior.
class PseudoImuError : public ImuErrorBase {
  public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Default constructor -- assumes information recomputation.
  PseudoImuError() = default;

  /// \brief Construct with times.
  /// \@param[in] t_0 Start time.
  /// \@param[in] t_1 End time.
  PseudoImuError(const okvis::Time& t_0, const okvis::Time& t_1);

  /// \brief Trivial destructor.
  virtual ~PseudoImuError() override = default;

  /**
   * @brief Propagates pose, speeds and biases with simple constant velocity / orientation prior.
   * @remark This can be used externally to perform propagation
   * @param[inout] T_WS Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @return Number of integration steps.
   */
  static int propagation(okvis::kinematics::Transformation& T_WS,
                         okvis::SpeedAndBias & speedAndBiases,
                         const okvis::Time& t_start, const okvis::Time& t_end);

  /// \brief Synchronise from other IMU error.
  /// \param other The other IMU error.
  void syncFrom(const PseudoImuError & other);

  // getters

  // getters
  /// \brief Get a clone of this.
  /// \return A clone as std::shared_ptr.
  virtual std::shared_ptr<ImuErrorBase> clone() const override final;

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const override final {
    return "PseudoImuError";
  }

  protected:

};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_IMUERROR_HPP_ */
