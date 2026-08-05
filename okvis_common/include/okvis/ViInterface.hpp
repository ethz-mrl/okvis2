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
 * @file ViInterface.hpp
 * @brief Header file for the ViInterface class.
 * @author Paul Furgale
 * @author Stefan Leutenegger
 * @author Andreas Froster
 */

#ifndef INCLUDE_OKVIS_VIINTERFACE_HPP_
#define INCLUDE_OKVIS_VIINTERFACE_HPP_

#include <cstdint>
#include <memory>
#include <functional>
#include <atomic>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#if defined(__has_include) && __has_include(<opencv2/features.hpp>)
#include <opencv2/features.hpp> // OpenCV 5 renamed features2d to features
#else
#include <opencv2/features2d/features2d.hpp>
#endif
#pragma GCC diagnostic pop
#include <okvis/assert_macros.hpp>
#include <okvis/Time.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Measurements.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief Map depth image in [m] to log scale for visualisation / TIF
/// \param frame The image to be transformed.
void expImageTransform(cv::Mat& frame);

/// \brief Map depth image in log scale (visualisation / TIF) back to [m].
/// \param frame The image to be transformed.
void logImageTransform(cv::Mat& frame);

/// \brief Helper class for trajectory propagations.
class Propagator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Constructor.
  /// \param t_start Start time.
  Propagator(const Time & t_start);

  /// \brief Default move onstructor.
  Propagator(Propagator && ) = default;

  /// \brief Default copy onstructor.
  Propagator(const Propagator & ) = default;

  /// \brief Append with IMU measurements.
  /// \param imuMeasurements The IMU measurements.
  /// \param T_WS **not used** (Starting pose used for linearisation).
  /// \param speedAndBiases Starting biases used for linearisation (speeds unused).
  /// \param t_end End time.
  /// \return True on success.
  bool appendTo(
      const ImuMeasurementDeque & imuMeasurements,
      const kinematics::Transformation& T_WS,
      const SpeedAndBias & speedAndBiases, const Time & t_end);

  /// \brief Obtain the state at the currently set t_end_ given states at
  /// t_start_.
  /// @param[in] T_WS_0 Pose at t_start_.
  /// @param[in] sb_0 Speed/biases at t_start_.
  /// @param[out] T_WS_1 Pose at t_end_.
  /// @param[out] sb_1 Speed/biases at t_end_.
  /// @param[out] omega_S Bias-corrected rotation rate.
  /// \return True on success.
  bool getState(
      const kinematics::Transformation & T_WS_0, const SpeedAndBias & sb_0,
      kinematics::Transformation & T_WS_1, SpeedAndBias & sb_1,
      Eigen::Vector3d & omega_S);

private:
  /// \brief sin(x)/x: to make things a bit faster than using angle-axis conversion.
  /// \param x Input x.
  /// \return Result of sin(x)/x.
  double sinc(double x);

  Time t_start_; ///< Start time.
  Time t_end_; ///< End time.
  double Delta_t_ = 0.0; ///< Time delta (end-start) in seconds.

  // increments (initialise with identity)
  Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); ///< Intermediate result
  Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); ///< Intermediate result
  Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); ///< Intermediate result

  // cross matrix accumulatrion
  Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  // sub-Jacobians
  Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  // rotation speed
  Eigen::Vector3d omega_S_ = Eigen::Vector3d::Zero(); ///< Intermediate result

  // linearisation point
  SpeedAndBias sb_ref_ = SpeedAndBias::Zero(); ///< Intermediate result
};

/// @brief Struct to hold all estimated states at a certain time instant.
struct State {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  kinematics::Transformation T_WS; ///< Transformation between World W and Sensor S.
  Eigen::Vector3d v_W; ///< Velocity in frame W [m/s].
  Eigen::Vector3d b_g; ///< Gyro bias [rad/s].
  Eigen::Vector3d b_a; ///< Accelerometer bias [m/s^2].
  Eigen::Vector3d omega_S; ///< Rotational velocity in frame S [rad/s].
  Time timestamp; ///< Timestamp corresponding to this state.
  StateId id; ///< Frame Id.
  ImuMeasurementDeque previousImuMeasurements; ///< IMU measurements up to this state's time.
  bool isKeyframe; ///< Is it a keyframe?
  AlignedMap<uint64_t, kinematics::Transformation> T_AiW; ///< Agent i to World pose (if available).
  std::set<StateId> covisibleFrameIds; ///< Covisible frame IDs.
  bool stateChanged; ///< Indicates if the state itself changed, or only other attributes (covis.).
};

/// @brief Simple enum to denote the tracking quality.
enum class TrackingQuality {
  Good, ///< Good quality for at least 30% of the image containing tracks.
  Marginal, ///< Marginal quality below that 30% level.
  Lost ///< Lost, if only one or less keypoints are matched per image.
};

/// @brief A helper struct to hold information about the tracking state.
struct TrackingState {
  StateId id; ///< ID this tracking info refers to.
  bool isKeyframe; ///< Is it a keyframe?
  TrackingQuality trackingQuality; ///< The tracking quality.
  bool recognisedPlace; ///< Has this fram recognised a place / relocalised / loop-closed?
  bool isFullGraphOptimising; ///< Is the background loop closure optimisation currently ongoing?
  StateId currentKeyframeId; ///< The ID of the current keyframe.
};

/// \brief A helper class to store, update and access states
///
/// \warning It's the caller's responsibility to synchronize calls to member
/// functions to avoid race conditions, e.g. calls to Trajectory::update() and
/// Trajectory::stateIds() from different threads. Using references returned by
/// member functions (e.g. by Trajectory::stateIds()) must also be synchronized
/// with non-const member function calls (e.g. Trajectory::update()).
class Trajectory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \name Setters
  /// \{

  /// \brief Update the trajectory.
  /// @param[in] trackingState Tracking state as received from Estimator callback.
  /// @param[in] updatedStates Updated states as received from Estimator callback.
  /// @param[out] affectedStateIds IDs of affected states.
  void update(const TrackingState & trackingState,
              std::shared_ptr<const AlignedMap<StateId, State> > updatedStates,
              std::set<StateId>& affectedStateIds);

  /**
   * \brief          Add IMU measurement for realtime propagation and obtain respective state.
   * \param stamp    The measurement timestamp.
   * \param alpha    The acceleration measured at this time.
   * \param omega    The angular velocity measured at this time.
   * \param state    The corresponding (propagated) state.
   * \return Returns true normally. False if not initialised.
   */
  bool addImuMeasurement(const okvis::Time & stamp,
                         const Eigen::Vector3d & alpha,
                         const Eigen::Vector3d & omega,
                         State& state);

  /// \}
  /// \name Getters
  /// \{

  /// \brief Obtain the (extrapolated) state at a certain timestamp. 
  ///        Note that this may access non-keyframe states (relative pose computations)
  ///        and may need to propagate to the precise timestamp. Therefore this is not a
  ///        trivial operation. Returns false, if no states/IMU measurements available for
  ///        the desired timestamp.
  /// @param[in] timestamp The requested timestamp.
  /// @param[out] state The respective state.
  /// \return True on success.
  bool getState(const Time & timestamp, State& state);

  /// \brief Get the state at an actually estimated state instance.
  /// @param[in] stateId The requested state ID.
  /// @param[out] state The respective state.
  /// \return True on success.
  bool getState(const StateId & stateId, State& state) const;

  /// \brief Get all the stateIds (estimated states).
  /// \return All the stateIds (estimated states).
  const std::set<StateId>& stateIds() const {return allStateIds_;}

  /// \brief Get the keyframe states.
  /// \return An std::map from keyframe state IDs to keyframe states.
  const AlignedMap<StateId, State>& keyframeStates() const;

  /// \brief Clears the cached propagators by StateId (save memory)
  /// @param[in] keyframeStateId The requested state ID (must be keyframe).
  /// \return True on success.
  bool clearCache(const StateId& keyframeStateId);

  /// \}
 private:
  AlignedMap<StateId, State> keyframeStates_; ///< Stores the keyframe states.
  std::map<StateId, StateId> keyframeIdByStateId_; ///< The reference keyframe Id by State Id.
  std::set<StateId> allStateIds_; ///< All stateIds.

  /// \brief helper struct for non-keyframe states.
  struct NonKeyframeState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    kinematics::Transformation T_Sk_S; ///< Transformation between keyframe Sk and Sensor S.
    Eigen::Vector3d v_Sk; ///< Velocity in frame Sk [m/s].
    Eigen::Vector3d omega_S_raw; ///< raw rotation rate in frame S, not compensated for bias.
    Time timestamp; ///< Timestamp corresponding to this state.
    StateId keyframeId; ///< Frame Id.
    StateId id; ///< Frame Id.
    ImuMeasurementDeque previousImuMeasurements; ///< IMU measurements up to this state's time.
  };

  /// \brief Stores the non-keyframe State information by State ID.
  std::map<StateId, AlignedMap<StateId, NonKeyframeState>> nonKeyframeStatesByKeyframeId_;

  /// \brief Associates the closest (earlier) state ID for a given timestamp.
  std::map<uint64_t, StateId> statesByTimestampUs_;

  /// \brief Caches the previously got propagated states by State ID propageted from and
  /// timestamp [us] propagated to.
  std::map<StateId, AlignedMap<uint64_t, Propagator>> propagatorsByKeyframeIdAndEndTimeUs_;

  /// \brief Propagate to the live state (if addImuMeasurement is called).
  std::unique_ptr<Propagator> livePropagator_;
  ImuMeasurementDeque imuMeasurements_; ///< IMU measurements received.
};

/**
 * @brief An abstract base class for interfaces between Front- and Backend.
 */
class ViInterface
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Unified callback function.
  typedef std::function<void(const State &, const TrackingState &,
                             std::shared_ptr<const AlignedMap<StateId, State>>,
                             std::shared_ptr<const okvis::MapPointVector>)> OptimisedGraphCallback;

  /// @brief Default constructor, not doing anything.
  ViInterface();

  /// @brief Destructor, closing file writing etc.
  virtual ~ViInterface();

  /// \name Setters
  /// \{

  /// \brief              Set a CVS file where the IMU data will be saved to.
  /// \param csvFile      The file.
  bool setImuCsvFile(std::fstream& csvFile);

  /// \brief              Set a CVS file where the IMU data will be saved to.
  /// \param csvFileName  The filename of a new file.
  bool setImuCsvFile(const std::string& csvFileName);

  /// \brief              Set a CVS file where the tracks (data associations) will be saved to.
  /// \param cameraId     The camera ID.
  /// \param csvFile      The file.
  bool setTracksCsvFile(size_t cameraId, std::fstream& csvFile);

  /// \brief              Set a CVS file where the tracks (data associations) will be saved to.
  /// \param cameraId     The camera ID.
  /// \param csvFileName  The filename of a new file.
  bool setTracksCsvFile(size_t cameraId, const std::string& csvFileName);

  /// \}
  /// \name Add measurements to the algorithm.
  /// \{
  /**
   * \brief             Add a set of new image.
   * \param stamp       The image timestamp.
   * \param images      The images. Can be gray (8UC1) or RGB (8UC3).
   * \param depthImages The depth images as float32, in [m]
   * \return            Returns true normally. False, if previous one not processed yet.
   */
  virtual bool addImages(const okvis::Time & stamp,
                         const std::map<size_t, cv::Mat> & images,
                         const std::map<size_t, cv::Mat> & depthImages
                         = std::map<size_t, cv::Mat>()) = 0;

  /**
   * \brief          Add an IMU measurement.
   * \param stamp    The measurement timestamp.
   * \param alpha    The acceleration measured at this time.
   * \param omega    The angular velocity measured at this time.
   * \return Returns true normally. False if the previous one has not been processed yet.
   */
  virtual bool addImuMeasurement(const okvis::Time & stamp,
                                 const Eigen::Vector3d & alpha,
                                 const Eigen::Vector3d & omega) = 0;
  /// \}
  /// \name Setters
  /// \{

  /// \brief Set the callback to be called every time the graph has been optimised.
  ///        A list of all changed states is provided, which can be usefull to keep
  ///        track e.g. upon loop closures
  virtual void setOptimisedGraphCallback(const OptimisedGraphCallback & optimisedGraphCallback);

  /**
   * \brief Set the blocking variable that indicates whether the addMeasurement() functions
   *        should return immediately (blocking=false), or only when the processing is complete.
   */
  virtual void setBlocking(bool blocking) = 0;
  /// \}
  

  /// @brief Display some visualisation.
  virtual void display(std::map<std::string, cv::Mat> & images) = 0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:

  /// \brief Write first line of IMU CSV file to describe columns.
  bool writeImuCsvDescription();

  /// \brief Write first line of tracks (data associations) CSV file to describe columns.
  bool writeTracksCsvDescription(size_t cameraId);

  OptimisedGraphCallback optimisedGraphCallback_; ///< Optimised graph callback function.

  std::shared_ptr<std::fstream> csvImuFile_;  ///< IMU CSV file.
  typedef std::map<size_t, std::shared_ptr<std::fstream>> FilePtrMap; ///< Map of file pointers.
  FilePtrMap csvTracksFiles_;  ///< Tracks CSV Files.

  std::atomic_bool realtimePropagation_; ///< Propagate to realtime?

};

}  // namespace okvis

#endif /* INCLUDE_OKVIS_VIINTERFACE_HPP_ */
