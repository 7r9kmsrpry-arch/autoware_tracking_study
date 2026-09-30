// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define EIGEN_MPL2_ONLY

#include "multi_object_tracker_node.hpp"

#include "autoware/multi_object_tracker/object_model/shapes.hpp"
#include "autoware/multi_object_tracker/object_model/types.hpp"
#include "autoware/multi_object_tracker/uncertainty/uncertainty_processor.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/optional.hpp>

#include <glog/logging.h>
#include <tf2_ros/create_timer_interface.h>
#include <tf2_ros/create_timer_ros.h>

#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::multi_object_tracker
{
using autoware_utils::ScopedTimeTrack;
using Label = autoware_perception_msgs::msg::ObjectClassification;
using LabelType = autoware_perception_msgs::msg::ObjectClassification::_label_type;

// コンストラクタ
MultiObjectTracker::MultiObjectTracker(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("multi_object_tracker", node_options),
  last_published_time_(this->now()), 
  last_updated_time_(this->now())
{
  // glog for debug
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging("multi_object_tracker");
    google::InstallFailureSignalHandler();
  }

  // Get parameters
  double publish_rate = declare_parameter<double>("publish_rate");  // 10[hz]
  world_frame_id_ = declare_parameter<std::string>("world_frame_id"); // map
  std::string ego_frame_id = declare_parameter<std::string>("ego_frame_id"); // base_link
  // Detection遅延時もタイマー駆動でTrackedObjectsを周期出力するか
  enable_delay_compensation_ = declare_parameter<bool>("enable_delay_compensation"); // false
  // 自己位置の誤差を物体のcovarianceに反映するか(自己位置の誤差までTrackerに伝えるかどうか)
  bool enable_odometry_uncertainty = declare_parameter<bool>("consider_odometry_uncertainty"); // false
  // デバッグ用に各処理の実行時間を計測・publishするか
  bool use_time_keeper = declare_parameter<bool>("publish_processing_time_detail"); // false

  // ROS interface - Publisher(Tracking結果(TrackedObjects)を出力するPublisherを作成)
  tracked_objects_pub_ = create_publisher<autoware_perception_msgs::msg::TrackedObjects>(
    "output/objects", rclcpp::QoS{1}); // 最新の1情報のみ保持

  // Odometry manager(world座標系と自車座標系のTFを扱うOdometryクラスを生成)
  odometry_ =
    std::make_shared<Odometry>(*this, world_frame_id_, ego_frame_id, enable_odometry_uncertainty);

  // ROS interface - Input channels
  // define input channel parameters
  // detection01〜detection12について、各入力のchannel名とDetectedObjects topic名をパラメータ(xmlファイル)から取得
  // チャネル：認識結果の種類(LiDAR, Cameraなど)
  std::vector<std::string> input_channels;
  std::vector<std::string> input_channel_topics;
  input_channels.resize(types::max_channel_size); // max_channel_size:12
  input_channel_topics.resize(types::max_channel_size);
  for (size_t i = 0; i < types::max_channel_size; i++) {
    // the index number is zero filled two digits format
    const int index = static_cast<int>(i + 1);
    const std::string channel_id =
      std::string("detection") + (index < 10 ? "0" : "") + std::to_string(index); // detection01~detection12
    input_channels.at(i) = declare_parameter<std::string>("input/" + channel_id + "/channel");
    input_channel_topics.at(i) = declare_parameter<std::string>("input/" + channel_id + "/objects");
  }

  // parse input channels
  // 実際に有効なチャンネルだけを取り出して、Tracker内部で使いやすいInputChannel設定にまとめ直す
  uint channel_index = 0;
  for (size_t i = 0; i < types::max_channel_size; i++) {
    const std::string & input_channel = input_channels.at(i);
    const std::string & input_channel_topic = input_channel_topics.at(i);
    if (input_channel.empty() || input_channel == "none") {
      continue;
    }

    // 1つの有効な入力について設定をまとめる構造体
    types::InputChannel input_channel_config;
    input_channel_config.index = channel_index; // チャンネル番号
    channel_index++;

    // topic name
    input_channel_config.input_topic = input_channel_topic;

    // required parameter, but can set a default value
    // この入力元から、未対応Detectionを使って新規Trackerを生成してよいか(新規物体をトラッキング対象とするかどうか)
    input_channel_config.is_spawn_enabled = declare_parameter<bool>(
      "input_channels." + input_channel + ".flags.can_spawn_new_tracker", true);

    // trust object existence probability
    // この入力元の物体存在確率を信用するかどうか(falseの場合、後段でデフォルトの存在確率で上書きする)
    // fastbevでは出力しないので、注意！
    input_channel_config.trust_existence_probability = declare_parameter<bool>(
      "input_channels." + input_channel + ".flags.can_trust_existence_probability", false);

    // trust object extension, size beyond the visible area
    // この入力元が推定したBBoxのサイズを信用するか(クラスタリング手法の場合、見えてる部分だけで決めるのでfalse)
    input_channel_config.trust_extension = declare_parameter<bool>(
      "input_channels." + input_channel + ".flags.can_trust_extension", true);

    // trust object classification
    // この入力元が推定した物体クラスを信用するか
    input_channel_config.trust_classification = declare_parameter<bool>(
      "input_channels." + input_channel + ".flags.can_trust_classification", true);

    // trust object orientation(yaw)
    // この入力元が推定した物体の向き(yaw)を信用するか
    input_channel_config.trust_orientation = declare_parameter<bool>(
      "input_channels." + input_channel + ".flags.can_trust_orientation", true);

    // optional parameters
    // デバッグやロギング時の入力チャンネルの表示用名称を設定
    const std::string default_name = input_channel;
    const std::string name_long = declare_parameter<std::string>(
      "input_channels." + input_channel + ".optional.name", default_name);
    input_channel_config.long_name = name_long;
    
    // 短縮名を設定
    const std::string default_name_short = input_channel.substr(0, 3);
    const std::string name_short = declare_parameter<std::string>(
      "input_channels." + input_channel + ".optional.short_name", default_name_short);
    input_channel_config.short_name = name_short;

    input_channels_config_.push_back(input_channel_config);
  }
  // 有効チャンネル名の個数
  input_channel_size_ = input_channels_config_.size();

  // Initialize input manager(複数のDetection入力を管理するクラス)
  input_manager_ = std::make_unique<InputManager>(*this, odometry_);
  input_manager_->init(input_channels_config_);  // Initialize input manager, set subscriptions(入力チャンネル設定をもとにsubscriberを初期化)
  input_manager_->setTriggerFunction(
    std::bind(&MultiObjectTracker::onTrigger, this));  // Set trigger function(Detection受信時にonTrigger()を呼ぶようコールバックを登録)

  // Create ROS time based timer.
  // If the delay compensation is enabled, the timer is used to publish the output at the correct
  // time.
  // 遅延補償有効時、一定周期でTrackedObjectsをpublishするためのタイマーを作成
  if (enable_delay_compensation_) {
    publisher_period_ = 1.0 / publish_rate;    // [s](default=100ms)
    constexpr double timer_multiplier = 10.0;  // 10 times frequent for publish timing check(pub周期の十倍の周期で時刻を確認する)
    const auto timer_period = rclcpp::Rate(publish_rate * timer_multiplier).period();
    publish_timer_ = rclcpp::create_timer(
      this, get_clock(), timer_period, std::bind(&MultiObjectTracker::onTimer, this)); // publish周期の10倍の頻度でonTimer()を呼び、publishタイミングを確認
  }

  // Initialize processor
  {
    // Parameters for processor
    TrackerProcessorConfig config;
    {
      // convert string to TrackerType
      // Tracker名の文字列をTrackerType enumへ変換する対応表
      static const std::unordered_map<std::string, TrackerType> TRACKER_TYPE_MAP = {
        {"multi_vehicle_tracker", TrackerType::MULTIPLE_VEHICLE},
        {"pedestrian_and_bicycle_tracker", TrackerType::PEDESTRIAN_AND_BICYCLE},
        {"normal_vehicle_tracker", TrackerType::NORMAL_VEHICLE},
        {"pedestrian_tracker", TrackerType::PEDESTRIAN},
        {"big_vehicle_tracker", TrackerType::BIG_VEHICLE},
        {"bicycle_tracker", TrackerType::BICYCLE},
        {"pass_through_tracker", TrackerType::PASS_THROUGH}};

      // tracker_nameに対応するTrackerTypeを返す関数を定義(対応する名前がない場合はUNKNOWNを返す)
      auto getTrackerType = [](const std::string & tracker_name) -> TrackerType {
        auto it = TRACKER_TYPE_MAP.find(tracker_name);
        return it != TRACKER_TYPE_MAP.end() ? it->second : TrackerType::UNKNOWN;
      };

      // 物体クラスごとに使用するTracker種類をROS parameterから取得して登録
      config.tracker_map.insert(
        std::make_pair(
          Label::CAR, getTrackerType(this->declare_parameter<std::string>("car_tracker"))));
      config.tracker_map.insert(
        std::make_pair(
          Label::TRUCK, getTrackerType(this->declare_parameter<std::string>("truck_tracker"))));
      config.tracker_map.insert(
        std::make_pair(
          Label::BUS, getTrackerType(this->declare_parameter<std::string>("bus_tracker"))));
      config.tracker_map.insert(
        std::make_pair(
          Label::TRAILER, getTrackerType(this->declare_parameter<std::string>("trailer_tracker"))));
      config.tracker_map.insert(
        std::make_pair(
          Label::PEDESTRIAN,
          getTrackerType(this->declare_parameter<std::string>("pedestrian_tracker"))));
      config.tracker_map.insert(
        std::make_pair(
          Label::BICYCLE, getTrackerType(this->declare_parameter<std::string>("bicycle_tracker"))));
      config.tracker_map.insert(
        std::make_pair(
          Label::MOTORCYCLE,
          getTrackerType(this->declare_parameter<std::string>("motorcycle_tracker"))));
      config.tracker_map.insert(
        std::make_pair(Label::UNKNOWN, TrackerType::UNKNOWN));  // Default for unknown objects

      // Declare parameters
      config.tracker_lifetime = declare_parameter<double>("tracker_lifetime"); // Trackerを観測なしで保持する時間[s]
      config.min_known_object_removal_iou =
        declare_parameter<double>("min_known_object_removal_iou"); // 既知クラスの重複Trackerを削除するためのIoU閾値
      config.min_unknown_object_removal_iou =
        declare_parameter<double>("min_unknown_object_removal_iou"); // UNKNOWNクラスの重複Trackerを削除するためのIoU閾値

      // Declare parameters for generalized IoU threshold
      // GIOU：重なっていない場合でも、BBox同士の近さを評価可能(距離が近いほど、0に近い負の数、遠いほど0から離れた負の数になる)
      std::vector<double> pruning_giou_thresholds =
        declare_parameter<std::vector<double>>("pruning_generalized_iou_thresholds"); // YAMLからクラス別のGIoU閾値を配列で取得
      // 配列の各要素を物体クラスと対応付けてmapに保存
      for (size_t i = 0; i < pruning_giou_thresholds.size(); ++i) {
        const auto label = static_cast<LabelType>(i);
        config.pruning_giou_thresholds[label] = pruning_giou_thresholds.at(i);
      }

      // Declare parameters for overlap distance threshold
      // 重複判定の対象とするTracker間の最大距離をクラス別に取得
      std::vector<double> pruning_distance_threshold_list =
        declare_parameter<std::vector<double>>("pruning_distance_thresholds");
      for (size_t i = 0; i < pruning_distance_threshold_list.size(); ++i) {
        const auto label = static_cast<LabelType>(i);
        config.pruning_distance_thresholds[label] = pruning_distance_threshold_list[i];
      }

      // Unknownクラスの速度に関する取り扱い
      config.enable_unknown_object_velocity_estimation =
        declare_parameter<bool>("enable_unknown_object_velocity_estimation"); // 速度を推定するか
      config.enable_unknown_object_motion_output =
        declare_parameter<bool>("enable_unknown_object_motion_output"); // 速度を出力するか
    }

    AssociatorConfig associator_config;
    { 
      // YAMLから取得した1次元整数配列をlabel_num × label_numのEigen行列に変換
      auto initializeMatrixInt = [](const std::vector<int64_t> & vector) {
        // 要素数がクラス数 × クラス数になっているか確認
        const int label_num = types::NUM_LABELS;
        if (vector.size() != label_num * label_num) {
          throw std::runtime_error("Invalid can_assign_matrix size");
        }

        // ROS parameterのint64_t配列をint配列へ変換
        std::vector<int> converted_vector(vector.begin(), vector.end());

        // Use row-major mapping to match the YAML layout(YAMLは行方向に並んでいるためRowMajorとして行列化)
        using RowMajorMatrixXi =
          Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<RowMajorMatrixXi> matrix_tmp(converted_vector.data(), label_num, label_num);

        // Convert to column-major (Eigen's default) for consistency(Eigen標準形式の行列として返す)
        return Eigen::MatrixXi(matrix_tmp);
      };
      
      // YAMLから取得した1次元double配列をlabel_num × label_numのEigen行列に変換
      auto initializeMatrixDouble = [](const std::vector<double> & vector) {
        // 要素数がクラス数 × クラス数になっているか確認
        const int label_num = types::NUM_LABELS;
        if (vector.size() != label_num * label_num) {
          throw std::runtime_error("Invalid association matrix configuration size");
        }

        // Use row-major mapping to match the YAML layout(YAMLは行方向に並んでいるためRowMajorとして行列化)
        using RowMajorMatrixXd =
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RowMajorMatrixXd> matrix_tmp(vector.data(), label_num, label_num);

        // Convert to column-major (Eigen's default) for consistency(Eigen標準形式の行列として返す)
        return Eigen::MatrixXd(matrix_tmp);
      };
      
      // YAMLからAssociation用の各閾値を読み込み、クラス×クラスの行列に変換
      Eigen::MatrixXi can_assign_matrix =
        initializeMatrixInt(this->declare_parameter<std::vector<int64_t>>("can_assign_matrix")); // TrackerTypeとDetectionクラスの対応付け可否
      associator_config.max_dist_matrix =
        initializeMatrixDouble(this->declare_parameter<std::vector<double>>("max_dist_matrix")); // TrackerとDetectionの最大許容距離
      associator_config.max_area_matrix =
        initializeMatrixDouble(this->declare_parameter<std::vector<double>>("max_area_matrix")); // DetectionのBBox面積の最大許容値
      associator_config.min_area_matrix =
        initializeMatrixDouble(this->declare_parameter<std::vector<double>>("min_area_matrix")); // DetectionのBBox面積の最小許容値
      associator_config.max_rad_matrix =
        initializeMatrixDouble(this->declare_parameter<std::vector<double>>("max_rad_matrix")); // TrackerとDetectionの最大許容yaw差 [rad]
      associator_config.min_iou_matrix =
        initializeMatrixDouble(this->declare_parameter<std::vector<double>>("min_iou_matrix")); // TrackerとDetectionを対応付けるための最小IoU

      // pre-process(Association計算用に閾値を前処理)
      const int label_num = associator_config.max_dist_matrix.rows(); // クラス数を取得(8クラス：UNKNOWN,CAR,TRUCK,BUS,TRAILER,MOTORBIKE,BICYCLE,PEDESTRIAN)
      for (int i = 0; i < label_num; i++) {
        for (int j = 0; j < label_num; j++) {
          // yawの閾値を正の値に設定
          associator_config.max_rad_matrix(i, j) = std::abs(associator_config.max_rad_matrix(i, j));
          // 距離閾値を2城下値に設定
          associator_config.max_dist_matrix(i, j) =
            associator_config.max_dist_matrix(i, j) * associator_config.max_dist_matrix(i, j);
        }
      }

      // Set the unknown-unknown association GIoU threshold(UNKNOWN TrackerとUNKNOWN Detectionを対応付けるためのGIoU閾値)
      associator_config.unknown_association_giou_threshold =
        declare_parameter<double>("unknown_association_giou_threshold");

      // Set the tracker map for associator config
      {
        // TrackerTypeごとのDetection対応可否を初期化
        associator_config.can_assign_map.clear();
        for (const auto & [label, tracker_type] : config.tracker_map) {
          associator_config.can_assign_map[tracker_type].fill(false);
        }
        // can_assign_map : tracker_type that can be assigned to each measurement label
        // relationship is given by tracker_map and can_assign_matrix
        // can_assign_matrix(Label × Detection Label)をcan_assign_map(TrackerType → 対応可能Detection Label)へ変換
        for (int i = 0; i < can_assign_matrix.rows(); ++i) {
          for (int j = 0; j < can_assign_matrix.cols(); ++j) {
            if (can_assign_matrix(i, j) == 1) {
              const auto tracker_type = config.tracker_map.at(i);
              associator_config.can_assign_map[tracker_type][j] = true;
            }
          }
        }
      }
    }

    // Initialize processor with parameters(Tracker設定、Association設定、入力チャンネル設定を用いてTrackerProcessorを生成)
    processor_ =
      std::make_unique<TrackerProcessor>(config, associator_config, input_channels_config_);
  }

  // Debugger(トラッキング処理の時間やTracker情報を収集するDebuggerを生成)
  debugger_ = std::make_unique<TrackerDebugger>(*this, world_frame_id_, input_channels_config_);

  // メッセージのpublish時刻をデバッグ用に出力するPublisherを生成
  published_time_publisher_ = std::make_unique<autoware_utils::PublishedTimePublisher>(this);

  if (use_time_keeper) {
    // 各処理時間をpublishするPublisherを生成
    detailed_processing_time_publisher_ =
      this->create_publisher<autoware_utils::ProcessingTimeDetail>(
        "~/debug/processing_time_detail_ms", 1);
    // 処理時間を計測・管理するTimeKeeperを生成
    time_keeper_ =
      std::make_shared<autoware_utils::TimeKeeper>(detailed_processing_time_publisher_);
    // TrackerProcessor内部でも同じTimeKeeperを使って処理時間を計測
    processor_->setTimeKeeper(time_keeper_);
  }
}

void MultiObjectTracker::onTrigger()
{
  // この関数が呼ばれるまでの流れ：DetectedObjects受信 → InputStream::onMessage() → InputManager::onTrigger() → MultiObjectTracker::onTrigger()

  // TimeKeeperが有効な場合、onTrigger()全体の処理時間を計測
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // 現在時刻を取得
  const rclcpp::Time current_time = this->now();

  // get objects from the input manager and run process
  // InputStreamがDetectedObjectsを受信して内部queueへ保存した後、InputManager経由でこのonTrigger()が呼ばれる
  // objects_listには、各InputStreamのqueueから今回処理対象となるDynamicObjectListを集め、timestamp順に並べたものが格納される
  ObjectsList objects_list;
  const bool is_objects_ready = input_manager_->getObjects(current_time, objects_list);

  // 処理対象のDetectionがなければ終了
  if (!is_objects_ready) return;

  // process start
  // 最終更新時刻を保存
  last_updated_time_ = current_time; 

  // 最新Detectionの時刻を取得
  const rclcpp::Time latest_time(objects_list.back().header.stamp);

  // デバッグ用の時間計測開始
  debugger_->startMeasurementTime(this->now(), latest_time);
  // run process for each DynamicObject
  for (const auto & objects_data : objects_list) {
    runProcess(objects_data);
  }
  // process end
  debugger_->endMeasurementTime(this->now());

  // Publish without delay compensation(Delay Compensationなしの場合はDetection処理後すぐにpublish)
  if (!publish_timer_) {
    const auto latest_object_time = rclcpp::Time(objects_list.back().header.stamp);
    checkAndPublish(latest_object_time);
  }
}

void MultiObjectTracker::onTimer()
{
  // TimeKeeperが有効な場合、onTimer()全体の処理時間を計測
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // Trackerの最終更新時刻が未設定の場合は現在時刻で初期化
  const rclcpp::Time current_time = this->now();
  if (last_updated_time_.nanoseconds() == 0) {
    // If the last updated time is not set, set it to the current time
    last_updated_time_ = current_time;
  }

  // ensure minimum interval: room for the next process(prediction)
  // 前回publishから十分な時間(default:85ms)が経過していなければ何もしない
  const double minimum_publish_interval = publisher_period_ * minimum_publish_interval_ratio; // 0.1×0.85 = 85ms
  const auto elapsed_time = (current_time - last_published_time_).seconds();
  if (elapsed_time < minimum_publish_interval) {
    return;
  }

  // if there was update after publishing, publish new messages
  // 前回publish後に新しいDetectionでTrackerが更新されていればpublish対象
  bool should_publish = last_published_time_ < last_updated_time_;

  // if there was no update, publish if the elapsed time is longer than the maximum publish latency
  // in this case, it will perform extrapolate/remove old objects
  // Detection更新がなくても、最大publish間隔(default:105ms)を超えたら予測結果をpublish
  const double maximum_publish_interval = publisher_period_ * maximum_publish_interval_ratio; // 0.1×1.05 = 105ms
  should_publish = should_publish || elapsed_time > maximum_publish_interval;

  // Publish with delay compensation to the current time
  // Delay Compensationにより現在時刻まで予測したTrackedObjectsをpublish
  if (should_publish) checkAndPublish(last_published_time_);
}

void MultiObjectTracker::runProcess(const types::DynamicObjectList & detected_objects)
{
  // TimeKeeperが有効な場合、runProcess()全体の処理時間を計測
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // Get the time of the measurement
  // Detectionの計測時刻を取得
  const rclcpp::Time measurement_time =
    rclcpp::Time(detected_objects.header.stamp, this->now().get_clock_type());

  // Get ego pose at the measurement time
  // Detection計測時刻におけるmap座標系での自車Poseを取得
  std::optional<geometry_msgs::msg::Pose> ego_pose;
  if (const auto odometry_info = odometry_->getOdometryFromTf(measurement_time)) {
    ego_pose = odometry_info->pose.pose;
  } else {
    RCLCPP_WARN(
      this->get_logger(), "No odometry information available at the measurement time: %.9f",
      measurement_time.seconds());
    ego_pose = std::nullopt;
  }

  /* predict trackers to the measurement time */
  // 既存Trackerを計測時刻まで予測
  processor_->predict(measurement_time, ego_pose);

  /* object association */
  // TrackerとDetectionを対応付け
  std::unordered_map<int, int> direct_assignment, reverse_assignment;
  processor_->associate(detected_objects, direct_assignment, reverse_assignment);

  // Collect debug information - tracker list, existence probabilities, association results
  // Debug情報収集
  debugger_->collectObjectInfo(
    measurement_time, processor_->getListTracker(), detected_objects, direct_assignment,
    reverse_assignment);

  /* tracker update */
  // 既存Trackerを更新
  processor_->update(detected_objects, direct_assignment);

  /* tracker pruning */
  // 不要Trackerを削除・整理
  processor_->prune(measurement_time);

  /* spawn new tracker */
  // 未対応Detectionから新規Tracker生成
  processor_->spawn(detected_objects, reverse_assignment);
}

void MultiObjectTracker::checkAndPublish(const rclcpp::Time & time)
{
  // TimeKeeperが有効な場合、checkAndPublish()全体の処理時間を計測
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  /* tracker pruning*/
  // publish前に古い・不確かな・重複Trackerを整理
  processor_->prune(time);

  // Publish
  // TrackedObjectsを生成してpublish
  publish(time);

  // Update last published time
  // 実際にpublish処理を行った時刻を保存
  last_published_time_ = this->now();
}

void MultiObjectTracker::publish(const rclcpp::Time & time) const
{
  // publish()全体の処理時間を計測
  std::unique_ptr<ScopedTimeTrack> st_ptr;
  if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

  // publish開始時刻をDebuggerに記録
  debugger_->startPublishTime(this->now());

  // TrackedObjectsのsubscriberがいなければpublish処理を省略
  const auto subscriber_count = tracked_objects_pub_->get_subscription_count() +
                                tracked_objects_pub_->get_intra_process_subscription_count();
  if (subscriber_count < 1) {
    return;
  }

  // Create output msg
  autoware_perception_msgs::msg::TrackedObjects output_msg;
  output_msg.header.frame_id = world_frame_id_; // world_frame_id_="map"

  // Delay compensation有効時は現在時刻までTrackerを外挿して出力する
  const rclcpp::Time object_time = enable_delay_compensation_ ? this->now() : time;

  // publish用に指定時刻まで外挿したTrackedObjectを取得
  // この外挿ではTracker内部の状態自体は更新しない
  processor_->getTrackedObjects(object_time, output_msg);

  // Publish
  // TrackedObjectsのpublish処理時間を計測してROS topicへpublish
  {
    std::unique_ptr<ScopedTimeTrack> st_pub_ptr;
    if (time_keeper_)
      st_pub_ptr = std::make_unique<ScopedTimeTrack>("tracker_publish", *time_keeper_);
    tracked_objects_pub_->publish(output_msg);
  }

  // debug
  {
    // debugスコープ内の処理時間を計測
    std::unique_ptr<ScopedTimeTrack> st_debug_ptr;
    if (time_keeper_)
      st_debug_ptr = std::make_unique<ScopedTimeTrack>("debug_publish", *time_keeper_);

    // 実際に出したメッセージのtimestamp情報
    published_time_publisher_->publish_if_subscribed(tracked_objects_pub_, output_msg.header.stamp);

    // Publish debugger information if enabled
    debugger_->endPublishTime(this->now(), time);

    // Update the diagnostic values
    const double min_extrapolation_time = (time - last_updated_time_).seconds();
    debugger_->updateDiagnosticValues(min_extrapolation_time, output_msg.objects.size());

    if (debugger_->shouldPublishTentativeObjects()) {
      autoware_perception_msgs::msg::TrackedObjects tentative_output_msg;
      tentative_output_msg.header.frame_id = world_frame_id_;
      processor_->getTentativeObjects(time, tentative_output_msg);
      debugger_->publishTentativeObjects(tentative_output_msg);
    }
    debugger_->publishObjectsMarkers();
  }
}

}  // namespace autoware::multi_object_tracker

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::multi_object_tracker::MultiObjectTracker)
