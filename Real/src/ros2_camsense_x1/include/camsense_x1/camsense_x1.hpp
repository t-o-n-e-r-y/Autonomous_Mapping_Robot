#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex> // dùng khoá để bảo vệ dữ liệu ví dụ: ranges[]...
#include <cstdint>

#include <boost/asio.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

class CamsenseX1 : public rclcpp::Node
{
public:
  explicit CamsenseX1(
    const std::string & name = "camsense_x1_node",
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~CamsenseX1();

private:
  enum class State
  {
    SYNC1,
    SYNC2,
    SYNC3,
    SYNC4,
    SPEED,
    START,
    DATA,
    END,
    CRC
  };

  void serial_read(uint8_t * data, size_t len);
  void reset_scan();
  void parse_packet();
  void process_points();
  void publish_scan(const rclcpp::Time & stamp);

  boost::asio::io_service io_;
  boost::asio::serial_port serial_;

  std::thread serial_thread_;
  std::atomic<bool> canceled_{false};

  // std::mutex scan_mutex_;  hiện tại không cần vì mỗi thread phụ xử lý 

  State state_{State::SYNC1};

  std::string port_;
  int baud_{115200};
  std::string frame_id_{"lidar_link"};
  double angle_offset_{50.0};

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;


  rclcpp::Time scan_start_time_{0, 0, RCL_ROS_TIME};
  bool scan_started_{false};

  std::vector<float> ranges_;
  std::vector<float> intensities_;

  uint8_t buf_[2]{};
  uint8_t frame_[24]{};

  double motor_speed_rpm_{0.0};
  bool speed_printed_{false};
  double start_angle_{0.0};
  double end_angle_{0.0};

  double last_start_angle_{-1.0};
  // bool has_scan_started_{false};  not needing first reset() when run
};