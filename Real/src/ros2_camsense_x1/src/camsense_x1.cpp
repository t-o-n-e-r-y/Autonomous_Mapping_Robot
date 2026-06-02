#include <camsense_x1/camsense_x1.hpp>

#include <sensor_msgs/msg/laser_scan.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <boost/asio.hpp>
#include <cmath>
#include <thread>
#include <atomic>
#include <algorithm>
#include <limits>
#include <mutex> //dùng khoá để bảo vệ dữ liệu ví dụ: ranges[]...
#include <chrono>
#include <stdexcept>

constexpr uint8_t SYNC1 = 0x55;
constexpr uint8_t SYNC2 = 0xAA;
constexpr uint8_t SYNC3 = 0x03;
constexpr uint8_t SYNC4 = 0x08;

constexpr int POINTS_PER_REV = 400;
constexpr double DEG2RAD = M_PI / 180.0;

CamsenseX1::CamsenseX1(
  const std::string & name,
  const rclcpp::NodeOptions & options)
: Node(name, options),
  io_(),
  serial_(io_),
  canceled_(false),
  state_(State::SYNC1)
{
  declare_parameter<std::string>("port", "/dev/ttyUSB0");
  declare_parameter<int>("baudrate", 115200);
  declare_parameter<std::string>("frame_id", "lidar_link");
  declare_parameter<double>("angle_offset", 0.0);

  get_parameter("port", port_);
  get_parameter("baudrate", baud_);
  get_parameter("frame_id", frame_id_);
  get_parameter("angle_offset", angle_offset_);

  auto scan_qos = rclcpp::SensorDataQoS();
  scan_qos.keep_last(1);
  scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>("scan", scan_qos);

  ranges_.resize(POINTS_PER_REV);
  intensities_.resize(POINTS_PER_REV);
  reset_scan();

  try {
    serial_.open(port_);
    serial_.set_option(boost::asio::serial_port_base::baud_rate(baud_));
    serial_.set_option(boost::asio::serial_port_base::character_size(8));
    serial_.set_option(boost::asio::serial_port_base::parity(
      boost::asio::serial_port_base::parity::none));
    serial_.set_option(boost::asio::serial_port_base::stop_bits(
      boost::asio::serial_port_base::stop_bits::one));
    serial_.set_option(boost::asio::serial_port_base::flow_control(
      boost::asio::serial_port_base::flow_control::none));
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      get_logger(),
      "Cannot open serial port %s: %s",
      port_.c_str(),
      e.what());
    throw;
  }

  RCLCPP_INFO(get_logger(), "Camsense X1 connected on %s", port_.c_str());
  RCLCPP_INFO(get_logger(), "========== Camsense X1 Info ==========");
  RCLCPP_INFO(get_logger(), "Port        : %s", port_.c_str());
  RCLCPP_INFO(get_logger(), "Baudrate    : %d", baud_);
  RCLCPP_INFO(get_logger(), "Frame ID    : %s", frame_id_.c_str());
  RCLCPP_INFO(get_logger(), "Angle offset: %.2f deg", angle_offset_);
  RCLCPP_INFO(get_logger(), "Resolution  : %d points/rev", POINTS_PER_REV);
  RCLCPP_INFO(get_logger(), "=====================================");

  serial_thread_ = std::thread([this]() {
    while (rclcpp::ok() && !canceled_) {
      try {
        parse_packet();
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Serial parse error: %s", e.what());
      }
    }
  });
}

CamsenseX1::~CamsenseX1()
{
  canceled_ = true;

  if (serial_.is_open()) {
    boost::system::error_code ec;
    serial_.cancel(ec);
    serial_.close(ec);
    // tạm thời đang không check ec là gì
  }

  if (serial_thread_.joinable()) {
    serial_thread_.join();
  }
}

void CamsenseX1::serial_read(uint8_t * data, size_t len)
{
  boost::asio::read(serial_, boost::asio::buffer(data, len));
}

void CamsenseX1::reset_scan()
{
  std::fill(
    ranges_.begin(),
    ranges_.end(),
    std::numeric_limits<float>::infinity());

  std::fill(
    intensities_.begin(),
    intensities_.end(),
    0.0f);
}

void CamsenseX1::parse_packet()
{
  switch (state_) {
    case State::SYNC1:
      serial_read(buf_, 1);
      state_ = (buf_[0] == SYNC1) ? State::SYNC2 : State::SYNC1;
      break;

    case State::SYNC2:
      serial_read(buf_, 1);
      state_ = (buf_[0] == SYNC2) ? State::SYNC3 : State::SYNC1;
      break;

    case State::SYNC3:
      serial_read(buf_, 1);
      state_ = (buf_[0] == SYNC3) ? State::SYNC4 : State::SYNC1;
      break;

    case State::SYNC4:
      serial_read(buf_, 1);
      state_ = (buf_[0] == SYNC4) ? State::SPEED : State::SYNC1;
      break;

    case State::SPEED: {
      serial_read(buf_, 2);
      uint16_t raw_speed = static_cast<uint16_t>(buf_[0]) |
                           (static_cast<uint16_t>(buf_[1]) << 8);

      motor_speed_rpm_ = raw_speed / 64.0;

      if (!speed_printed_ && motor_speed_rpm_ > 1.0) {
        RCLCPP_INFO(
          get_logger(),
          "Lidar speed : %.2f RPM (%.2f Hz)",
          motor_speed_rpm_,
          motor_speed_rpm_ / 60.0);
        speed_printed_ = true;
      }

      state_ = State::START;
      break;
    }

    case State::START: {
      serial_read(buf_, 2);
      uint16_t raw = static_cast<uint16_t>(buf_[0]) |
                     (static_cast<uint16_t>(buf_[1]) << 8);
      start_angle_ = raw / 64.0 - 640.0;
      state_ = State::DATA;
      break;
    }

    case State::DATA:
      serial_read(frame_, 24);
      state_ = State::END;
      break;

    case State::END: {
      serial_read(buf_, 2);
      uint16_t raw = static_cast<uint16_t>(buf_[0]) |
                     (static_cast<uint16_t>(buf_[1]) << 8);
      end_angle_ = raw / 64.0 - 640.0;
      state_ = State::CRC;
      break;
    }

    case State::CRC:
      serial_read(buf_, 2);  // tạm bỏ qua CRC
      process_points();
      state_ = State::SYNC1;
      break;
  }
}

void CamsenseX1::process_points()
{
  const rclcpp::Time packet_stamp = now();

  // packet đầu tiên của vòng scan hiện tại
  if (!scan_started_) {
    scan_start_time_ = packet_stamp;
    scan_started_ = true;
  }

  // Góc quay wrap về đầu vòng -> publish vòng cũ, reset vòng mới
  if (last_start_angle_ >= 0.0 && start_angle_ < last_start_angle_) {
    publish_scan(scan_start_time_);
    reset_scan();
    scan_start_time_ = packet_stamp;   // packet hiện tại là packet đầu tiên của vòng mới
  }

  last_start_angle_ = start_angle_;

  double span = end_angle_ - start_angle_;
  if (span < 0.0) {
    span += 360.0;
  }

  // 8 điểm -> 7 khoảng
  double step = span / 7.0;

  for (int i = 0; i < 8; i++) {
    int j = i * 3;

    uint16_t raw_range = static_cast<uint16_t>(frame_[j]) |
                         (static_cast<uint16_t>(frame_[j + 1]) << 8);
    uint8_t intensity = frame_[j + 2];

    if (raw_range == 0 || raw_range == 0x8000) {
      continue;
    }

    double angle = start_angle_ + step * i - angle_offset_;
    while (angle < 0.0) {
      angle += 360.0;
    }
    while (angle >= 360.0) {
      angle -= 360.0;
    }

    int idx = static_cast<int>(
      std::round(angle / 360.0 * static_cast<double>(POINTS_PER_REV - 1)));
    idx = std::clamp(idx, 0, POINTS_PER_REV - 1);

    idx = (POINTS_PER_REV - 1) - idx;

    float r = static_cast<float>(raw_range) / 1000.0f;
    if (r < 0.008f || r > 8.0f) {
      continue;
    }

    ranges_[idx] = r;
    intensities_[idx] = static_cast<float>(intensity);
  }
}

void CamsenseX1::publish_scan(const rclcpp::Time & stamp)
{
  sensor_msgs::msg::LaserScan scan;
  (void)stamp;
  scan.header.stamp = now() - rclcpp::Duration::from_seconds(0.10);
  scan.header.frame_id = frame_id_;

  scan.angle_min = 0.0f;
  scan.angle_max = static_cast<float>(2.0 * M_PI);
  scan.angle_increment = static_cast<float>(2.0 * M_PI / POINTS_PER_REV);

  if (motor_speed_rpm_ > 1.0) {
    scan.scan_time = static_cast<float>(60.0 / motor_speed_rpm_);
  } else {
    scan.scan_time = 0.1f;
  }

  scan.time_increment = scan.scan_time / static_cast<float>(POINTS_PER_REV);

  scan.range_min = 0.08f;
  scan.range_max = 8.0f;

  scan.ranges = ranges_;
  scan.intensities = intensities_;

  scan_pub_->publish(scan);
}