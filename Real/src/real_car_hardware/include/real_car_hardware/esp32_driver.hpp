#ifndef ESP32_DRIVER_HPP
#define ESP32_DRIVER_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include <errno.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <iomanip>

class Esp32Driver
{
public:
  static constexpr uint8_t START_BYTE = 0xAA;
  static constexpr uint8_t END_BYTE   = 0x55;

  // RX mới: [AA][L pos][R pos][55]
  static constexpr std::size_t RX_PACKET_SIZE = 10;

  // TX giữ nguyên: [AA][L vel][R vel][55]
  static constexpr std::size_t TX_PACKET_SIZE = 10;

  // Buffer tích lũy dữ liệu RX giữa nhiều lần read()
  static constexpr std::size_t RX_BUFFER_CAPACITY = 2048;

  explicit Esp32Driver(const std::string & device_name)
  : device_name_(device_name), fd_(-1), rx_size_(0)
  {
  }

  ~Esp32Driver()
  {
    close_port();
  }

  bool init()
  {
    std::cout << "Initializing serial connection with: " << device_name_ << std::endl;

    close_port();

    fd_ = ::open(device_name_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
      std::cerr << "Failed to open port: " << device_name_
                << " error=" << std::strerror(errno) << std::endl;
      return false;
    }

    termios tty{};
    if (::tcgetattr(fd_, &tty) != 0) {
      std::cerr << "Error getting termios attributes: "
                << std::strerror(errno) << std::endl;
      close_port();
      return false;
    }

    // fixed 115200
    ::cfsetospeed(&tty, B115200);
    ::cfsetispeed(&tty, B115200);

    // Raw mode cơ bản cho binary packet
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;        // 8 data bits
    tty.c_cflag &= ~PARENB;    // no parity
    tty.c_cflag &= ~CSTOPB;    // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;   // no hw flow control

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY |
                     IGNBRK | BRKINT | PARMRK | ISTRIP |
                     INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~OPOST;

    // Non-canonical + không chờ byte
    // read() trả ngay:
    // - >0 nếu có dữ liệu
    // - 0 nếu chưa có dữ liệu
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;

    if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
      std::cerr << "Error setting termios attributes: "
                << std::strerror(errno) << std::endl;
      close_port();
      return false;
    }

    ::tcflush(fd_, TCIOFLUSH);
    rx_size_ = 0;

    std::cout << "Succeeded to open and configure the port at 115200 8N1"
              << std::endl;

    
    return true;
  }

  void close_port()
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    rx_size_ = 0;
  }

  bool is_open() const
  {
    return fd_ >= 0;
  }

  // Đọc state mới nhất có thể parse được.
  // Nếu chưa có frame hoàn chỉnh mới, hàm trả false.
  bool read_state(float & left_pos, float & right_pos)
  {
    if (fd_ < 0) {
      std::cout << "[driver] fd closed\n";
      return false;
    }

    if (!read_available_bytes()) {
      std::cout << "[driver] read_available_bytes failed\n";
      return false;
    }

    const bool ok = parse_latest_packet(left_pos, right_pos);

    static int cnt = 0;
    cnt++;
    if (cnt % 50 == 0) {
      std::cout << "[driver] rx_size_=" << rx_size_
                << " parse_ok=" << ok << std::endl;
    }

    return ok;
  }

  bool write_command(float left_vel, float right_vel)
  {
    if (fd_ < 0) {
      return false;
    }

    std::array<uint8_t, TX_PACKET_SIZE> packet{};
    packet[0] = START_BYTE;
    put_float(&packet[1], left_vel);
    put_float(&packet[5], right_vel);
    packet[9] = END_BYTE;

    std::size_t total_written = 0;
    while (total_written < packet.size()) {
      const ssize_t n = ::write(
        fd_,
        packet.data() + total_written,
        packet.size() - total_written);

      if (n > 0) {
        total_written += static_cast<std::size_t>(n);
        continue;
      }

      if (n < 0 && errno == EINTR) {
        continue;
      }

      std::cerr << "Serial write failed: " << std::strerror(errno) << std::endl;
      return false;
    }

    ::tcdrain(fd_);
    return true;
  }

private:
  bool read_available_bytes()
  {
    std::array<uint8_t, 256> temp{};

    while (true) {
      const ssize_t n = ::read(fd_, temp.data(), temp.size());

      if (n > 0) {
        append_rx_bytes(temp.data(), static_cast<std::size_t>(n));
        continue;
      }
      if (n == 0) break; 
      
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) break; 

      std::cerr << "Serial read failed: " << std::strerror(errno) << std::endl;
      return false;
    }
    return true;
  }

  void append_rx_bytes(const uint8_t * data, std::size_t len)
  {
    if (len == 0) {
      return;
    }

    // Nếu chunk mới quá lớn, chỉ giữ phần cuối cùng
    if (len >= RX_BUFFER_CAPACITY) {
      std::memcpy(rx_buffer_.data(), data + (len - RX_BUFFER_CAPACITY), RX_BUFFER_CAPACITY);
      rx_size_ = RX_BUFFER_CAPACITY;
      return;
    }

    // Nếu tràn buffer, bỏ dữ liệu cũ nhất để giữ dữ liệu mới nhất
    if (rx_size_ + len > RX_BUFFER_CAPACITY) {
      const std::size_t overflow = (rx_size_ + len) - RX_BUFFER_CAPACITY;

      if (overflow < rx_size_) {
        std::memmove(rx_buffer_.data(), rx_buffer_.data() + overflow, rx_size_ - overflow);
        rx_size_ -= overflow;
      } else {
        rx_size_ = 0;
      }
    }

    std::memcpy(rx_buffer_.data() + rx_size_, data, len);
    rx_size_ += len;
  }

  bool parse_latest_packet(float & left_pos, float & right_pos)
  {
    if (rx_size_ < RX_PACKET_SIZE) {
      return false;
    }

    // Quét ngược để lấy packet mới nhất
    for (int i = static_cast<int>(rx_size_ - RX_PACKET_SIZE); i >= 0; --i) {
      const std::size_t idx = static_cast<std::size_t>(i);

      if (rx_buffer_[idx] != START_BYTE) {
        continue;
      }

      if (rx_buffer_[idx + RX_PACKET_SIZE - 1] != END_BYTE) {
        continue;
      }

      left_pos  = get_float(&rx_buffer_[idx + 1]);
      right_pos = get_float(&rx_buffer_[idx + 5]);

      // Bỏ luôn packet vừa parse và toàn bộ dữ liệu cũ hơn nó,
      // chỉ giữ phần phía sau để ghép tiếp nếu đó là packet mới đang dở dang
      const std::size_t packet_end = idx + RX_PACKET_SIZE;
      const std::size_t remain = rx_size_ - packet_end;

      if (remain > 0) {
        std::memmove(rx_buffer_.data(), rx_buffer_.data() + packet_end, remain);
      }
      rx_size_ = remain;
      
      std::cout << "[driver] packet matched: left=" << left_pos
          << " right=" << right_pos << std::endl;

      return true;
    }

    // Không tìm thấy packet hợp lệ.
    // Giữ lại tối đa RX_PACKET_SIZE - 1 byte cuối để ghép với lần sau,
    // vì một packet mới có thể đang bị thiếu đoạn cuối.
    const std::size_t keep = (rx_size_ < (RX_PACKET_SIZE - 1))
      ? rx_size_
      : (RX_PACKET_SIZE - 1);

    if (keep > 0 && rx_size_ > keep) {
      std::memmove(rx_buffer_.data(), rx_buffer_.data() + (rx_size_ - keep), keep);
    }
    rx_size_ = keep;

    return false;
  }

  static void put_float(uint8_t * dst, float value)
  {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    std::memcpy(dst, &value, 4);
  }

  static float get_float(const uint8_t * src)
  {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    float value = 0.0f;
    std::memcpy(&value, src, 4);
    return value;
  }

private:
  std::string device_name_;
  int fd_;

  std::array<uint8_t, RX_BUFFER_CAPACITY> rx_buffer_{};
  std::size_t rx_size_;
};

#endif  // ESP32_DRIVER_HPP