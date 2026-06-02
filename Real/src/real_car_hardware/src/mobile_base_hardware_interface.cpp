#include "real_car_hardware/mobile_base_hardware_interface.hpp"

#include <cmath>

// giả sử esp32 trả về đi thẳng vel left dương, vel right âm


namespace mobile_base_hardware {

hardware_interface::CallbackReturn MobileBaseHardwareInterface::on_init
    (const hardware_interface::HardwareComponentInterfaceParams & params)
{
    if (
    hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }


    // left_motor_id_ = std::stoi(info_.hardware_parameters["left_motor_id"]);
    // right_motor_id_ = std::stoi(info_.hardware_parameters["right_motor_id"]);
    port_ = info_.hardware_parameters["esp32_port"];

    driver_ = std::make_shared<Esp32Driver>(port_);

    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MobileBaseHardwareInterface::on_configure
    (const rclcpp_lifecycle::State & previous_state)
{
    (void)previous_state;
    if (!driver_->init()) {
        RCLCPP_ERROR(get_logger(), "Failed to initialize ESP32 driver on port %s", port_.c_str());
        return hardware_interface::CallbackReturn::ERROR;
    }

    // for (const auto & [name, descr] : joint_command_interfaces_)
    // {
    //     RCLCPP_INFO(get_logger(), "COMMAND INTERFACE NAME: ");
    //     RCLCPP_INFO(get_logger(), name.c_str());
    // }
    // for (const auto & [name, descr] : joint_state_interfaces_)
    // {
    //     RCLCPP_INFO(get_logger(), "STATE INTERFACE NAME: ");
    //     RCLCPP_INFO(get_logger(), name.c_str());
    // }

    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MobileBaseHardwareInterface::on_activate
    (const rclcpp_lifecycle::State & previous_state)
{
    (void)previous_state;
    left_vel_ = 0.0;
    right_vel_ = 0.0;
    left_pos_ = 0.0;
    right_pos_ = 0.0;

    prev_left_pos_ = 0.0;
    prev_right_pos_ = 0.0;
    have_prev_pos_ = false;

    first_write_ = true;
    last_write_left_vel_ = 0.0f;
    last_write_right_vel_ = 0.0f;

    last_packet_time_ = rclcpp::Time{0, 0, RCL_STEADY_TIME};
    have_last_packet_time_ = false;

    set_state("base_left_wheel_joint/velocity", 0.0);
    set_state("base_right_wheel_joint/velocity", 0.0);
    set_state("base_left_wheel_joint/position", 0.0);
    set_state("base_right_wheel_joint/position", 0.0);
    if (!driver_->write_command(0.0f, 0.0f)) {
        // return hardware_interface::CallbackReturn::ERROR;
        RCLCPP_WARN(get_logger(), "Failed to write command to ESP32");
    }
    RCLCPP_INFO(get_logger(), "on_activate");
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MobileBaseHardwareInterface::on_deactivate
    (const rclcpp_lifecycle::State & previous_state)
{
    (void)previous_state;
    left_vel_ = 0.0f;
    right_vel_ = 0.0f;
    set_state("base_left_wheel_joint/velocity", 0.0);
    set_state("base_right_wheel_joint/velocity", 0.0);
    if (!driver_->write_command(0.0f, 0.0f)) {
        RCLCPP_WARN(get_logger(), "Failed to write command to ESP32");
    }
    RCLCPP_INFO(get_logger(), "deactivate");
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type MobileBaseHardwareInterface::read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period)
{
    (void)period;

    float left_pos = 0.0f;
    float right_pos = 0.0f;

        // const bool got_packet = driver_->read_state(left_pos, right_pos); 
    // hiện tại do xe quay bị sai nên ta mới đảo, code thực tế theo quy tắc là chính là hàm này
    const bool got_packet = driver_->read_state(right_pos, left_pos); // cai nay fake, đảo right với left
    
    if (got_packet) {
        const double new_left_pos = static_cast<double>(left_pos);
        const double new_right_pos = static_cast<double>(right_pos);

        if (have_prev_pos_ && have_last_packet_time_) {
            const double dt_packet = (time - last_packet_time_).seconds();

            if (dt_packet > 1e-6) {
                left_vel_ = (new_left_pos - prev_left_pos_) / dt_packet;
                right_vel_ = (new_right_pos - prev_right_pos_) / dt_packet;
            } else {
                left_vel_ = 0.0;
                right_vel_ = 0.0;
            }
        } else {
            left_vel_ = 0.0;
            right_vel_ = 0.0;
            have_prev_pos_ = true;
            have_last_packet_time_ = true;
        }

        if (std::fabs(left_vel_) < 0.03) {
            left_vel_ = 0.0;
        }
        if (std::fabs(right_vel_) < 0.03) {
            right_vel_ = 0.0;
        }

        left_pos_ = new_left_pos;
        right_pos_ = new_right_pos;

        prev_left_pos_ = new_left_pos;
        prev_right_pos_ = new_right_pos;
        last_packet_time_ = time;

        RCLCPP_INFO(
            rclcpp::get_logger("mobile_base_hw"),
            "packet ok | pos=(%.4f, %.4f) vel=(%.4f, %.4f)",
            left_pos_, right_pos_, left_vel_, right_vel_);
    }

    // luôn publish state gần nhất
    set_state("base_left_wheel_joint/position", left_pos_);
    set_state("base_right_wheel_joint/position", right_pos_);
    set_state("base_left_wheel_joint/velocity", left_vel_);
    set_state("base_right_wheel_joint/velocity", right_vel_);

    return hardware_interface::return_type::OK;
}


hardware_interface::return_type MobileBaseHardwareInterface::write(
    const rclcpp::Time & time,
    const rclcpp::Duration & period)
{
    (void)period;

    float left_vel =
        static_cast<float>(get_command("base_left_wheel_joint/velocity"));

    float right_vel =
        -static_cast<float>(get_command("base_right_wheel_joint/velocity"));

    // Chặn command lỗi
    if (!std::isfinite(left_vel) || !std::isfinite(right_vel)) {
        RCLCPP_WARN(
            get_logger(),
            "Non-finite command detected: left=%f right=%f",
            left_vel,
            right_vel);

        return hardware_interface::return_type::OK;
    }

    // Nếu cache cũ bị NaN/Inf thì reset
    if (!std::isfinite(last_write_left_vel_) ||
        !std::isfinite(last_write_right_vel_))
    {
        RCLCPP_WARN(
            get_logger(),
            "last_write_* is NaN/Inf, resetting write cache");

        first_write_ = true;
        last_write_left_vel_ = 0.0f;
        last_write_right_vel_ = 0.0f;
    }

    constexpr float EPS = 1e-4f;

    // Gửi heartbeat mỗi 50 ms = 20 Hz
    constexpr double HEARTBEAT_SEC = 0.05;

    const float diff_left = std::fabs(left_vel - last_write_left_vel_);
    const float diff_right = std::fabs(right_vel - last_write_right_vel_);

    const bool changed =
        first_write_ ||
        diff_left > EPS ||
        diff_right > EPS;

    const bool heartbeat_due =
        first_write_ ||
        !have_last_serial_write_time_ ||
        (time - last_serial_write_time_).seconds() >= HEARTBEAT_SEC;

    RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        500,
        "write() | left=%.4f last_left=%.4f diff_left=%.6f | "
        "right=%.4f last_right=%.4f diff_right=%.6f | "
        "changed=%d heartbeat_due=%d",
        left_vel,
        last_write_left_vel_,
        diff_left,
        right_vel,
        last_write_right_vel_,
        diff_right,
        static_cast<int>(changed),
        static_cast<int>(heartbeat_due));

    // Không đổi và chưa tới thời gian heartbeat thì không gửi
    if (!changed && !heartbeat_due) {
        return hardware_interface::return_type::OK;
    }

    RCLCPP_INFO(
        get_logger(),
        "SERIAL WRITE HEARTBEAT -> left=%.4f right=%.4f",
        left_vel,
        right_vel);

    const bool write_driver = driver_->write_command(left_vel, right_vel);

    if (!write_driver) {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "Failed to write command to ESP32");

        return hardware_interface::return_type::OK;
    }

    last_write_left_vel_ = left_vel;
    last_write_right_vel_ = right_vel;
    last_serial_write_time_ = time;
    have_last_serial_write_time_ = true;
    first_write_ = false;

    return hardware_interface::return_type::OK;
}

} // namespace mobile_base_hardware

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(mobile_base_hardware::MobileBaseHardwareInterface, hardware_interface::SystemInterface)