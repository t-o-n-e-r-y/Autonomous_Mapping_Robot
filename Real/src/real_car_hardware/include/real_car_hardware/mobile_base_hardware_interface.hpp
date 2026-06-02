#ifndef MOBILE_BASE_HARDWARE_INTERFACE_HPP
#define MOBILE_BASE_HARDWARE_INTERFACE_HPP

#include "hardware_interface/system_interface.hpp"
#include "real_car_hardware/esp32_driver.hpp"

namespace mobile_base_hardware {

class MobileBaseHardwareInterface : public hardware_interface::SystemInterface
{
public:
    // Lifecycle node override
    hardware_interface::CallbackReturn
        on_configure(const rclcpp_lifecycle::State & previous_state) override;
    hardware_interface::CallbackReturn
        on_activate(const rclcpp_lifecycle::State & previous_state) override;
    hardware_interface::CallbackReturn
        on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

    // SystemInterface override
    hardware_interface::CallbackReturn 
        on_init(const hardware_interface::HardwareComponentInterfaceParams & params) override;
    hardware_interface::return_type
        read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
    hardware_interface::return_type
        write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
    std::shared_ptr<Esp32Driver> driver_;
    std::string port_;

    double left_vel_ = 0.0;
    double right_vel_ = 0.0;
    double left_pos_ = 0.0;
    double right_pos_ = 0.0;
    double prev_left_pos_ = 0.0;
    double prev_right_pos_ = 0.0;
    bool have_prev_pos_ = false;

    bool first_write_ = true;
    float last_write_left_vel_ = 0.0f;
    float last_write_right_vel_ = 0.0f;
    
    rclcpp::Time last_packet_time_{0, 0, RCL_STEADY_TIME};
    bool have_last_packet_time_ = false;

    bool have_last_serial_write_time_ = false;
    rclcpp::Time last_serial_write_time_;
}; // class MobileBaseHardwareInterface

} // namespace mobile_base_hardware


#endif