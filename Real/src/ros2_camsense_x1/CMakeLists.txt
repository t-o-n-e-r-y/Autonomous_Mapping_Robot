cmake_minimum_required(VERSION 3.8)
project(ros2_camsense_x1)

# Default standards
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================
# Dependencies
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# Boost.Asio
find_package(Boost REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

# =========================
# Build
# =========================

# --- 1. Target cho Node gốc (camsense_x1) ---
add_executable(camsense_x1_node
  src/camsense_x1_node.cpp
  src/camsense_x1.cpp
)
target_include_directories(camsense_x1_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(camsense_x1_node rclcpp sensor_msgs)
target_link_libraries(camsense_x1_node Boost::system Threads::Threads)




# =========================
# Install
# =========================
install(TARGETS 
  camsense_x1_node
  DESTINATION lib/${PROJECT_NAME}
)

# --- Install configs + launch files ---
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# =========================
# Testing
# =========================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()