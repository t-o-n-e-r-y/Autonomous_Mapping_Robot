#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from std_msgs.msg import Int32
from nav_msgs.msg import Path, GridCells, OccupancyGrid
from geometry_msgs.msg import PointStamped, TwistStamped, Pose, Point, Quaternion
from tf2_ros import Buffer, TransformListener, TransformException

from .path_planner import PathPlanner


class PurePursuitNode(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit")

        # ============================================================
        # Parameters: topics
        # ============================================================
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("cmd_vel_topic", "/con_vel")
        self.declare_parameter("tele_vel_topic", "/tele_vel")

        self.declare_parameter("lookahead_topic", "/lookahead")
        self.declare_parameter("fov_cells_topic", "/fov_cells")
        self.declare_parameter("close_wall_cells_topic", "/close_wall_cells")
        self.declare_parameter("replan_request_topic", "/replan")

        # ============================================================
        # Parameters: frames / timing
        # ============================================================
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("control_frequency", 20.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.05)
        self.declare_parameter("teleop_free_timeout_sec", 3.0)
        self.declare_parameter("replan_request_cooldown_sec", 1.0)

        # ============================================================
        # Parameters: behavior / debug
        # ============================================================
        self.declare_parameter("debug", True)
        self.declare_parameter("allow_reverse", False)
        self.declare_parameter("start_in_free_mode", True)

        # ============================================================
        # Parameters: pure pursuit
        # ============================================================
        self.declare_parameter("lookahead_distance", 0.18)      # m
        self.declare_parameter("max_drive_speed", 0.08)         # m/s
        self.declare_parameter("max_turn_speed", 0.6)           # rad/s
        self.declare_parameter("turn_speed_kp", 1.25)
        self.declare_parameter("distance_tolerance", 0.10)      # m
        self.declare_parameter("heading_align_threshold_deg", 35.0)
        self.declare_parameter("heading_align_turn_kp", 1.8)
        self.declare_parameter("final_yaw_tolerance_deg", 8.0)
        self.declare_parameter("final_align_turn_kp", 1.5)

        # ============================================================
        # Parameters: forward obstacle FOV
        # ============================================================
        self.declare_parameter("forward_fov_deg", 20.0)
        self.declare_parameter("forward_fov_distance_cells", 60)
        self.declare_parameter("forward_ignore_radius_cells", 2)

        # ============================================================
        # Parameters: slow down around robot
        # ============================================================
        self.declare_parameter("obstacle_avoidance_max_slow_down_distance", 0.16)  # m
        self.declare_parameter("obstacle_avoidance_min_slow_down_distance", 0.12)  # m
        self.declare_parameter("obstacle_avoidance_min_slow_down_factor", 0.25)
        self.declare_parameter("slow_down_check_radius_cells", 10)
        self.declare_parameter("slow_down_ignore_radius_cells", 2)

        # ============================================================
        # Parameters: completed goal handling
        # ============================================================
        self.declare_parameter("completed_goal_endpoint_tolerance", 0.05)  # m

        # ============================================================
        # Read parameters
        # ============================================================
        self.map_topic = self.get_parameter("map_topic").value
        self.path_topic = self.get_parameter("path_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.tele_vel_topic = self.get_parameter("tele_vel_topic").value

        self.lookahead_topic = self.get_parameter("lookahead_topic").value
        self.fov_cells_topic = self.get_parameter("fov_cells_topic").value
        self.close_wall_cells_topic = self.get_parameter("close_wall_cells_topic").value
        self.replan_request_topic = self.get_parameter("replan_request_topic").value

        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.control_frequency = float(self.get_parameter("control_frequency").value)
        self.tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)
        self.teleop_free_timeout_sec = float(self.get_parameter("teleop_free_timeout_sec").value)


        self.replan_request_cooldown_sec = float(
            self.get_parameter("replan_request_cooldown_sec").value
        )
        
        self.is_in_debug_mode = bool(self.get_parameter("debug").value)
        self.allow_reverse = bool(self.get_parameter("allow_reverse").value)
        self.start_in_free_mode = bool(self.get_parameter("start_in_free_mode").value)

        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.max_drive_speed = float(self.get_parameter("max_drive_speed").value)
        self.max_turn_speed = float(self.get_parameter("max_turn_speed").value)
        self.turn_speed_kp = float(self.get_parameter("turn_speed_kp").value)
        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)

        self.heading_align_threshold_rad = math.radians(
            float(self.get_parameter("heading_align_threshold_deg").value)
        )
        self.heading_align_turn_kp = float(self.get_parameter("heading_align_turn_kp").value)

        self.final_yaw_tolerance_rad = math.radians(
            float(self.get_parameter("final_yaw_tolerance_deg").value)
        )
        self.final_align_turn_kp = float(self.get_parameter("final_align_turn_kp").value)

        self.forward_fov_deg = float(self.get_parameter("forward_fov_deg").value)
        self.forward_fov_distance_cells = int(
            self.get_parameter("forward_fov_distance_cells").value
        )
        self.forward_ignore_radius_cells = int(
            self.get_parameter("forward_ignore_radius_cells").value
        )

        self.obstacle_avoidance_max_slow_down_distance = float(
            self.get_parameter("obstacle_avoidance_max_slow_down_distance").value
        )
        self.obstacle_avoidance_min_slow_down_distance = float(
            self.get_parameter("obstacle_avoidance_min_slow_down_distance").value
        )
        self.obstacle_avoidance_min_slow_down_factor = float(
            self.get_parameter("obstacle_avoidance_min_slow_down_factor").value
        )
        self.slow_down_check_radius_cells = int(
            self.get_parameter("slow_down_check_radius_cells").value
        )
        self.slow_down_ignore_radius_cells = int(
            self.get_parameter("slow_down_ignore_radius_cells").value
        )

        self.completed_goal_endpoint_tolerance = float(
            self.get_parameter("completed_goal_endpoint_tolerance").value
        )

        # ============================================================
        # State
        # ============================================================
        self.pose: Optional[Pose] = None
        self.pose_yaw: Optional[float] = None

        self.map_msg: Optional[OccupancyGrid] = None
        self.path_msg: Path = Path()

        self.alpha: float = 0.0
        self.reversed: bool = False
        self.closest_distance_m: float = float("inf")
        

        self.goal_reached_and_released: bool = self.start_in_free_mode
        self.released_goal_position: Optional[Tuple[float, float]] = None

        self.last_teleop_msg_time_ns: Optional[int] = None

        self.last_replan_request_time_ns: Optional[int] = None

        # ============================================================
        # TF
        # ============================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ============================================================
        # QoS
        # ============================================================
        default_qos = QoSProfile(depth=10)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ============================================================
        # Publishers
        # ============================================================
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            self.cmd_vel_topic,
            default_qos,
        )
        self.lookahead_pub = self.create_publisher(
            PointStamped,
            self.lookahead_topic,
            default_qos,
        )

        # Debug RViz
        self.fov_cells_pub = self.create_publisher(
            GridCells,
            self.fov_cells_topic,
            default_qos,
        )
        self.close_wall_cells_pub = self.create_publisher(
            GridCells,
            self.close_wall_cells_topic,
            default_qos,
        )

        # Functional topic
        self.replan_request_pub = self.create_publisher(
            Int32,
            self.replan_request_topic,
            default_qos,
        )

        # ============================================================
        # Subscribers
        # ============================================================
        self.tele_vel_sub = self.create_subscription(
            TwistStamped,
            self.tele_vel_topic,
            self.update_tele_vel,
            default_qos,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.update_map,
            map_qos,
        )
        self.path_sub = self.create_subscription(
            Path,
            self.path_topic,
            self.update_path,
            default_qos,
        )

        # ============================================================
        # Timer
        # ============================================================
        period = 1.0 / max(self.control_frequency, 1e-6)
        self.control_timer = self.create_timer(period, self.control_loop)

        self.get_logger().info("PurePursuitNode started.")

    # ================================================================
    # Basic callbacks
    # ================================================================
    def update_tele_vel(self, msg: TwistStamped) -> None:
        self.last_teleop_msg_time_ns = self.get_clock().now().nanoseconds

    def update_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def update_path(self, msg: Path) -> None:
        self.path_msg = msg

        if not self.goal_reached_and_released:
            return

        if not msg.poses:
            return

        new_goal = msg.poses[-1].pose.position

        # startup free mode
        if self.released_goal_position is None:
            self.goal_reached_and_released = False
            self.get_logger().info(
                "First valid path received. Controller leaves startup free mode."
            )
            return

        # free sau khi đã hoàn thành goal trước đó
        if not self.is_same_as_released_goal(new_goal):
            self.goal_reached_and_released = False
            self.released_goal_position = None
            self.get_logger().info(
                "A new path with a different final point was received. Controller re-enabled."
            )

    # ================================================================
    # TF / math helpers
    # ================================================================
    def is_teleop_free_active(self) -> bool:
        if self.last_teleop_msg_time_ns is None:
            return False

        elapsed_sec = (
            self.get_clock().now().nanoseconds - self.last_teleop_msg_time_ns
        ) * 1e-9
        return elapsed_sec < self.teleop_free_timeout_sec


    def update_pose_from_tf(self) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
        except TransformException:
            self.pose = None
            self.pose_yaw = None
            return

        t = transform.transform.translation
        r = transform.transform.rotation

        self.pose = Pose(
            position=Point(x=t.x, y=t.y, z=t.z),
            orientation=Quaternion(x=r.x, y=r.y, z=r.z, w=r.w),
        )
        self.pose_yaw = self.quaternion_to_yaw(r.x, r.y, r.z, r.w)

    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            return 0.0

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def distance_xy(x0: float, y0: float, x1: float, y1: float) -> float:
        return math.hypot(x1 - x0, y1 - y0)

    # ================================================================
    # Path helpers
    # ================================================================
    def is_same_as_released_goal(self, point: Point) -> bool:
        if self.released_goal_position is None:
            return False

        released_x, released_y = self.released_goal_position
        return (
            self.distance_xy(point.x, point.y, released_x, released_y)
            <= self.completed_goal_endpoint_tolerance
        )

    def release_controller_for_current_goal(self, goal_pose: Pose) -> None:
        self.stop()

        self.goal_reached_and_released = True
        self.released_goal_position = (
            goal_pose.position.x,
            goal_pose.position.y,
        )

        self.get_logger().info(
            "Goal completed. Controller released until a path with a different final point is received."
        )

    def get_distance_to_waypoint_index(self, i: int) -> float:
        if self.pose is None or not self.path_msg.poses:
            return -1.0

        position = self.pose.position
        waypoint = self.path_msg.poses[i].pose.position
        return self.distance_xy(position.x, position.y, waypoint.x, waypoint.y)

    def find_nearest_waypoint_index(self) -> int:
        if not self.path_msg.poses or self.pose is None:
            return -1

        nearest_waypoint_index = -1
        closest_distance = float("inf")

        for i in range(len(self.path_msg.poses)):
            distance = self.get_distance_to_waypoint_index(i)
            if 0.0 <= distance < closest_distance:
                closest_distance = distance
                nearest_waypoint_index = i

        return nearest_waypoint_index

    def find_lookahead(
        self,
        nearest_waypoint_index: int,
        lookahead_distance: float,
    ) -> Optional[Point]:
        if not self.path_msg.poses:
            return None

        if nearest_waypoint_index < 0:
            return self.path_msg.poses[-1].pose.position

        for i in range(nearest_waypoint_index, len(self.path_msg.poses)):
            if self.get_distance_to_waypoint_index(i) >= lookahead_distance:
                return self.path_msg.poses[i].pose.position

        return self.path_msg.poses[-1].pose.position

    def get_goal_pose(self) -> Optional[Pose]:
        if not self.path_msg.poses:
            return None
        return self.path_msg.poses[-1].pose

    def get_pose_yaw_safe(self, pose: Pose) -> Optional[float]:
        q = pose.orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm < 1e-9:
            return None

        return self.quaternion_to_yaw(
            q.x / norm,
            q.y / norm,
            q.z / norm,
            q.w / norm,
        )

    def get_goal_yaw(self) -> Optional[float]:
        if not self.path_msg.poses:
            return None

        goal_pose = self.path_msg.poses[-1].pose
        goal_yaw = self.get_pose_yaw_safe(goal_pose)
        if goal_yaw is not None:
            return goal_yaw

        if len(self.path_msg.poses) >= 2:
            p0 = self.path_msg.poses[-2].pose.position
            p1 = self.path_msg.poses[-1].pose.position
            if self.distance_xy(p0.x, p0.y, p1.x, p1.y) > 1e-6:
                return math.atan2(p1.y - p0.y, p1.x - p0.x)

        return None

    # ================================================================
    # Slow down around robot
    # ================================================================
    def update_closest_obstacle_distance(self) -> None:
        self.closest_distance_m = float("inf")

        if self.pose is None or self.map_msg is None:
            return

        robot_cell = PathPlanner.world_to_grid(self.map_msg, self.pose.position)
        if not PathPlanner.is_cell_in_bounds(self.map_msg, robot_cell):
            return

        res = self.map_msg.info.resolution
        inner_r = float(max(self.slow_down_ignore_radius_cells, 0))
        outer_r = float(max(self.slow_down_check_radius_cells, 0))

        for dx in range(-int(outer_r), int(outer_r) + 1):
            for dy in range(-int(outer_r), int(outer_r) + 1):
                distance_cells = math.hypot(dx, dy)

                if distance_cells > outer_r:
                    continue

                # bỏ vùng quá gần tâm xe
                if distance_cells <= inner_r:
                    continue

                gx = robot_cell[0] + dx
                gy = robot_cell[1] + dy
                cell = (gx, gy)

                if not PathPlanner.is_cell_in_bounds(self.map_msg, cell):
                    continue

                if PathPlanner.is_cell_walkable(self.map_msg, cell):
                    continue

                distance_m = distance_cells * res

                if distance_m < self.closest_distance_m:
                    self.closest_distance_m = distance_m

    # ================================================================
    # Forward obstacle check using front FOV only
    # ================================================================
    # sửa hàm maybe_request_replan
    def maybe_request_replan(self, path_index: int) -> None:
        now_ns = self.get_clock().now().nanoseconds
        cooldown_ns = int(self.replan_request_cooldown_sec * 1e9)

        if path_index < 0:
            return

        if self.last_replan_request_time_ns is not None:
            if now_ns - self.last_replan_request_time_ns < cooldown_ns:
                return

        self.replan_request_pub.publish(Int32(data=int(path_index)))
        self.last_replan_request_time_ns = now_ns
        self.get_logger().warn(
            f"Forward obstacle detected. Replan requested from path index {path_index}."
        )

    def is_obstacle_in_forward_fov(self) -> bool:
        if self.pose is None or self.pose_yaw is None or self.map_msg is None:
            return False

        robot_cell = PathPlanner.world_to_grid(self.map_msg, self.pose.position)
        if not PathPlanner.is_cell_in_bounds(self.map_msg, robot_cell):
            return False

        fov_rad = math.radians(self.forward_fov_deg)
        ignore_r = float(max(self.forward_ignore_radius_cells, 0))
        outer_r = float(max(self.forward_fov_distance_cells, 0))

        fov_cells: List[Tuple[int, int]] = []
        wall_cells: List[Tuple[int, int]] = []

        obstacle_found = False

        for dx in range(-int(outer_r), int(outer_r) + 1):
            for dy in range(-int(outer_r), int(outer_r) + 1):
                distance_cells = math.hypot(dx, dy)

                if distance_cells > outer_r:
                    continue

                if distance_cells <= ignore_r:
                    continue

                gx = robot_cell[0] + dx
                gy = robot_cell[1] + dy
                cell = (gx, gy)

                if not PathPlanner.is_cell_in_bounds(self.map_msg, cell):
                    continue

                angle = math.atan2(dy, dx) - self.pose_yaw
                angle = self.normalize_angle(angle)

                # chỉ quét phía trước
                if abs(angle) > (fov_rad / 2.0):
                    continue

                if self.is_in_debug_mode:
                    fov_cells.append(cell)

                if not PathPlanner.is_cell_walkable(self.map_msg, cell):
                    obstacle_found = True
                    if self.is_in_debug_mode:
                        wall_cells.append(cell)

        if self.is_in_debug_mode:
            self.fov_cells_pub.publish(
                PathPlanner.get_grid_cells(self.map_msg, fov_cells)
            )
            self.close_wall_cells_pub.publish(
                PathPlanner.get_grid_cells(self.map_msg, wall_cells)
            )

        return obstacle_found

    # ================================================================
    # Output helpers
    # ================================================================
    def send_speed(self, linear_speed: float, angular_speed: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = linear_speed
        msg.twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(msg)

    def stop(self) -> None:
        self.send_speed(0.0, 0.0)

    def publish_lookahead(self, point: Point) -> None:
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point = point
        self.lookahead_pub.publish(msg)

    # ================================================================
    # Main control loop
    # ================================================================
    def control_loop(self) -> None:
        self.update_pose_from_tf()

        if self.pose is None or self.pose_yaw is None:
            return

        if self.is_teleop_free_active():
            return

        if self.goal_reached_and_released:
            return

        if not self.path_msg.poses:
            self.stop()
            return


        goal_pose = self.get_goal_pose()
        if goal_pose is None:
            self.stop()
            return

        goal = goal_pose.position
        goal_yaw = self.get_goal_yaw()

        nearest_waypoint_index = self.find_nearest_waypoint_index()
        lookahead = self.find_lookahead(nearest_waypoint_index, self.lookahead_distance)
        if lookahead is None:
            self.stop()
            return

        self.publish_lookahead(lookahead)

        x = self.pose.position.x
        y = self.pose.position.y
        yaw = self.pose_yaw

        dx = lookahead.x - x
        dy = lookahead.y - y
        self.alpha = self.normalize_angle(math.atan2(dy, dx) - yaw)

        self.reversed = False

        lookahead_distance = self.distance_xy(x, y, lookahead.x, lookahead.y)
        distance_to_goal = self.distance_xy(x, y, goal.x, goal.y)

        # ============================================================
        # 1) Đã tới gần goal -> không chạy nữa, chỉ căn góc cuối
        # ============================================================
        if distance_to_goal < self.distance_tolerance:
            if goal_yaw is None:
                self.release_controller_for_current_goal(goal_pose)
                return

            final_yaw_error = self.normalize_angle(goal_yaw - yaw)

            if abs(final_yaw_error) <= self.final_yaw_tolerance_rad:
                self.release_controller_for_current_goal(goal_pose)
                return

            turn_speed = self.final_align_turn_kp * final_yaw_error
            turn_speed = self.clamp(turn_speed, -self.max_turn_speed, self.max_turn_speed)
            self.send_speed(0.0, -turn_speed)
            return

        # ============================================================
        # 2) Lệch hướng nhiều -> quay tại chỗ trước rồi mới chạy
        # ============================================================
        if abs(self.alpha) > self.heading_align_threshold_rad:
            turn_speed = self.heading_align_turn_kp * self.alpha
            turn_speed = self.clamp(turn_speed, -self.max_turn_speed, self.max_turn_speed)
            self.send_speed(0.0, -turn_speed)
            return

        if lookahead_distance < 1e-6:
            self.stop()
            return

        # ============================================================
        # 3) Update khoảng cách gần vật quanh xe để slow down
        # ============================================================
        self.update_closest_obstacle_distance()

        # ============================================================
        # 4) Quét FOV phía trước. Có vật cản thì replan
        # ============================================================
        if self.is_obstacle_in_forward_fov():
            self.maybe_request_replan(nearest_waypoint_index)

        # ============================================================
        # 5) Chạy pure pursuit bình thường
        # ============================================================
        drive_speed = self.max_drive_speed

        curvature = 2.0 * math.sin(self.alpha) / max(lookahead_distance, 1e-6)
        turn_speed = self.turn_speed_kp * drive_speed * curvature
        turn_speed = self.clamp(turn_speed, -self.max_turn_speed, self.max_turn_speed)

        # Slow down if close to obstacle around robot
        if self.closest_distance_m < self.obstacle_avoidance_max_slow_down_distance:
            d0 = self.obstacle_avoidance_min_slow_down_distance
            d1 = self.obstacle_avoidance_max_slow_down_distance
            f0 = self.obstacle_avoidance_min_slow_down_factor
            f1 = 1.0

            if self.closest_distance_m <= d0:
                slow_down_factor = f0
            elif self.closest_distance_m >= d1:
                slow_down_factor = f1
            else:
                ratio = (self.closest_distance_m - d0) / max(d1 - d0, 1e-6)
                slow_down_factor = f0 + ratio * (f1 - f0)

            drive_speed *= slow_down_factor

        self.send_speed(drive_speed, -turn_speed)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()