#!/usr/bin/env python3

from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener, TransformException

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

# Giữ nguyên các hàm lõi trong PathPlanner, không viết đè lại
from .path_planner import PathPlanner


GridCell = Tuple[int, int]


class GoalToPathNode(Node):
    def __init__(self) -> None:
        super().__init__("goal_to_path_node")

        # ---------- Parameters ----------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("replan_topic", "/replan")
        self.declare_parameter("path_topic", "/path")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")

        self.declare_parameter("cspace_padding", 15)
        self.declare_parameter("path_corridor_half_width_cells", 2)
        self.declare_parameter("occupancy_threshold", 50)
        self.declare_parameter("treat_unknown_as_obstacle", True)

        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self.replan_topic = self.get_parameter("replan_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value

        self.cspace_padding = self.get_parameter("cspace_padding").get_parameter_value().integer_value
        self.path_corridor_half_width_cells = (
            self.get_parameter("path_corridor_half_width_cells")
            .get_parameter_value()
            .integer_value
        )
        self.occupancy_threshold = (
            self.get_parameter("occupancy_threshold")
            .get_parameter_value()
            .integer_value
        )
        self.treat_unknown_as_obstacle = (
            self.get_parameter("treat_unknown_as_obstacle")
            .get_parameter_value()
            .bool_value
        )

        # ---------- Internal state ----------
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_cspace: Optional[OccupancyGrid] = None
        self.latest_cost_map = None

        self.map_dirty = False

        self.latest_goal_msg: Optional[PoseStamped] = None
        self.latest_path_msg: Optional[Path] = None
        self.latest_path_cells: Optional[List[GridCell]] = None

        # ---------- TF ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- QoS ----------
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        default_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------- ROS interfaces ----------
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos,
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self.goal_callback,
            default_qos,
        )

        self.replan_sub = self.create_subscription(
            Int32,
            self.replan_topic,
            self.replan_callback,
            default_qos,
        )

        self.path_pub = self.create_publisher(
            Path,
            self.path_topic,
            default_qos,
        )

        self.get_logger().info("GoalToPathNode started.")
        self.get_logger().info(f" map_topic                     : {self.map_topic}")
        self.get_logger().info(f" goal_topic                    : {self.goal_topic}")
        self.get_logger().info(f" replan_topic                  : {self.replan_topic}")
        self.get_logger().info(f" path_topic                    : {self.path_topic}")
        self.get_logger().info(f" map_frame                     : {self.map_frame}")
        self.get_logger().info(f" base_frame                    : {self.base_frame}")
        self.get_logger().info(f" cspace_padding                : {self.cspace_padding}")
        self.get_logger().info(
            f" path_corridor_half_width_cells: {self.path_corridor_half_width_cells}"
        )
        self.get_logger().info(f" occupancy_threshold           : {self.occupancy_threshold}")
        self.get_logger().info(
            f" treat_unknown_as_obstacle     : {self.treat_unknown_as_obstacle}"
        )

    # -------------------------------------------------------------------------
    # Map handling
    # -------------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg
        self.map_dirty = True

    def ensure_preprocessed_map(self, force: bool = False) -> bool:
        if self.latest_map is None:
            self.get_logger().warn("Map not ready yet.")
            return False

        need_recompute = (
            force
            or self.map_dirty
            or self.latest_cspace is None
            or self.latest_cost_map is None
        )

        if not need_recompute:
            return True

        try:
            cspace, _ = PathPlanner.calc_cspace(
                mapdata=self.latest_map,
                include_cells=False,
                padding=int(self.cspace_padding),
            )
            cost_map = PathPlanner.calc_cost_map(cspace)

            self.latest_cspace = cspace
            self.latest_cost_map = cost_map
            self.map_dirty = False
            return True
        except Exception as exc:
            self.latest_cspace = None
            self.latest_cost_map = None
            self.get_logger().error(f"Failed to preprocess map: {exc}")
            return False

    # -------------------------------------------------------------------------
    # Goal handling
    # -------------------------------------------------------------------------
    def goal_callback(self, goal_msg: PoseStamped) -> None:
        self.latest_goal_msg = goal_msg
        self.plan_to_goal(goal_msg, reason="new_goal", force_map_recompute=False)

    # -------------------------------------------------------------------------
    # Replan handling
    # -------------------------------------------------------------------------
    def replan_callback(self, msg: Int32) -> None:
        requested_index = int(msg.data)

        if requested_index < 0:
            self.get_logger().warn(f"Ignoring replan index < 0: {requested_index}")
            return

        if self.latest_goal_msg is None:
            self.get_logger().warn("Replan requested but no previous goal exists.")
            return

        if not self.ensure_preprocessed_map(force=True):
            self.get_logger().warn("Replan aborted because map preprocessing failed.")
            return

        if not self.latest_path_cells:
            self.get_logger().warn(
                "Replan requested but no cached path_cells exist. Planning directly to latest goal."
            )
            self.plan_to_goal(
                self.latest_goal_msg,
                reason="replan_no_cached_path",
                force_map_recompute=False,
            )
            return

        start_idx = min(requested_index, len(self.latest_path_cells) - 1)

        blocked, hit_cell, hit_index = self.is_path_blocked_from_index(
            self.latest_path_cells,
            start_idx,
        )

        if blocked:
            self.get_logger().warn(
                f"Path blocked from requested index={start_idx}, "
                f"hit_index={hit_index}, hit_cell={hit_cell}. Replanning."
            )
            self.plan_to_goal(
                self.latest_goal_msg,
                reason="replan_path_blocked",
                force_map_recompute=False,
            )
        else:
            self.get_logger().info(
                f"Replan requested at index={start_idx}, but path corridor is still clear."
            )

    # -------------------------------------------------------------------------
    # Planning core
    # -------------------------------------------------------------------------
    def plan_to_goal(
        self,
        goal_msg: PoseStamped,
        reason: str = "unknown",
        force_map_recompute: bool = False,
    ) -> None:
        if goal_msg.header.frame_id and goal_msg.header.frame_id != self.map_frame:
            self.get_logger().warn(
                f"Goal frame is '{goal_msg.header.frame_id}', expected '{self.map_frame}'. "
                "Please publish goal in map frame or add pose transform logic."
            )
            return

        if not self.ensure_preprocessed_map(force=force_map_recompute):
            self.get_logger().warn("Map / cspace / cost_map not ready yet.")
            return

        start_world = self.lookup_current_robot_position()
        if start_world is None or self.latest_cspace is None or self.latest_cost_map is None:
            return

        goal_world = Point(
            x=goal_msg.pose.position.x,
            y=goal_msg.pose.position.y,
            z=0.0,
        )

        start_cell = PathPlanner.world_to_grid(self.latest_cspace, start_world)
        goal_cell = PathPlanner.world_to_grid(self.latest_cspace, goal_world)

        self.get_logger().info(
            f"[{reason}] Planning from start_cell={start_cell} to goal_cell={goal_cell}"
        )

        try:
            path_cells, distance_cost, snapped_start, snapped_goal = PathPlanner.a_star(
                mapdata=self.latest_cspace,
                cost_map=self.latest_cost_map,
                start=start_cell,
                goal=goal_cell,
            )
        except Exception as exc:
            self.get_logger().error(f"[{reason}] A* planning failed: {exc}")
            return

        if not path_cells:
            self.get_logger().warn(
                f"[{reason}] No valid path. "
                f"snapped_start={snapped_start}, snapped_goal={snapped_goal}"
            )
            return

        path_msg = PathPlanner.path_to_message(
            mapdata=self.latest_cspace,
            path=path_cells,
            frame_id=self.map_frame,
        )
        path_msg.header.stamp = self.get_clock().now().to_msg()

        if path_msg.poses:
            path_msg.poses[-1].pose.orientation = goal_msg.pose.orientation

        self.path_pub.publish(path_msg)

        # Cache cả hai: path publish và path cell gốc từ A*
        self.latest_goal_msg = goal_msg
        self.latest_path_cells = list(path_cells)
        self.latest_path_msg = path_msg

        self.get_logger().info(
            f"[{reason}] Published path with {len(path_cells)} poses to {self.path_topic}, "
            f"distance_cost={distance_cost:.3f}, "
            f"snapped_start={snapped_start}, snapped_goal={snapped_goal}"
        )

    # -------------------------------------------------------------------------
    # TF
    # -------------------------------------------------------------------------
    def lookup_current_robot_position(self) -> Optional[Point]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.3),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"Could not get TF {self.map_frame} -> {self.base_frame}: {exc}"
            )
            return None

        return Point(
            x=transform.transform.translation.x,
            y=transform.transform.translation.y,
            z=0.0,
        )

    # -------------------------------------------------------------------------
    # Path blockage check
    # -------------------------------------------------------------------------
    def is_path_blocked_from_index(
        self,
        path_cells: List[GridCell],
        start_idx: int,
    ) -> Tuple[bool, Optional[GridCell], Optional[int]]:
        if self.latest_cspace is None:
            return False, None, None

        if not path_cells:
            return False, None, None

        start_idx = max(0, min(start_idx, len(path_cells) - 1))
        radius = max(int(self.path_corridor_half_width_cells), 0)

        for i in range(start_idx, len(path_cells)):
            cx, cy = path_cells[i]

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue

                    nx = cx + dx
                    ny = cy + dy

                    if not PathPlanner.is_cell_in_bounds(self.latest_cspace, (nx, ny)):
                        continue

                    if self.is_cell_blocked(self.latest_cspace, nx, ny):
                        return True, (nx, ny), i

        return False, None, None

    # -------------------------------------------------------------------------
    # Grid utility
    # -------------------------------------------------------------------------
    def is_cell_blocked(self, mapdata: OccupancyGrid, x: int, y: int) -> bool:
        idx = int(y) * int(mapdata.info.width) + int(x)
        value = int(mapdata.data[idx])

        if value < 0:
            return self.treat_unknown_as_obstacle

        return value >= int(self.occupancy_threshold)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalToPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()