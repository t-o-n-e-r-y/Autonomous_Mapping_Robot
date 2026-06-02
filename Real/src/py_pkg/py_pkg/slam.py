#!/usr/bin/env python3
import math
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, TransformStamped
from std_msgs.msg import Bool

import tf2_ros
from tf2_ros import TransformException

from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares

UNKNOWN = -1


# =============================================================================
# 2D pose utilities
# =============================================================================

@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)


@dataclass
class OptimizationStats:
    success: bool
    initial_rmse: float
    final_rmse: float
    improvement_ratio: float
    max_translation_jump_m: float
    max_rotation_jump_deg: float


@dataclass
class StaticMapSnapshot:
    log_odds: np.ndarray
    observed: np.ndarray
    occupied_mask: np.ndarray
    distance_field_m: np.ndarray
    stamp: object
    map_to_odom: Pose2D


@dataclass
class StaticLocalizeResult:
    base_pose_in_map: Pose2D
    map_to_odom: Pose2D
    score: float
    used_guess: bool


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


def pose_from_transform_msg(tf_msg: TransformStamped) -> Pose2D:
    return Pose2D(
        x=float(tf_msg.transform.translation.x),
        y=float(tf_msg.transform.translation.y),
        yaw=float(quaternion_to_yaw(tf_msg.transform.rotation)),
    )


def compose_pose(a: Pose2D, b: Pose2D) -> Pose2D:
    """
    T = A * B
    """
    ca = math.cos(a.yaw)
    sa = math.sin(a.yaw)
    x = a.x + ca * b.x - sa * b.y
    y = a.y + sa * b.x + ca * b.y
    yaw = normalize_angle(a.yaw + b.yaw)
    return Pose2D(x, y, yaw)


def inverse_pose(p: Pose2D) -> Pose2D:
    c = math.cos(p.yaw)
    s = math.sin(p.yaw)
    x = -c * p.x - s * p.y
    y = s * p.x - c * p.y
    yaw = normalize_angle(-p.yaw)
    return Pose2D(x, y, yaw)


def relative_pose(a: Pose2D, b: Pose2D) -> Pose2D:
    """
    Pose of B in frame A.
    inv(A) * B
    """
    return compose_pose(inverse_pose(a), b)


# =============================================================================
# Grid helpers
# =============================================================================

def is_no_information_ray(scan: LaserScan, index: int, r: float) -> bool:
    if not math.isinf(r):
        return False
    if not hasattr(scan, "intensities"):
        return False
    if index >= len(scan.intensities):
        return False
    try:
        intensity = float(scan.intensities[index])
    except (TypeError, ValueError):
        return False
    return intensity == 0.0


def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    x, y = x0, y0

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    cells.append((x1, y1))
    return cells


# =============================================================================
# Submap
# =============================================================================

class Submap:
    """
    Internal local map. Not a ROS TF frame.
    It has its own local coordinates and an odom anchor at creation time.
    """

    def __init__(
        self,
        submap_id: int,
        resolution: float,
        width: int,
        height: int,
        origin_x: float,
        origin_y: float,
        odom_anchor_pose: Pose2D,
    ) -> None:
        self.id = int(submap_id)
        self.frame_id = f"submap_{self.id:05d}"

        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

        self.odom_anchor_pose = odom_anchor_pose.copy()

        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.observed = np.zeros((self.height, self.width), dtype=np.bool_)

        self.scan_count = 0
        self.is_finished = False

        self.distance_field_m: Optional[np.ndarray] = None
        self.distance_field_dirty = True
        self.distance_field_update_counter = 0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(math.floor((x - self.origin_x) / self.resolution))
        iy = int(math.floor((y - self.origin_y) / self.resolution))
        return ix, iy

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height

    def add_log_odds(self, ix: int, iy: int, delta: float, l_min: float, l_max: float) -> None:
        if not self.in_bounds(ix, iy):
            return
        self.log_odds[iy, ix] = np.clip(self.log_odds[iy, ix] + delta, l_min, l_max)
        self.observed[iy, ix] = True

    def known_cells(self) -> int:
        return int(np.count_nonzero(self.observed))

    def to_occupancy_grid_msg(self, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        probs = 1.0 / (1.0 + np.exp(-self.log_odds))

        out = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)
        known = self.observed
        out[known] = np.clip(np.round(probs[known] * 100.0), 0, 100).astype(np.int8)

        msg.data = out.reshape(-1).tolist()
        return msg


# =============================================================================
# Internal data objects between front-end and back-end
# =============================================================================

@dataclass
class LocalState:
    stamp: object
    active_submap_id: int
    submap_to_base: Pose2D
    match_score: float
    tracking_ok: bool


@dataclass
class SubmapEdgeData:
    parent_id: int
    child_id: int
    relative_pose: Pose2D
    edge_type: str = "odom"


@dataclass
class SubmapRolloverEvent:
    stamp: object
    finished_submap: Submap
    new_active_submap: Submap
    edge: SubmapEdgeData


@dataclass
class GraphNodeData:
    submap_id: int
    frame_id: str
    global_pose: Pose2D
    fixed: bool = False
    archived: bool = False


@dataclass
class GraphEdgeData:
    parent_id: int
    child_id: int
    relative_pose: Pose2D
    edge_type: str = "odom"
    trans_weight: float = 10.0
    rot_weight: float = 12.0


# =============================================================================
# Front-end core
# =============================================================================

class FrontendCore:
    def __init__(
        self,
        submap_resolution: float,
        submap_width: int,
        submap_height: int,
        submap_origin_x: float,
        submap_origin_y: float,
        min_scans_per_submap: int,
        max_scans_per_submap: int,
        submap_translation_thresh_m: float,
        submap_rotation_thresh_deg: float,
        enable_local_matching: bool,
        use_previous_submap_for_matching: bool,
        min_submap_scans_before_matching: int,
        min_known_cells_before_matching: int,
        scan_match_stride: int,
        match_linear_window_m: float,
        match_angular_window_deg: float,
        match_coarse_linear_step_m: float,
        match_coarse_angular_step_deg: float,
        match_fine_linear_step_m: float,
        match_fine_angular_step_deg: float,
        match_accept_min_score_delta: float,
        max_match_translation_correction_m: float,
        max_match_rotation_correction_deg: float,
        max_usable_range: float,
        l_occ: float,
        l_free: float,
        l_min: float,
        l_max: float,
        occupied_stop_threshold: float,
        distance_field_occ_threshold: float,
        distance_field_sigma_m: float,
        distance_field_max_dist_m: float,
        distance_field_recompute_every_n_scans: int,
        min_tracking_score: float,
        min_fallback_score: float,
        max_tracking_translation_error_m: float,
        max_tracking_rotation_error_deg: float,
        max_fallback_translation_error_m: float,
        max_fallback_rotation_error_deg: float,
        tracking_bad_scan_tolerance: int,
        force_new_submap_on_tracking_loss: bool,
        use_odom_fallback_on_bad_match: bool,
        tracking_guard_warmup_scans: int,
        tracking_guard_warmup_known_cells: int,
    ) -> None:
        self.submap_resolution = float(submap_resolution)
        self.submap_width = int(submap_width)
        self.submap_height = int(submap_height)
        self.submap_origin_x = float(submap_origin_x)
        self.submap_origin_y = float(submap_origin_y)

        self.min_scans_per_submap = int(min_scans_per_submap)
        self.max_scans_per_submap = int(max_scans_per_submap)
        self.submap_translation_thresh_m = float(submap_translation_thresh_m)
        self.submap_rotation_thresh_deg = float(submap_rotation_thresh_deg)

        self.enable_local_matching = bool(enable_local_matching)
        self.use_previous_submap_for_matching = bool(use_previous_submap_for_matching)
        self.min_submap_scans_before_matching = int(min_submap_scans_before_matching)
        self.min_known_cells_before_matching = int(min_known_cells_before_matching)
        self.scan_match_stride = int(scan_match_stride)
        self.match_linear_window_m = float(match_linear_window_m)
        self.match_angular_window_deg = float(match_angular_window_deg)
        self.match_coarse_linear_step_m = float(match_coarse_linear_step_m)
        self.match_coarse_angular_step_deg = float(match_coarse_angular_step_deg)
        self.match_fine_linear_step_m = float(match_fine_linear_step_m)
        self.match_fine_angular_step_deg = float(match_fine_angular_step_deg)
        self.match_accept_min_score_delta = float(match_accept_min_score_delta)
        self.max_match_translation_correction_m = float(max_match_translation_correction_m)
        self.max_match_rotation_correction_deg = float(max_match_rotation_correction_deg)

        self.max_usable_range = float(max_usable_range)
        self.l_occ = float(l_occ)
        self.l_free = float(l_free)
        self.l_min = float(l_min)
        self.l_max = float(l_max)
        self.occupied_stop_threshold = float(occupied_stop_threshold)

        self.next_submap_id = 0
        self.active_submap: Optional[Submap] = None
        self.previous_submap: Optional[Submap] = None

        self.active_submap_dirty = False
        self.last_active_stamp = None

        self.distance_field_occ_threshold = float(distance_field_occ_threshold)
        self.distance_field_sigma_m = float(distance_field_sigma_m)
        self.distance_field_max_dist_m = float(distance_field_max_dist_m)
        self.distance_field_recompute_every_n_scans = int(distance_field_recompute_every_n_scans)

        self.min_tracking_score = float(min_tracking_score)
        self.min_fallback_score = float(min_fallback_score)
        self.max_tracking_translation_error_m = float(max_tracking_translation_error_m)
        self.max_tracking_rotation_error_deg = float(max_tracking_rotation_error_deg)
        self.max_fallback_translation_error_m = float(max_fallback_translation_error_m)
        self.max_fallback_rotation_error_deg = float(max_fallback_rotation_error_deg)
        self.tracking_bad_scan_tolerance = int(tracking_bad_scan_tolerance)
        self.force_new_submap_on_tracking_loss = bool(force_new_submap_on_tracking_loss)
        self.use_odom_fallback_on_bad_match = bool(use_odom_fallback_on_bad_match)
        self.tracking_guard_warmup_scans = int(tracking_guard_warmup_scans)
        self.tracking_guard_warmup_known_cells = int(tracking_guard_warmup_known_cells)

        self.consecutive_bad_tracking_count = 0

    def update_distance_field_if_needed(self, submap: Submap) -> None:
        if not submap.distance_field_dirty:
            return

        submap.distance_field_update_counter += 1
        if submap.distance_field_update_counter < self.distance_field_recompute_every_n_scans:
            return

        submap.distance_field_update_counter = 0
        occupied_mask = submap.observed & (submap.log_odds >= self.distance_field_occ_threshold)

        if not np.any(occupied_mask):
            submap.distance_field_m = np.full(
                (submap.height, submap.width),
                self.distance_field_max_dist_m,
                dtype=np.float32,
            )
            submap.distance_field_dirty = False
            return

        dist_cells = distance_transform_edt(~occupied_mask)
        dist_m = dist_cells.astype(np.float32) * submap.resolution
        dist_m = np.minimum(dist_m, self.distance_field_max_dist_m)

        submap.distance_field_m = dist_m
        submap.distance_field_dirty = False

    def create_new_submap(self, odom_anchor_pose: Pose2D) -> Submap:
        submap = Submap(
            submap_id=self.next_submap_id,
            resolution=self.submap_resolution,
            width=self.submap_width,
            height=self.submap_height,
            origin_x=self.submap_origin_x,
            origin_y=self.submap_origin_y,
            odom_anchor_pose=odom_anchor_pose,
        )
        self.next_submap_id += 1
        return submap

    def odom_pose_to_submap_pose(self, submap: Submap, odom_base_pose: Pose2D) -> Pose2D:
        return relative_pose(submap.odom_anchor_pose, odom_base_pose)

    def convert_pose_between_submaps(
        self,
        pose_in_src: Pose2D,
        src_submap: Submap,
        dst_submap: Submap,
    ) -> Pose2D:
        odom_base = compose_pose(src_submap.odom_anchor_pose, pose_in_src)
        return relative_pose(dst_submap.odom_anchor_pose, odom_base)

    def should_rollover_submap(self, submap: Submap, current_odom_base: Pose2D) -> bool:
        if submap.scan_count < self.min_scans_per_submap:
            return False

        delta = relative_pose(submap.odom_anchor_pose, current_odom_base)
        translation = math.hypot(delta.x, delta.y)
        rotation_deg = abs(math.degrees(delta.yaw))

        if submap.scan_count >= self.max_scans_per_submap:
            return True
        if translation >= self.submap_translation_thresh_m:
            return True
        if rotation_deg >= self.submap_rotation_thresh_deg:
            return True
        return False

    def compute_score(
        self,
        submap: Submap,
        scan: LaserScan,
        base_pose_in_submap: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> float:
        self.update_distance_field_if_needed(submap)

        if submap.distance_field_m is None:
            return -float("inf")

        sensor_pose = compose_pose(base_pose_in_submap, base_to_scan_pose)

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        hit_mask = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges < scan.range_max)
            & (ranges <= self.max_usable_range)
        )

        hit_indices = np.nonzero(hit_mask)[0]
        if len(hit_indices) == 0:
            return -float("inf")

        hit_indices = hit_indices[::self.scan_match_stride]
        if len(hit_indices) == 0:
            return -float("inf")

        hit_ranges = ranges[hit_indices]
        hit_angles = scan.angle_min + hit_indices * scan.angle_increment

        c = math.cos(sensor_pose.yaw)
        s = math.sin(sensor_pose.yaw)

        lx = hit_ranges * np.cos(hit_angles)
        ly = hit_ranges * np.sin(hit_angles)

        wx = sensor_pose.x + c * lx - s * ly
        wy = sensor_pose.y + s * lx + c * ly

        ix = np.floor((wx - submap.origin_x) / submap.resolution).astype(np.int32)
        iy = np.floor((wy - submap.origin_y) / submap.resolution).astype(np.int32)

        in_bounds = (ix >= 0) & (ix < submap.width) & (iy >= 0) & (iy < submap.height)
        if not np.any(in_bounds):
            return -float("inf")

        ix = ix[in_bounds]
        iy = iy[in_bounds]

        if len(ix) < 10:
            return -float("inf")

        d = submap.distance_field_m[iy, ix]
        sigma = max(self.distance_field_sigma_m, 1e-6)

        likelihood = np.exp(-0.5 * (d / sigma) ** 2)
        return float(np.mean(likelihood))

    def grid_search_match(
        self,
        submap: Submap,
        scan: LaserScan,
        guess_pose: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> Tuple[Pose2D, float]:
        guess_score = self.compute_score(submap, scan, guess_pose, base_to_scan_pose)

        if (
            not self.enable_local_matching
            or submap.scan_count < self.min_submap_scans_before_matching
            or submap.known_cells() < self.min_known_cells_before_matching
        ):
            return guess_pose.copy(), guess_score

        def search_around(
            center: Pose2D,
            linear_window: float,
            angular_window_deg: float,
            linear_step: float,
            angular_step_deg: float,
        ) -> Tuple[Pose2D, float]:
            best_pose = center.copy()
            best_score = self.compute_score(submap, scan, best_pose, base_to_scan_pose)

            dx_values = np.arange(-linear_window, linear_window + 1e-9, linear_step)
            dy_values = np.arange(-linear_window, linear_window + 1e-9, linear_step)
            dyaw_values = np.radians(
                np.arange(-angular_window_deg, angular_window_deg + 1e-9, angular_step_deg)
            )

            for dx in dx_values:
                for dy in dy_values:
                    for dyaw in dyaw_values:
                        candidate = Pose2D(
                            center.x + float(dx),
                            center.y + float(dy),
                            normalize_angle(center.yaw + float(dyaw)),
                        )

                        correction = relative_pose(guess_pose, candidate)
                        corr_trans = math.hypot(correction.x, correction.y)
                        corr_rot_deg = abs(math.degrees(correction.yaw))

                        if corr_trans > self.max_match_translation_correction_m:
                            continue
                        if corr_rot_deg > self.max_match_rotation_correction_deg:
                            continue

                        score = self.compute_score(submap, scan, candidate, base_to_scan_pose)
                        if score > best_score:
                            best_score = score
                            best_pose = candidate

            return best_pose, best_score

        coarse_pose, coarse_score = search_around(
            guess_pose,
            self.match_linear_window_m,
            self.match_angular_window_deg,
            self.match_coarse_linear_step_m,
            self.match_coarse_angular_step_deg,
        )

        fine_pose, fine_score = search_around(
            coarse_pose,
            self.match_coarse_linear_step_m,
            self.match_coarse_angular_step_deg,
            self.match_fine_linear_step_m,
            self.match_fine_angular_step_deg,
        )

        if fine_score < guess_score + self.match_accept_min_score_delta:
            return guess_pose.copy(), guess_score

        return fine_pose, fine_score

    def should_apply_tracking_guard(self) -> bool:
        if self.active_submap is None:
            return False
        if self.active_submap.scan_count < self.tracking_guard_warmup_scans:
            return False
        if self.active_submap.known_cells() < self.tracking_guard_warmup_known_cells:
            return False
        return True

    def evaluate_tracking_quality(
        self,
        odom_guess: Pose2D,
        candidate_pose: Pose2D,
        score: float,
    ) -> Tuple[bool, float, float]:
        correction = relative_pose(odom_guess, candidate_pose)
        corr_trans = math.hypot(correction.x, correction.y)
        corr_rot_deg = abs(math.degrees(correction.yaw))

        tracking_ok = (
            math.isfinite(score)
            and score >= self.min_tracking_score
            and corr_trans <= self.max_tracking_translation_error_m
            and corr_rot_deg <= self.max_tracking_rotation_error_deg
        )
        return tracking_ok, corr_trans, corr_rot_deg

    def fallback_is_allowed(
        self,
        score: float,
        corr_trans: float,
        corr_rot_deg: float,
    ) -> bool:
        return (
            self.use_odom_fallback_on_bad_match
            and math.isfinite(score)
            and score >= self.min_fallback_score
            and corr_trans <= self.max_fallback_translation_error_m
            and corr_rot_deg <= self.max_fallback_rotation_error_deg
        )

    def build_rollover_event(
        self,
        stamp,
        current_odom_base: Pose2D,
    ) -> Optional[SubmapRolloverEvent]:
        if self.active_submap is None:
            return None

        finished = self.active_submap
        finished.is_finished = True

        self.previous_submap = finished
        self.active_submap = self.create_new_submap(current_odom_base)

        edge = SubmapEdgeData(
            parent_id=finished.id,
            child_id=self.active_submap.id,
            relative_pose=relative_pose(
                finished.odom_anchor_pose,
                self.active_submap.odom_anchor_pose,
            ),
            edge_type="odom",
        )

        return SubmapRolloverEvent(
            stamp=stamp,
            finished_submap=finished,
            new_active_submap=self.active_submap,
            edge=edge,
        )

    def estimate_active_pose(
        self,
        scan: LaserScan,
        odom_base_pose: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> Tuple[Pose2D, float, bool]:
        assert self.active_submap is not None

        active_guess = self.odom_pose_to_submap_pose(self.active_submap, odom_base_pose)
        best_pose = active_guess.copy()
        best_score = -float("inf")

        active_pose_candidate, active_score = self.grid_search_match(
            self.active_submap,
            scan,
            active_guess,
            base_to_scan_pose,
        )
        best_pose = active_pose_candidate
        best_score = active_score

        if self.previous_submap is not None and self.use_previous_submap_for_matching:
            prev_guess = self.odom_pose_to_submap_pose(self.previous_submap, odom_base_pose)

            prev_pose_candidate, prev_score = self.grid_search_match(
                self.previous_submap,
                scan,
                prev_guess,
                base_to_scan_pose,
            )

            if prev_score > best_score:
                converted = self.convert_pose_between_submaps(
                    prev_pose_candidate,
                    self.previous_submap,
                    self.active_submap,
                )
                best_pose = converted
                best_score = prev_score

        tracking_ok, _, _ = self.evaluate_tracking_quality(
            active_guess,
            best_pose,
            best_score,
        )
        return best_pose, float(best_score), tracking_ok

    def integrate_scan_into_submap(
        self,
        submap: Submap,
        scan: LaserScan,
        base_pose_in_submap: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> bool:
        sensor_pose = compose_pose(base_pose_in_submap, base_to_scan_pose)

        sensor_ix, sensor_iy = submap.world_to_grid(sensor_pose.x, sensor_pose.y)
        if not submap.in_bounds(sensor_ix, sensor_iy):
            return False

        angle = scan.angle_min

        for i, r in enumerate(scan.ranges):
            if math.isnan(r):
                angle += scan.angle_increment
                continue

            if is_no_information_ray(scan, i, r):
                angle += scan.angle_increment
                continue

            if math.isinf(r) or r >= scan.range_max:
                has_hit = False
                usable_r = min(scan.range_max, self.max_usable_range)
            elif r < scan.range_min:
                angle += scan.angle_increment
                continue
            elif r > self.max_usable_range:
                has_hit = False
                usable_r = self.max_usable_range
            else:
                has_hit = True
                usable_r = r

            lx = usable_r * math.cos(angle)
            ly = usable_r * math.sin(angle)

            c = math.cos(sensor_pose.yaw)
            s = math.sin(sensor_pose.yaw)

            wx = sensor_pose.x + c * lx - s * ly
            wy = sensor_pose.y + s * lx + c * ly

            end_ix, end_iy = submap.world_to_grid(wx, wy)
            cells = bresenham(sensor_ix, sensor_iy, end_ix, end_iy)

            if len(cells) == 0:
                angle += scan.angle_increment
                continue

            if has_hit:
                cx, cy = cells[-1]
                submap.add_log_odds(cx, cy, self.l_occ, self.l_min, self.l_max)
                free_cells = cells[:-2]
            else:
                free_cells = cells[:-1]

            for cx, cy in free_cells:
                if not submap.in_bounds(cx, cy):
                    break
                if submap.log_odds[cy, cx] > self.occupied_stop_threshold:
                    break
                submap.add_log_odds(cx, cy, self.l_free, self.l_min, self.l_max)

            angle += scan.angle_increment

        submap.scan_count += 1
        submap.distance_field_dirty = True
        return True

    def process_scan(
        self,
        scan: LaserScan,
        odom_base_pose: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> Tuple[Optional[LocalState], Optional[SubmapRolloverEvent], bool]:
        created_first_submap = False

        if self.active_submap is None:
            self.active_submap = self.create_new_submap(odom_base_pose)
            created_first_submap = True

        odom_guess = self.odom_pose_to_submap_pose(self.active_submap, odom_base_pose)
        matched_pose, match_score, tracking_ok = self.estimate_active_pose(
            scan,
            odom_base_pose,
            base_to_scan_pose,
        )

        _, corr_trans, corr_rot_deg = self.evaluate_tracking_quality(
            odom_guess,
            matched_pose,
            match_score,
        )

        chosen_pose = matched_pose
        used_odom_fallback = False
        dropped_scan = False
        apply_guard = self.should_apply_tracking_guard()

        if not apply_guard:
            chosen_pose = odom_guess
            tracking_ok = True
            match_score = max(match_score, 0.0) if math.isfinite(match_score) else 0.0
            self.consecutive_bad_tracking_count = 0
        elif tracking_ok:
            self.consecutive_bad_tracking_count = 0
        else:
            self.consecutive_bad_tracking_count += 1

            if self.fallback_is_allowed(match_score, corr_trans, corr_rot_deg):
                chosen_pose = odom_guess
                used_odom_fallback = True
            else:
                dropped_scan = True

        if dropped_scan:
            rollover_event = None
            if (
                self.force_new_submap_on_tracking_loss
                and self.consecutive_bad_tracking_count >= self.tracking_bad_scan_tolerance
                and self.active_submap is not None
                and self.active_submap.scan_count >= self.min_scans_per_submap
            ):
                rollover_event = self.build_rollover_event(scan.header.stamp, odom_base_pose)
                self.consecutive_bad_tracking_count = 0

            return None, rollover_event, created_first_submap

        integrated = self.integrate_scan_into_submap(
            self.active_submap,
            scan,
            chosen_pose,
            base_to_scan_pose,
        )
        if not integrated:
            return None, None, created_first_submap

        self.last_active_stamp = scan.header.stamp
        self.active_submap_dirty = True

        local_state = LocalState(
            stamp=scan.header.stamp,
            active_submap_id=self.active_submap.id,
            submap_to_base=chosen_pose,
            match_score=match_score,
            tracking_ok=tracking_ok and not used_odom_fallback,
        )

        rollover_event: Optional[SubmapRolloverEvent] = None

        should_force_rollover = (
            self.force_new_submap_on_tracking_loss
            and self.consecutive_bad_tracking_count >= self.tracking_bad_scan_tolerance
            and self.active_submap is not None
            and self.active_submap.scan_count >= self.min_scans_per_submap
        )

        if should_force_rollover or self.should_rollover_submap(self.active_submap, odom_base_pose):
            rollover_event = self.build_rollover_event(scan.header.stamp, odom_base_pose)
            self.consecutive_bad_tracking_count = 0

        return local_state, rollover_event, created_first_submap


# =============================================================================
# Back-end core
# =============================================================================

class BackendCore:
    def __init__(
        self,
        map_resolution: float,
        map_width: int,
        map_height: int,
        map_origin_x: float,
        map_origin_y: float,
        l_min: float,
        l_max: float,
        distance_field_occ_threshold: float,
        distance_field_sigma_m: float,
        distance_field_max_dist_m: float,
        loop_search_radius_m: float,
        loop_min_submap_separation: int,
        loop_match_occ_threshold: float,
        loop_sample_step_cells: int,
        loop_linear_window_m: float,
        loop_angular_window_deg: float,
        loop_coarse_linear_step_m: float,
        loop_coarse_angular_step_deg: float,
        loop_fine_linear_step_m: float,
        loop_fine_angular_step_deg: float,
        loop_score_accept: float,
        loop_score_margin: float,
        loop_min_coverage_ratio: float,
        loop_max_translation_delta_from_guess_m: float,
        loop_max_rotation_delta_from_guess_deg: float,
        optimize_on_every_rollover: bool,
        optimize_max_nfev: int,
        optimization_min_improvement_ratio: float,
        optimization_max_allowed_translation_jump_m: float,
        optimization_max_allowed_rotation_jump_deg: float,
        enable_pruning: bool,
        keep_recent_submaps_full_res: int,
    ) -> None:
        self.map_resolution = float(map_resolution)
        self.map_width = int(map_width)
        self.map_height = int(map_height)
        self.map_origin_x = float(map_origin_x)
        self.map_origin_y = float(map_origin_y)
        self.l_min = float(l_min)
        self.l_max = float(l_max)

        self.distance_field_occ_threshold = float(distance_field_occ_threshold)
        self.distance_field_sigma_m = float(distance_field_sigma_m)
        self.distance_field_max_dist_m = float(distance_field_max_dist_m)

        self.loop_search_radius_m = float(loop_search_radius_m)
        self.loop_min_submap_separation = int(loop_min_submap_separation)
        self.loop_match_occ_threshold = float(loop_match_occ_threshold)
        self.loop_sample_step_cells = int(loop_sample_step_cells)
        self.loop_linear_window_m = float(loop_linear_window_m)
        self.loop_angular_window_deg = float(loop_angular_window_deg)
        self.loop_coarse_linear_step_m = float(loop_coarse_linear_step_m)
        self.loop_coarse_angular_step_deg = float(loop_coarse_angular_step_deg)
        self.loop_fine_linear_step_m = float(loop_fine_linear_step_m)
        self.loop_fine_angular_step_deg = float(loop_fine_angular_step_deg)
        self.loop_score_accept = float(loop_score_accept)
        self.loop_score_margin = float(loop_score_margin)
        self.loop_min_coverage_ratio = float(loop_min_coverage_ratio)
        self.loop_max_translation_delta_from_guess_m = float(
            loop_max_translation_delta_from_guess_m
        )
        self.loop_max_rotation_delta_from_guess_deg = float(
            loop_max_rotation_delta_from_guess_deg
        )

        self.optimize_on_every_rollover = bool(optimize_on_every_rollover)
        self.optimize_max_nfev = int(optimize_max_nfev)
        self.optimization_min_improvement_ratio = float(optimization_min_improvement_ratio)
        self.optimization_max_allowed_translation_jump_m = float(
            optimization_max_allowed_translation_jump_m
        )
        self.optimization_max_allowed_rotation_jump_deg = float(
            optimization_max_allowed_rotation_jump_deg
        )

        self.enable_pruning = bool(enable_pruning)
        self.keep_recent_submaps_full_res = int(keep_recent_submaps_full_res)

        self.submaps: Dict[int, Submap] = {}
        self.graph_nodes: Dict[int, GraphNodeData] = {}
        self.graph_edges: List[GraphEdgeData] = []

        self.current_active_submap_id: Optional[int] = None
        self.latest_local_state: Optional[LocalState] = None
        self.latest_map_to_odom: Optional[Pose2D] = None
        self.last_optimization_stats: Optional[OptimizationStats] = None

        self.map_dirty = False
        self.last_map_stamp = None

        self.frozen_global_log_odds = np.zeros(
            (self.map_height, self.map_width), dtype=np.float32
        )
        self.frozen_global_observed = np.zeros(
            (self.map_height, self.map_width), dtype=np.bool_
        )

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def has_submap(self, submap_id: int) -> bool:
        return submap_id in self.graph_nodes

    def has_mutable_graph(self) -> bool:
        if self.optimize_on_every_rollover:
            return True
        return any(e.edge_type == "loop" for e in self.graph_edges)

    def bootstrap_first_submap(self, submap: Submap, stamp) -> None:
        if submap.id in self.graph_nodes:
            return

        self.submaps[submap.id] = submap
        self.graph_nodes[submap.id] = GraphNodeData(
            submap_id=submap.id,
            frame_id=submap.frame_id,
            global_pose=Pose2D(0.0, 0.0, 0.0),
            fixed=True,
            archived=False,
        )
        self.current_active_submap_id = submap.id
        self.map_dirty = True
        self.last_map_stamp = stamp

    def handle_rollover_event(self, event: SubmapRolloverEvent) -> None:
        finished = event.finished_submap
        new_active = event.new_active_submap
        edge = event.edge

        if finished.id not in self.graph_nodes:
            self.submaps[finished.id] = finished
            self.graph_nodes[finished.id] = GraphNodeData(
                submap_id=finished.id,
                frame_id=finished.frame_id,
                global_pose=Pose2D(0.0, 0.0, 0.0),
                fixed=False,
                archived=False,
            )

        finished_global = self.graph_nodes[finished.id].global_pose
        new_global = compose_pose(finished_global, edge.relative_pose)

        self.submaps[finished.id] = finished
        self.submaps[new_active.id] = new_active

        self.graph_nodes[new_active.id] = GraphNodeData(
            submap_id=new_active.id,
            frame_id=new_active.frame_id,
            global_pose=new_global,
            fixed=False,
            archived=False,
        )

        self.graph_edges.append(
            GraphEdgeData(
                parent_id=edge.parent_id,
                child_id=edge.child_id,
                relative_pose=edge.relative_pose,
                edge_type=edge.edge_type,
                trans_weight=10.0,
                rot_weight=12.0,
            )
        )

        self.current_active_submap_id = new_active.id
        self.map_dirty = True
        self.last_map_stamp = event.stamp

    def update_local_state(self, local_state: LocalState, odom_base_pose: Pose2D, stamp) -> None:
        self.latest_local_state = local_state

        active_id = local_state.active_submap_id
        if active_id not in self.graph_nodes:
            return

        map_to_submap = self.graph_nodes[active_id].global_pose
        submap_to_base = local_state.submap_to_base

        map_to_base = compose_pose(map_to_submap, submap_to_base)
        map_to_odom = compose_pose(map_to_base, inverse_pose(odom_base_pose))

        self.latest_map_to_odom = map_to_odom
        self.map_dirty = True
        self.last_map_stamp = stamp

    def get_latest_map_to_odom(self) -> Optional[Pose2D]:
        return self.latest_map_to_odom.copy() if self.latest_map_to_odom is not None else None

    # ------------------------------------------------------------------
    # Distance field for finished submaps
    # ------------------------------------------------------------------
    def _ensure_distance_field(self, submap: Submap) -> None:
        if submap.distance_field_m is not None and not submap.distance_field_dirty:
            return

        occupied_mask = submap.observed & (submap.log_odds >= self.distance_field_occ_threshold)

        if not np.any(occupied_mask):
            submap.distance_field_m = np.full(
                (submap.height, submap.width),
                self.distance_field_max_dist_m,
                dtype=np.float32,
            )
            submap.distance_field_dirty = False
            return

        dist_cells = distance_transform_edt(~occupied_mask)
        dist_m = dist_cells.astype(np.float32) * submap.resolution
        dist_m = np.minimum(dist_m, self.distance_field_max_dist_m)

        submap.distance_field_m = dist_m
        submap.distance_field_dirty = False

    def _occupied_points_xy(self, submap: Submap) -> Optional[np.ndarray]:
        mask = submap.observed & (submap.log_odds >= self.loop_match_occ_threshold)
        iy, ix = np.nonzero(mask)
        if len(ix) == 0:
            return None

        step = max(1, self.loop_sample_step_cells)
        ix = ix[::step]
        iy = iy[::step]

        sx = submap.origin_x + (ix.astype(np.float32) + 0.5) * submap.resolution
        sy = submap.origin_y + (iy.astype(np.float32) + 0.5) * submap.resolution
        pts = np.stack([sx, sy], axis=1)

        if len(pts) > 1500:
            stride = int(math.ceil(len(pts) / 1500.0))
            pts = pts[::stride]

        return pts

    def _score_relative_pose(
        self,
        target_submap: Submap,
        target_from_source: Pose2D,
        source_points_xy: np.ndarray,
    ) -> Tuple[float, float]:
        self._ensure_distance_field(target_submap)
        if target_submap.distance_field_m is None:
            return -float("inf"), 0.0

        c = math.cos(target_from_source.yaw)
        s = math.sin(target_from_source.yaw)

        tx = target_from_source.x + c * source_points_xy[:, 0] - s * source_points_xy[:, 1]
        ty = target_from_source.y + s * source_points_xy[:, 0] + c * source_points_xy[:, 1]

        ix = np.floor((tx - target_submap.origin_x) / target_submap.resolution).astype(np.int32)
        iy = np.floor((ty - target_submap.origin_y) / target_submap.resolution).astype(np.int32)

        in_bounds = (
            (ix >= 0) & (ix < target_submap.width) &
            (iy >= 0) & (iy < target_submap.height)
        )
        if not np.any(in_bounds):
            return -float("inf"), 0.0

        ix = ix[in_bounds]
        iy = iy[in_bounds]
        coverage = float(len(ix)) / float(len(source_points_xy))
        if len(ix) < 20:
            return -float("inf"), coverage

        d = target_submap.distance_field_m[iy, ix]
        sigma = max(self.distance_field_sigma_m, 1e-6)
        likelihood = np.exp(-0.5 * (d / sigma) ** 2)
        score = float(np.mean(likelihood) * coverage)
        return score, coverage

    def _search_relative_pose(
        self,
        target_submap: Submap,
        source_points_xy: np.ndarray,
        guess_rel: Pose2D,
        linear_window: float,
        angular_window_deg: float,
        linear_step: float,
        angular_step_deg: float,
    ) -> Tuple[Pose2D, float, float]:
        best_pose = guess_rel.copy()
        best_score, best_coverage = self._score_relative_pose(target_submap, best_pose, source_points_xy)

        dx_values = np.arange(-linear_window, linear_window + 1e-9, linear_step)
        dy_values = np.arange(-linear_window, linear_window + 1e-9, linear_step)
        dyaw_values = np.radians(
            np.arange(-angular_window_deg, angular_window_deg + 1e-9, angular_step_deg)
        )

        for dx in dx_values:
            for dy in dy_values:
                for dyaw in dyaw_values:
                    cand = Pose2D(
                        guess_rel.x + float(dx),
                        guess_rel.y + float(dy),
                        normalize_angle(guess_rel.yaw + float(dyaw)),
                    )
                    score, coverage = self._score_relative_pose(target_submap, cand, source_points_xy)
                    if score > best_score:
                        best_score = score
                        best_pose = cand
                        best_coverage = coverage

        return best_pose, best_score, best_coverage

    def _register_submap_pair(
        self,
        source_id: int,
        target_id: int,
    ) -> Tuple[Optional[Pose2D], float, float, Pose2D, float]:
        if source_id not in self.submaps or target_id not in self.submaps:
            return None, -float("inf"), 0.0, Pose2D(), -float("inf")

        source_submap = self.submaps[source_id]
        target_submap = self.submaps[target_id]

        source_points = self._occupied_points_xy(source_submap)
        if source_points is None or len(source_points) < 20:
            return None, -float("inf"), 0.0, Pose2D(), -float("inf")

        target_pose_global = self.graph_nodes[target_id].global_pose
        source_pose_global = self.graph_nodes[source_id].global_pose
        guess_rel = relative_pose(target_pose_global, source_pose_global)

        guess_score, _ = self._score_relative_pose(target_submap, guess_rel, source_points)

        coarse_pose, coarse_score, _ = self._search_relative_pose(
            target_submap=target_submap,
            source_points_xy=source_points,
            guess_rel=guess_rel,
            linear_window=self.loop_linear_window_m,
            angular_window_deg=self.loop_angular_window_deg,
            linear_step=self.loop_coarse_linear_step_m,
            angular_step_deg=self.loop_coarse_angular_step_deg,
        )

        fine_pose, fine_score, fine_coverage = self._search_relative_pose(
            target_submap=target_submap,
            source_points_xy=source_points,
            guess_rel=coarse_pose,
            linear_window=self.loop_coarse_linear_step_m,
            angular_window_deg=self.loop_coarse_angular_step_deg,
            linear_step=self.loop_fine_linear_step_m,
            angular_step_deg=self.loop_fine_angular_step_deg,
        )

        return fine_pose, fine_score, fine_coverage, guess_rel, guess_score

    def _edge_exists(self, a: int, b: int, edge_type: str) -> bool:
        for e in self.graph_edges:
            if e.edge_type != edge_type:
                continue
            if e.parent_id == a and e.child_id == b:
                return True
            if e.parent_id == b and e.child_id == a:
                return True
        return False

    def try_add_loop_closure(self, source_id: int) -> bool:
        if source_id not in self.graph_nodes:
            return False
        if source_id not in self.submaps:
            return False

        source_node = self.graph_nodes[source_id]
        source_pose = source_node.global_pose

        best_target_id: Optional[int] = None
        best_rel: Optional[Pose2D] = None
        best_score = -float("inf")
        best_coverage = 0.0
        best_guess_rel: Optional[Pose2D] = None
        best_guess_score = -float("inf")

        candidate_ids: List[int] = []
        for target_id, node in self.graph_nodes.items():
            if target_id == source_id:
                continue
            if node.archived:
                continue
            if target_id == self.current_active_submap_id:
                continue
            if abs(target_id - source_id) < self.loop_min_submap_separation:
                continue
            if target_id not in self.submaps:
                continue

            dist = math.hypot(
                node.global_pose.x - source_pose.x,
                node.global_pose.y - source_pose.y,
            )
            if dist > self.loop_search_radius_m:
                continue

            candidate_ids.append(target_id)

        candidate_ids.sort(
            key=lambda sid: math.hypot(
                self.graph_nodes[sid].global_pose.x - source_pose.x,
                self.graph_nodes[sid].global_pose.y - source_pose.y,
            )
        )

        for target_id in candidate_ids:
            if self._edge_exists(target_id, source_id, "loop"):
                continue

            rel_pose, score, coverage, guess_rel, guess_score = self._register_submap_pair(
                source_id,
                target_id,
            )
            if rel_pose is None:
                continue

            delta_from_guess = relative_pose(guess_rel, rel_pose)
            delta_trans = math.hypot(delta_from_guess.x, delta_from_guess.y)
            delta_rot_deg = abs(math.degrees(delta_from_guess.yaw))

            if delta_trans > self.loop_max_translation_delta_from_guess_m:
                continue
            if delta_rot_deg > self.loop_max_rotation_delta_from_guess_deg:
                continue
            if coverage < self.loop_min_coverage_ratio:
                continue
            if score < self.loop_score_accept:
                continue
            if score < guess_score + self.loop_score_margin:
                continue

            if score > best_score:
                best_score = score
                best_target_id = target_id
                best_rel = rel_pose
                best_coverage = coverage
                best_guess_rel = guess_rel
                best_guess_score = guess_score

        if best_target_id is None or best_rel is None or best_guess_rel is None:
            return False

        self.graph_edges.append(
            GraphEdgeData(
                parent_id=best_target_id,
                child_id=source_id,
                relative_pose=best_rel,
                edge_type="loop",
                trans_weight=20.0,
                rot_weight=24.0,
            )
        )
        self.map_dirty = True
        return True

    # ------------------------------------------------------------------
    # Pose graph optimization
    # ------------------------------------------------------------------
    def _build_variable_pack(self) -> Tuple[List[int], Dict[int, int], np.ndarray]:
        variable_ids = [
            sid
            for sid, node in sorted(self.graph_nodes.items())
            if not node.fixed and not node.archived
        ]
        var_index = {sid: i for i, sid in enumerate(variable_ids)}

        x0 = np.zeros(len(variable_ids) * 3, dtype=np.float64)
        for sid, i in var_index.items():
            p = self.graph_nodes[sid].global_pose
            x0[3 * i + 0] = p.x
            x0[3 * i + 1] = p.y
            x0[3 * i + 2] = p.yaw
        return variable_ids, var_index, x0

    def _unpack_state(self, x: np.ndarray, var_index: Dict[int, int]) -> Dict[int, Pose2D]:
        poses: Dict[int, Pose2D] = {}
        for sid, node in self.graph_nodes.items():
            if sid in var_index:
                i = var_index[sid]
                poses[sid] = Pose2D(
                    x=float(x[3 * i + 0]),
                    y=float(x[3 * i + 1]),
                    yaw=normalize_angle(float(x[3 * i + 2])),
                )
            else:
                poses[sid] = node.global_pose.copy()
        return poses

    def _residual_vector(self, poses: Dict[int, Pose2D]) -> np.ndarray:
        res: List[float] = []
        for e in self.graph_edges:
            if e.parent_id not in poses or e.child_id not in poses:
                continue

            parent_pose = poses[e.parent_id]
            child_pose = poses[e.child_id]
            predicted = relative_pose(parent_pose, child_pose)
            err = relative_pose(e.relative_pose, predicted)

            res.append(e.trans_weight * err.x)
            res.append(e.trans_weight * err.y)
            res.append(e.rot_weight * normalize_angle(err.yaw))

        return np.asarray(res, dtype=np.float64)

    def optimize_graph(self) -> bool:
        has_loop_edge = any(e.edge_type == "loop" for e in self.graph_edges)
        if not has_loop_edge and not self.optimize_on_every_rollover:
            self.last_optimization_stats = None
            return False

        variable_ids, var_index, x0 = self._build_variable_pack()
        if len(variable_ids) == 0:
            self.last_optimization_stats = None
            return False

        original_poses = {sid: self.graph_nodes[sid].global_pose.copy() for sid in variable_ids}
        initial_poses = self._unpack_state(x0, var_index)
        initial_res = self._residual_vector(initial_poses)
        initial_rmse = float(np.sqrt(np.mean(initial_res ** 2))) if len(initial_res) > 0 else 0.0

        def residuals(x: np.ndarray) -> np.ndarray:
            poses = self._unpack_state(x, var_index)
            return self._residual_vector(poses)

        try:
            result = least_squares(
                residuals,
                x0,
                method="trf",
                max_nfev=self.optimize_max_nfev,
                verbose=0,
            )
        except Exception:
            self.last_optimization_stats = None
            return False

        if not result.success:
            self.last_optimization_stats = None
            return False

        optimized = self._unpack_state(result.x, var_index)
        final_res = self._residual_vector(optimized)
        final_rmse = float(np.sqrt(np.mean(final_res ** 2))) if len(final_res) > 0 else 0.0

        improvement_ratio = 0.0
        if initial_rmse > 1e-9:
            improvement_ratio = (initial_rmse - final_rmse) / initial_rmse

        max_translation_jump_m = 0.0
        max_rotation_jump_deg = 0.0
        for sid in variable_ids:
            before = original_poses[sid]
            after = optimized[sid]
            dx = after.x - before.x
            dy = after.y - before.y
            jump_t = math.hypot(dx, dy)
            jump_r = abs(math.degrees(normalize_angle(after.yaw - before.yaw)))
            max_translation_jump_m = max(max_translation_jump_m, jump_t)
            max_rotation_jump_deg = max(max_rotation_jump_deg, jump_r)

        stats = OptimizationStats(
            success=True,
            initial_rmse=initial_rmse,
            final_rmse=final_rmse,
            improvement_ratio=improvement_ratio,
            max_translation_jump_m=max_translation_jump_m,
            max_rotation_jump_deg=max_rotation_jump_deg,
        )
        self.last_optimization_stats = stats

        if final_rmse > initial_rmse + 1e-9:
            return False
        if initial_rmse > 1e-9 and improvement_ratio < self.optimization_min_improvement_ratio:
            return False
        if max_translation_jump_m > self.optimization_max_allowed_translation_jump_m:
            return False
        if max_rotation_jump_deg > self.optimization_max_allowed_rotation_jump_deg:
            return False

        for sid in variable_ids:
            self.graph_nodes[sid].global_pose = optimized[sid]

        self.map_dirty = True
        return True

    # ------------------------------------------------------------------
    # Prune / archive
    # ------------------------------------------------------------------
    def _fuse_submap_into_frozen_layer(self, submap_id: int) -> None:
        if submap_id not in self.submaps:
            return
        if submap_id not in self.graph_nodes:
            return

        submap = self.submaps[submap_id]
        node = self.graph_nodes[submap_id]

        iy, ix = np.nonzero(submap.observed)
        if len(ix) == 0:
            return

        sx = submap.origin_x + (ix.astype(np.float32) + 0.5) * submap.resolution
        sy = submap.origin_y + (iy.astype(np.float32) + 0.5) * submap.resolution

        c = math.cos(node.global_pose.yaw)
        s = math.sin(node.global_pose.yaw)

        mx = node.global_pose.x + c * sx - s * sy
        my = node.global_pose.y + s * sx + c * sy

        gx = np.floor((mx - self.map_origin_x) / self.map_resolution).astype(np.int32)
        gy = np.floor((my - self.map_origin_y) / self.map_resolution).astype(np.int32)

        in_bounds = (
            (gx >= 0) & (gx < self.map_width) &
            (gy >= 0) & (gy < self.map_height)
        )
        if not np.any(in_bounds):
            return

        gx = gx[in_bounds]
        gy = gy[in_bounds]
        vals = submap.log_odds[iy[in_bounds], ix[in_bounds]]

        np.add.at(self.frozen_global_log_odds, (gy, gx), vals)
        self.frozen_global_observed[gy, gx] = True
        self.frozen_global_log_odds = np.clip(
            self.frozen_global_log_odds,
            self.l_min,
            self.l_max,
        )

    def maybe_prune_old_submaps(self) -> int:
        if not self.enable_pruning:
            return 0

        # Critical safety guard:
        # Do not freeze/prune old submaps while graph remains mutable.
        # Otherwise future optimization will move live submaps but frozen layer stays in old poses.
        if self.has_mutable_graph():
            return 0

        live_finished_ids = [
            sid
            for sid in sorted(self.submaps.keys())
            if sid != self.current_active_submap_id
            and sid in self.graph_nodes
            and not self.graph_nodes[sid].archived
        ]

        pruned = 0
        while len(live_finished_ids) > self.keep_recent_submaps_full_res:
            sid = live_finished_ids.pop(0)
            if sid not in self.submaps:
                continue

            self._fuse_submap_into_frozen_layer(sid)
            del self.submaps[sid]
            self.graph_nodes[sid].archived = True
            self.graph_nodes[sid].fixed = True
            pruned += 1

        if pruned > 0:
            self.map_dirty = True

        return pruned

    # ------------------------------------------------------------------
    # Render /map
    # ------------------------------------------------------------------
    def render_global_map_msg(self, stamp) -> OccupancyGrid:
        global_log_odds = self.frozen_global_log_odds.copy()
        global_observed = self.frozen_global_observed.copy()

        for submap_id, node in self.graph_nodes.items():
            if node.archived:
                continue
            if submap_id not in self.submaps:
                continue

            submap = self.submaps[submap_id]
            if submap.known_cells() == 0:
                continue

            iy, ix = np.nonzero(submap.observed)
            if len(ix) == 0:
                continue

            sx = submap.origin_x + (ix.astype(np.float32) + 0.5) * submap.resolution
            sy = submap.origin_y + (iy.astype(np.float32) + 0.5) * submap.resolution

            c = math.cos(node.global_pose.yaw)
            s = math.sin(node.global_pose.yaw)

            mx = node.global_pose.x + c * sx - s * sy
            my = node.global_pose.y + s * sx + c * sy

            gx = np.floor((mx - self.map_origin_x) / self.map_resolution).astype(np.int32)
            gy = np.floor((my - self.map_origin_y) / self.map_resolution).astype(np.int32)

            in_bounds = (
                (gx >= 0) & (gx < self.map_width) &
                (gy >= 0) & (gy < self.map_height)
            )
            if not np.any(in_bounds):
                continue

            gx = gx[in_bounds]
            gy = gy[in_bounds]
            vals = submap.log_odds[iy[in_bounds], ix[in_bounds]]

            np.add.at(global_log_odds, (gy, gx), vals)
            global_observed[gy, gx] = True

        global_log_odds = np.clip(global_log_odds, self.l_min, self.l_max)
        probs = 1.0 / (1.0 + np.exp(-global_log_odds))

        out = np.full((self.map_height, self.map_width), UNKNOWN, dtype=np.int8)
        known = global_observed
        out[known] = np.clip(np.round(probs[known] * 100.0), 0, 100).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = out.reshape(-1).tolist()
        return msg


# =============================================================================
# Main ROS node
# =============================================================================

class SlamSubmapGraphNode(Node):
    def __init__(self) -> None:
        super().__init__("slam_submap_graph_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("mode_topic", "/mode")

        self.declare_parameter("process_rate_hz", 30.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.0)

        # Realtime robot: scan cũ quá thì bỏ, không giữ tận 3 giây
        self.declare_parameter("scan_queue_ttl_sec", 0.3)

        # Không cho queue phình lên 200 scan, vì như vậy là robot đang xử lý quá khứ
        self.declare_parameter("max_pending_scans", 5)

        # Mỗi lần timer chỉ xử lý tối đa vài scan, tránh block node lâu
        self.declare_parameter("max_scans_per_process_tick", 1)

        # Nếu scan chờ TF quá lâu thì bỏ
        self.declare_parameter("max_wait_tf_sec", 0.2)

        # In debug timing mỗi N scan xử lý / miss
        self.declare_parameter("debug_timing_every_n", 20)

        self.declare_parameter("submap_resolution", 0.05)
        self.declare_parameter("submap_width", 240)
        self.declare_parameter("submap_height", 240)
        self.declare_parameter("submap_origin_x", -6.0)
        self.declare_parameter("submap_origin_y", -6.0)

        self.declare_parameter("min_scans_per_submap", 20)
        self.declare_parameter("max_scans_per_submap", 80)
        self.declare_parameter("submap_translation_thresh_m", 0.5)
        self.declare_parameter("submap_rotation_thresh_deg", 15.0)

        self.declare_parameter("enable_local_matching", True)
        self.declare_parameter("use_previous_submap_for_matching", True)
        self.declare_parameter("min_submap_scans_before_matching", 5)
        self.declare_parameter("min_known_cells_before_matching", 50)
        self.declare_parameter("scan_match_stride", 4)
        self.declare_parameter("match_linear_window_m", 0.10)
        self.declare_parameter("match_angular_window_deg", 5.0)
        self.declare_parameter("match_coarse_linear_step_m", 0.05)
        self.declare_parameter("match_coarse_angular_step_deg", 2.0)
        self.declare_parameter("match_fine_linear_step_m", 0.02)
        self.declare_parameter("match_fine_angular_step_deg", 0.5)
        self.declare_parameter("match_accept_min_score_delta", 0.02)
        self.declare_parameter("max_match_translation_correction_m", 0.15)
        self.declare_parameter("max_match_rotation_correction_deg", 8.0)

        self.declare_parameter("distance_field_occ_threshold", 1.0)
        self.declare_parameter("distance_field_sigma_m", 0.15)
        self.declare_parameter("distance_field_max_dist_m", 1.0)
        self.declare_parameter("distance_field_recompute_every_n_scans", 2)
        self.declare_parameter("min_tracking_score", 0.42)
        self.declare_parameter("min_fallback_score", 0.30)
        self.declare_parameter("max_tracking_translation_error_m", 0.08)
        self.declare_parameter("max_tracking_rotation_error_deg", 3.5)
        self.declare_parameter("max_fallback_translation_error_m", 0.18)
        self.declare_parameter("max_fallback_rotation_error_deg", 8.0)
        self.declare_parameter("tracking_bad_scan_tolerance", 6)
        self.declare_parameter("force_new_submap_on_tracking_loss", True)
        self.declare_parameter("use_odom_fallback_on_bad_match", True)
        self.declare_parameter("tracking_guard_warmup_scans", 12)
        self.declare_parameter("tracking_guard_warmup_known_cells", 120)

        self.declare_parameter("dynamic_obstacle_ttl_sec", 2.5)
        self.declare_parameter("dynamic_obstacle_log_odds", 3.5)
        self.declare_parameter("static_mode_max_usable_range", 8.0)
        self.declare_parameter("static_localize_linear_window_m", 0.12)
        self.declare_parameter("static_localize_angular_window_deg", 6.0)
        self.declare_parameter("static_localize_coarse_linear_step_m", 0.04)
        self.declare_parameter("static_localize_coarse_angular_step_deg", 2.0)
        self.declare_parameter("static_localize_fine_linear_step_m", 0.015)
        self.declare_parameter("static_localize_fine_angular_step_deg", 0.5)
        self.declare_parameter("static_localize_min_score", 0.30)
        self.declare_parameter("static_localize_min_score_delta", 0.01)
        self.declare_parameter("static_localize_max_translation_correction_m", 0.20)
        self.declare_parameter("static_localize_max_rotation_correction_deg", 8.0)

        self.declare_parameter("max_usable_range", 8.0)
        self.declare_parameter("l_occ", 0.85)
        self.declare_parameter("l_free", -0.40)
        self.declare_parameter("l_min", -4.0)
        self.declare_parameter("l_max", 4.0)
        self.declare_parameter("occupied_stop_threshold", 1.5)

        self.declare_parameter("map_resolution", 0.05)
        self.declare_parameter("map_width", 600)
        self.declare_parameter("map_height", 600)
        self.declare_parameter("map_origin_x", -15.0)
        self.declare_parameter("map_origin_y", -15.0)

        self.declare_parameter("loop_search_radius_m", 2.5)
        self.declare_parameter("loop_min_submap_separation", 5)
        self.declare_parameter("loop_match_occ_threshold", 1.0)
        self.declare_parameter("loop_sample_step_cells", 3)
        self.declare_parameter("loop_linear_window_m", 0.75)
        self.declare_parameter("loop_angular_window_deg", 12.0)
        self.declare_parameter("loop_coarse_linear_step_m", 0.10)
        self.declare_parameter("loop_coarse_angular_step_deg", 2.0)
        self.declare_parameter("loop_fine_linear_step_m", 0.03)
        self.declare_parameter("loop_fine_angular_step_deg", 0.5)
        self.declare_parameter("loop_score_accept", 0.55)
        self.declare_parameter("loop_score_margin", 0.03)
        self.declare_parameter("loop_min_coverage_ratio", 0.35)
        self.declare_parameter("loop_max_translation_delta_from_guess_m", 0.60)
        self.declare_parameter("loop_max_rotation_delta_from_guess_deg", 10.0)

        self.declare_parameter("optimize_on_every_rollover", False)
        self.declare_parameter("optimize_max_nfev", 60)
        self.declare_parameter("optimization_min_improvement_ratio", 0.01)
        self.declare_parameter("optimization_max_allowed_translation_jump_m", 1.50)
        self.declare_parameter("optimization_max_allowed_rotation_jump_deg", 25.0)

        self.declare_parameter("enable_pruning", True)
        self.declare_parameter("keep_recent_submaps_full_res", 25)

        self.declare_parameter("map_publish_rate_hz", 0.1)
        self.declare_parameter("active_submap_publish_rate_hz", 0.2)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.map_topic = self.get_parameter("map_topic").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.mode_topic = self.get_parameter("mode_topic").value


        self.process_rate_hz = float(self.get_parameter("process_rate_hz").value)
        self.tf_lookup_timeout_sec = float(self.get_parameter("tf_lookup_timeout_sec").value)
        self.scan_queue_ttl_sec = float(self.get_parameter("scan_queue_ttl_sec").value)
        self.max_pending_scans = int(self.get_parameter("max_pending_scans").value)

        self.max_scans_per_process_tick = int(
            self.get_parameter("max_scans_per_process_tick").value
        )
        self.max_wait_tf_sec = float(
            self.get_parameter("max_wait_tf_sec").value
        )
        self.debug_timing_every_n = int(
            self.get_parameter("debug_timing_every_n").value
        )

        self.processed_scan_count = 0
        self.tf_miss_debug_count = 0

        submap_resolution = float(self.get_parameter("submap_resolution").value)
        submap_width = int(self.get_parameter("submap_width").value)
        submap_height = int(self.get_parameter("submap_height").value)
        submap_origin_x = float(self.get_parameter("submap_origin_x").value)
        submap_origin_y = float(self.get_parameter("submap_origin_y").value)

        min_scans_per_submap = int(self.get_parameter("min_scans_per_submap").value)
        max_scans_per_submap = int(self.get_parameter("max_scans_per_submap").value)
        submap_translation_thresh_m = float(self.get_parameter("submap_translation_thresh_m").value)
        submap_rotation_thresh_deg = float(self.get_parameter("submap_rotation_thresh_deg").value)

        enable_local_matching = bool(self.get_parameter("enable_local_matching").value)
        use_previous_submap_for_matching = bool(
            self.get_parameter("use_previous_submap_for_matching").value
        )
        min_submap_scans_before_matching = int(
            self.get_parameter("min_submap_scans_before_matching").value
        )
        min_known_cells_before_matching = int(
            self.get_parameter("min_known_cells_before_matching").value
        )
        scan_match_stride = int(self.get_parameter("scan_match_stride").value)
        match_linear_window_m = float(self.get_parameter("match_linear_window_m").value)
        match_angular_window_deg = float(self.get_parameter("match_angular_window_deg").value)
        match_coarse_linear_step_m = float(self.get_parameter("match_coarse_linear_step_m").value)
        match_coarse_angular_step_deg = float(self.get_parameter("match_coarse_angular_step_deg").value)
        match_fine_linear_step_m = float(self.get_parameter("match_fine_linear_step_m").value)
        match_fine_angular_step_deg = float(self.get_parameter("match_fine_angular_step_deg").value)
        match_accept_min_score_delta = float(self.get_parameter("match_accept_min_score_delta").value)
        max_match_translation_correction_m = float(
            self.get_parameter("max_match_translation_correction_m").value
        )
        max_match_rotation_correction_deg = float(
            self.get_parameter("max_match_rotation_correction_deg").value
        )
        distance_field_occ_threshold = float(self.get_parameter("distance_field_occ_threshold").value)
        distance_field_sigma_m = float(self.get_parameter("distance_field_sigma_m").value)
        distance_field_max_dist_m = float(self.get_parameter("distance_field_max_dist_m").value)
        distance_field_recompute_every_n_scans = int(
            self.get_parameter("distance_field_recompute_every_n_scans").value
        )
        min_tracking_score = float(self.get_parameter("min_tracking_score").value)
        min_fallback_score = float(self.get_parameter("min_fallback_score").value)
        max_tracking_translation_error_m = float(
            self.get_parameter("max_tracking_translation_error_m").value
        )
        max_tracking_rotation_error_deg = float(
            self.get_parameter("max_tracking_rotation_error_deg").value
        )
        max_fallback_translation_error_m = float(
            self.get_parameter("max_fallback_translation_error_m").value
        )
        max_fallback_rotation_error_deg = float(
            self.get_parameter("max_fallback_rotation_error_deg").value
        )
        tracking_bad_scan_tolerance = int(
            self.get_parameter("tracking_bad_scan_tolerance").value
        )
        force_new_submap_on_tracking_loss = bool(
            self.get_parameter("force_new_submap_on_tracking_loss").value
        )
        use_odom_fallback_on_bad_match = bool(
            self.get_parameter("use_odom_fallback_on_bad_match").value
        )
        tracking_guard_warmup_scans = int(
            self.get_parameter("tracking_guard_warmup_scans").value
        )
        tracking_guard_warmup_known_cells = int(
            self.get_parameter("tracking_guard_warmup_known_cells").value
        )

        self.dynamic_obstacle_ttl_sec = float(
            self.get_parameter("dynamic_obstacle_ttl_sec").value
        )
        self.dynamic_obstacle_log_odds = float(
            self.get_parameter("dynamic_obstacle_log_odds").value
        )
        self.static_mode_max_usable_range = float(
            self.get_parameter("static_mode_max_usable_range").value
        )
        self.static_localize_linear_window_m = float(
            self.get_parameter("static_localize_linear_window_m").value
        )
        self.static_localize_angular_window_deg = float(
            self.get_parameter("static_localize_angular_window_deg").value
        )
        self.static_localize_coarse_linear_step_m = float(
            self.get_parameter("static_localize_coarse_linear_step_m").value
        )
        self.static_localize_coarse_angular_step_deg = float(
            self.get_parameter("static_localize_coarse_angular_step_deg").value
        )
        self.static_localize_fine_linear_step_m = float(
            self.get_parameter("static_localize_fine_linear_step_m").value
        )
        self.static_localize_fine_angular_step_deg = float(
            self.get_parameter("static_localize_fine_angular_step_deg").value
        )
        self.static_localize_min_score = float(
            self.get_parameter("static_localize_min_score").value
        )
        self.static_localize_min_score_delta = float(
            self.get_parameter("static_localize_min_score_delta").value
        )
        self.static_localize_max_translation_correction_m = float(
            self.get_parameter("static_localize_max_translation_correction_m").value
        )
        self.static_localize_max_rotation_correction_deg = float(
            self.get_parameter("static_localize_max_rotation_correction_deg").value
        )

        max_usable_range = float(self.get_parameter("max_usable_range").value)
        l_occ = float(self.get_parameter("l_occ").value)
        l_free = float(self.get_parameter("l_free").value)
        l_min = float(self.get_parameter("l_min").value)
        l_max = float(self.get_parameter("l_max").value)
        occupied_stop_threshold = float(self.get_parameter("occupied_stop_threshold").value)

        map_resolution = float(self.get_parameter("map_resolution").value)
        map_width = int(self.get_parameter("map_width").value)
        map_height = int(self.get_parameter("map_height").value)
        map_origin_x = float(self.get_parameter("map_origin_x").value)
        map_origin_y = float(self.get_parameter("map_origin_y").value)

        self.map_publish_rate_hz = float(self.get_parameter("map_publish_rate_hz").value)
        self.active_submap_publish_rate_hz = float(
            self.get_parameter("active_submap_publish_rate_hz").value
        )
        loop_search_radius_m = float(self.get_parameter("loop_search_radius_m").value)
        loop_min_submap_separation = int(self.get_parameter("loop_min_submap_separation").value)
        loop_match_occ_threshold = float(self.get_parameter("loop_match_occ_threshold").value)
        loop_sample_step_cells = int(self.get_parameter("loop_sample_step_cells").value)
        loop_linear_window_m = float(self.get_parameter("loop_linear_window_m").value)
        loop_angular_window_deg = float(self.get_parameter("loop_angular_window_deg").value)
        loop_coarse_linear_step_m = float(self.get_parameter("loop_coarse_linear_step_m").value)
        loop_coarse_angular_step_deg = float(self.get_parameter("loop_coarse_angular_step_deg").value)
        loop_fine_linear_step_m = float(self.get_parameter("loop_fine_linear_step_m").value)
        loop_fine_angular_step_deg = float(self.get_parameter("loop_fine_angular_step_deg").value)
        loop_score_accept = float(self.get_parameter("loop_score_accept").value)
        loop_score_margin = float(self.get_parameter("loop_score_margin").value)
        loop_min_coverage_ratio = float(self.get_parameter("loop_min_coverage_ratio").value)
        loop_max_translation_delta_from_guess_m = float(
            self.get_parameter("loop_max_translation_delta_from_guess_m").value
        )
        loop_max_rotation_delta_from_guess_deg = float(
            self.get_parameter("loop_max_rotation_delta_from_guess_deg").value
        )

        optimize_on_every_rollover = bool(self.get_parameter("optimize_on_every_rollover").value)
        optimize_max_nfev = int(self.get_parameter("optimize_max_nfev").value)
        optimization_min_improvement_ratio = float(
            self.get_parameter("optimization_min_improvement_ratio").value
        )
        optimization_max_allowed_translation_jump_m = float(
            self.get_parameter("optimization_max_allowed_translation_jump_m").value
        )
        optimization_max_allowed_rotation_jump_deg = float(
            self.get_parameter("optimization_max_allowed_rotation_jump_deg").value
        )

        enable_pruning = bool(self.get_parameter("enable_pruning").value)
        keep_recent_submaps_full_res = int(
            self.get_parameter("keep_recent_submaps_full_res").value
        )

        self.frontend = FrontendCore(
            submap_resolution=submap_resolution,
            submap_width=submap_width,
            submap_height=submap_height,
            submap_origin_x=submap_origin_x,
            submap_origin_y=submap_origin_y,
            min_scans_per_submap=min_scans_per_submap,
            max_scans_per_submap=max_scans_per_submap,
            submap_translation_thresh_m=submap_translation_thresh_m,
            submap_rotation_thresh_deg=submap_rotation_thresh_deg,
            enable_local_matching=enable_local_matching,
            use_previous_submap_for_matching=use_previous_submap_for_matching,
            min_submap_scans_before_matching=min_submap_scans_before_matching,
            min_known_cells_before_matching=min_known_cells_before_matching,
            scan_match_stride=scan_match_stride,
            match_linear_window_m=match_linear_window_m,
            match_angular_window_deg=match_angular_window_deg,
            match_coarse_linear_step_m=match_coarse_linear_step_m,
            match_coarse_angular_step_deg=match_coarse_angular_step_deg,
            match_fine_linear_step_m=match_fine_linear_step_m,
            match_fine_angular_step_deg=match_fine_angular_step_deg,
            match_accept_min_score_delta=match_accept_min_score_delta,
            max_match_translation_correction_m=max_match_translation_correction_m,
            max_match_rotation_correction_deg=max_match_rotation_correction_deg,
            max_usable_range=max_usable_range,
            l_occ=l_occ,
            l_free=l_free,
            l_min=l_min,
            l_max=l_max,
            occupied_stop_threshold=occupied_stop_threshold,
            distance_field_occ_threshold=distance_field_occ_threshold,
            distance_field_sigma_m=distance_field_sigma_m,
            distance_field_max_dist_m=distance_field_max_dist_m,
            distance_field_recompute_every_n_scans=distance_field_recompute_every_n_scans,
            min_tracking_score=min_tracking_score,
            min_fallback_score=min_fallback_score,
            max_tracking_translation_error_m=max_tracking_translation_error_m,
            max_tracking_rotation_error_deg=max_tracking_rotation_error_deg,
            max_fallback_translation_error_m=max_fallback_translation_error_m,
            max_fallback_rotation_error_deg=max_fallback_rotation_error_deg,
            tracking_bad_scan_tolerance=tracking_bad_scan_tolerance,
            force_new_submap_on_tracking_loss=force_new_submap_on_tracking_loss,
            use_odom_fallback_on_bad_match=use_odom_fallback_on_bad_match,
            tracking_guard_warmup_scans=tracking_guard_warmup_scans,
            tracking_guard_warmup_known_cells=tracking_guard_warmup_known_cells,
        )

        self.backend = BackendCore(
            map_resolution=map_resolution,
            map_width=map_width,
            map_height=map_height,
            map_origin_x=map_origin_x,
            map_origin_y=map_origin_y,
            l_min=l_min,
            l_max=l_max,
            distance_field_occ_threshold=distance_field_occ_threshold,
            distance_field_sigma_m=distance_field_sigma_m,
            distance_field_max_dist_m=distance_field_max_dist_m,
            loop_search_radius_m=loop_search_radius_m,
            loop_min_submap_separation=loop_min_submap_separation,
            loop_match_occ_threshold=loop_match_occ_threshold,
            loop_sample_step_cells=loop_sample_step_cells,
            loop_linear_window_m=loop_linear_window_m,
            loop_angular_window_deg=loop_angular_window_deg,
            loop_coarse_linear_step_m=loop_coarse_linear_step_m,
            loop_coarse_angular_step_deg=loop_coarse_angular_step_deg,
            loop_fine_linear_step_m=loop_fine_linear_step_m,
            loop_fine_angular_step_deg=loop_fine_angular_step_deg,
            loop_score_accept=loop_score_accept,
            loop_score_margin=loop_score_margin,
            loop_min_coverage_ratio=loop_min_coverage_ratio,
            loop_max_translation_delta_from_guess_m=loop_max_translation_delta_from_guess_m,
            loop_max_rotation_delta_from_guess_deg=loop_max_rotation_delta_from_guess_deg,
            optimize_on_every_rollover=optimize_on_every_rollover,
            optimize_max_nfev=optimize_max_nfev,
            optimization_min_improvement_ratio=optimization_min_improvement_ratio,
            optimization_max_allowed_translation_jump_m=optimization_max_allowed_translation_jump_m,
            optimization_max_allowed_rotation_jump_deg=optimization_max_allowed_rotation_jump_deg,
            enable_pruning=enable_pruning,
            keep_recent_submaps_full_res=keep_recent_submaps_full_res,
        )

        # TF buffer giữ lịch sử lâu hơn một chút để lookup scan theo timestamp.
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=10.0)
        )

        # Cho TF listener tự spin thread riêng.
        # Như vậy callback /tf không bị process_scan / render_map trong node SLAM chặn.
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True
        )

        # Nếu bạn đã có node riêng broadcast TF từ /map_to_odom,
        # thì không cần broadcaster ở đây nữa.
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        transform_stamp_qos = QoSProfile(depth=1)
        transform_stamp_qos.reliability = ReliabilityPolicy.RELIABLE
        transform_stamp_qos.durability = DurabilityPolicy.VOLATILE

        self.map_to_odom_pub = self.create_publisher(
            TransformStamped,
            "/map_to_odom",
            transform_stamp_qos,
        )

        self.pending_scans: Deque[Tuple[int, LaserScan]] = deque()

        self.scan_count = 0
        self.tf_hit_count = 0
        self.tf_miss_count = 0
        self.drop_count = 0
        self.queue_expire_count = 0
        self.rollover_count = 0

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            scan_qos,
        )

        self.mode_sub = self.create_subscription(
            Bool,
            self.mode_topic,
            self.mode_callback,
            10,
        )

        self.mode_enabled = False
        self.static_snapshot: Optional[StaticMapSnapshot] = None
        self.static_dynamic_cells: Dict[Tuple[int, int], int] = {}
        self.static_last_map_to_odom: Optional[Pose2D] = None

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        debug_qos = QoSProfile(depth=3)
        debug_qos.reliability = ReliabilityPolicy.RELIABLE
        debug_qos.durability = DurabilityPolicy.VOLATILE

        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, map_qos)
        self.active_submap_pub = self.create_publisher(
            OccupancyGrid,
            "/slam_debug/active_submap",
            debug_qos,
        )

        self.process_timer = self.create_timer(
            1.0 / self.process_rate_hz,
            self.process_pending_scans,
        )

        self.map_timer = self.create_timer(
            1.0 / self.map_publish_rate_hz,
            self.publish_global_map_if_dirty,
        )

        self.active_submap_timer = self.create_timer(
            1.0 / self.active_submap_publish_rate_hz,
            self.publish_active_submap_if_dirty,
        )

        self.debug_timer = self.create_timer(3.0, self.debug_status)

        self.get_logger().info("RUNNING 1-NODE SUBMAP + POSE GRAPH SLAM")
        self.get_logger().info(
            f"submap={submap_width}x{submap_height}, res={submap_resolution}, "
            f"origin=({submap_origin_x}, {submap_origin_y})"
        )
        self.get_logger().info(
            f"global_map={map_width}x{map_height}, res={map_resolution}, "
            f"origin=({map_origin_x}, {map_origin_y})"
        )
        self.get_logger().info(
            f"tracking_guard: min_score={min_tracking_score:.2f}, "
            f"fallback_score={min_fallback_score:.2f}, "
            f"max_corr=({max_tracking_translation_error_m:.2f} m, "
            f"{max_tracking_rotation_error_deg:.1f} deg), "
            f"bad_scan_tolerance={tracking_bad_scan_tolerance}, "
            f"warmup=({tracking_guard_warmup_scans} scans, "
            f"{tracking_guard_warmup_known_cells} known cells)"
        )
        self.get_logger().info(
            "Safety guards enabled: stricter loop closure, optimize sanity checks, "
            "and pruning disabled while graph can still move."
        )
        self.get_logger().info(
            f"mode_topic={self.mode_topic}, dynamic_obstacle_ttl={self.dynamic_obstacle_ttl_sec:.2f}s"
        )
        self.get_logger().info(
            "Ray filter enabled: ranges=inf with intensity=0.0 are ignored completely."
        )


    def mode_callback(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested == self.mode_enabled:
            return

        if requested:
            self.enter_static_mode()
        else:
            self.exit_static_mode()

    def clear_slam_runtime(self) -> None:
        self.frontend = FrontendCore(
            submap_resolution=self.frontend.submap_resolution,
            submap_width=self.frontend.submap_width,
            submap_height=self.frontend.submap_height,
            submap_origin_x=self.frontend.submap_origin_x,
            submap_origin_y=self.frontend.submap_origin_y,
            min_scans_per_submap=self.frontend.min_scans_per_submap,
            max_scans_per_submap=self.frontend.max_scans_per_submap,
            submap_translation_thresh_m=self.frontend.submap_translation_thresh_m,
            submap_rotation_thresh_deg=self.frontend.submap_rotation_thresh_deg,
            enable_local_matching=self.frontend.enable_local_matching,
            use_previous_submap_for_matching=self.frontend.use_previous_submap_for_matching,
            min_submap_scans_before_matching=self.frontend.min_submap_scans_before_matching,
            min_known_cells_before_matching=self.frontend.min_known_cells_before_matching,
            scan_match_stride=self.frontend.scan_match_stride,
            match_linear_window_m=self.frontend.match_linear_window_m,
            match_angular_window_deg=self.frontend.match_angular_window_deg,
            match_coarse_linear_step_m=self.frontend.match_coarse_linear_step_m,
            match_coarse_angular_step_deg=self.frontend.match_coarse_angular_step_deg,
            match_fine_linear_step_m=self.frontend.match_fine_linear_step_m,
            match_fine_angular_step_deg=self.frontend.match_fine_angular_step_deg,
            match_accept_min_score_delta=self.frontend.match_accept_min_score_delta,
            max_match_translation_correction_m=self.frontend.max_match_translation_correction_m,
            max_match_rotation_correction_deg=self.frontend.max_match_rotation_correction_deg,
            max_usable_range=self.frontend.max_usable_range,
            l_occ=self.frontend.l_occ,
            l_free=self.frontend.l_free,
            l_min=self.frontend.l_min,
            l_max=self.frontend.l_max,
            occupied_stop_threshold=self.frontend.occupied_stop_threshold,
            distance_field_occ_threshold=self.frontend.distance_field_occ_threshold,
            distance_field_sigma_m=self.frontend.distance_field_sigma_m,
            distance_field_max_dist_m=self.frontend.distance_field_max_dist_m,
            distance_field_recompute_every_n_scans=self.frontend.distance_field_recompute_every_n_scans,
            min_tracking_score=self.frontend.min_tracking_score,
            min_fallback_score=self.frontend.min_fallback_score,
            max_tracking_translation_error_m=self.frontend.max_tracking_translation_error_m,
            max_tracking_rotation_error_deg=self.frontend.max_tracking_rotation_error_deg,
            max_fallback_translation_error_m=self.frontend.max_fallback_translation_error_m,
            max_fallback_rotation_error_deg=self.frontend.max_fallback_rotation_error_deg,
            tracking_bad_scan_tolerance=self.frontend.tracking_bad_scan_tolerance,
            force_new_submap_on_tracking_loss=self.frontend.force_new_submap_on_tracking_loss,
            use_odom_fallback_on_bad_match=self.frontend.use_odom_fallback_on_bad_match,
            tracking_guard_warmup_scans=self.frontend.tracking_guard_warmup_scans,
            tracking_guard_warmup_known_cells=self.frontend.tracking_guard_warmup_known_cells,
        )

        self.backend = BackendCore(
            map_resolution=self.backend.map_resolution,
            map_width=self.backend.map_width,
            map_height=self.backend.map_height,
            map_origin_x=self.backend.map_origin_x,
            map_origin_y=self.backend.map_origin_y,
            l_min=self.backend.l_min,
            l_max=self.backend.l_max,
            distance_field_occ_threshold=self.backend.distance_field_occ_threshold,
            distance_field_sigma_m=self.backend.distance_field_sigma_m,
            distance_field_max_dist_m=self.backend.distance_field_max_dist_m,
            loop_search_radius_m=self.backend.loop_search_radius_m,
            loop_min_submap_separation=self.backend.loop_min_submap_separation,
            loop_match_occ_threshold=self.backend.loop_match_occ_threshold,
            loop_sample_step_cells=self.backend.loop_sample_step_cells,
            loop_linear_window_m=self.backend.loop_linear_window_m,
            loop_angular_window_deg=self.backend.loop_angular_window_deg,
            loop_coarse_linear_step_m=self.backend.loop_coarse_linear_step_m,
            loop_coarse_angular_step_deg=self.backend.loop_coarse_angular_step_deg,
            loop_fine_linear_step_m=self.backend.loop_fine_linear_step_m,
            loop_fine_angular_step_deg=self.backend.loop_fine_angular_step_deg,
            loop_score_accept=self.backend.loop_score_accept,
            loop_score_margin=self.backend.loop_score_margin,
            loop_min_coverage_ratio=self.backend.loop_min_coverage_ratio,
            loop_max_translation_delta_from_guess_m=self.backend.loop_max_translation_delta_from_guess_m,
            loop_max_rotation_delta_from_guess_deg=self.backend.loop_max_rotation_delta_from_guess_deg,
            optimize_on_every_rollover=self.backend.optimize_on_every_rollover,
            optimize_max_nfev=self.backend.optimize_max_nfev,
            optimization_min_improvement_ratio=self.backend.optimization_min_improvement_ratio,
            optimization_max_allowed_translation_jump_m=self.backend.optimization_max_allowed_translation_jump_m,
            optimization_max_allowed_rotation_jump_deg=self.backend.optimization_max_allowed_rotation_jump_deg,
            enable_pruning=self.backend.enable_pruning,
            keep_recent_submaps_full_res=self.backend.keep_recent_submaps_full_res,
        )
        self.pending_scans.clear()

    def _static_world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.static_snapshot is None:
            return None

        ix = int(math.floor((x - self.backend.map_origin_x) / self.backend.map_resolution))
        iy = int(math.floor((y - self.backend.map_origin_y) / self.backend.map_resolution))
        if 0 <= ix < self.backend.map_width and 0 <= iy < self.backend.map_height:
            return ix, iy
        return None

    def enter_static_mode(self) -> None:
        stamp = self.backend.last_map_stamp
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        map_msg = self.backend.render_global_map_msg(stamp)
        grid = np.asarray(map_msg.data, dtype=np.int16).reshape(
            map_msg.info.height,
            map_msg.info.width,
        )

        observed = grid != UNKNOWN
        probs = np.zeros_like(grid, dtype=np.float32)
        probs[observed] = np.clip(grid[observed].astype(np.float32) / 100.0, 1e-3, 1.0 - 1e-3)

        log_odds = np.zeros_like(probs, dtype=np.float32)
        if np.any(observed):
            p = probs[observed]
            log_odds[observed] = np.log(p / (1.0 - p))

        occupied_mask = observed & (grid >= 50)

        if np.any(occupied_mask):
            dist_cells = distance_transform_edt(~occupied_mask)
            distance_field_m = dist_cells.astype(np.float32) * map_msg.info.resolution
            distance_field_m = np.minimum(distance_field_m, self.static_mode_max_usable_range)
        else:
            distance_field_m = np.full(
                (map_msg.info.height, map_msg.info.width),
                self.static_mode_max_usable_range,
                dtype=np.float32,
            )

        current_map_to_odom = self.backend.get_latest_map_to_odom()
        if current_map_to_odom is None:
            current_map_to_odom = Pose2D()

        self.static_snapshot = StaticMapSnapshot(
            log_odds=log_odds,
            observed=observed,
            occupied_mask=occupied_mask,
            distance_field_m=distance_field_m,
            stamp=stamp,
            map_to_odom=current_map_to_odom.copy(),
        )
        self.static_last_map_to_odom = current_map_to_odom.copy()
        self.static_dynamic_cells.clear()
        self.mode_enabled = True

        self.clear_slam_runtime()
        self.get_logger().info("Switched to static-map localization mode.")

    def exit_static_mode(self) -> None:
        self.mode_enabled = False
        self.static_snapshot = None
        self.static_dynamic_cells.clear()
        self.static_last_map_to_odom = None

        self.clear_slam_runtime()
        self.get_logger().info("Returned to normal SLAM mode.")


    def time_msg_to_sec(self, stamp_msg) -> float:
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9


    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


    def debug_scan_timing(
        self,
        scan: LaserScan,
        enqueue_time_ns: int,
        prefix: str,
    ) -> None:
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9

        scan_s = self.time_msg_to_sec(scan.header.stamp)
        enqueue_s = enqueue_time_ns * 1e-9

        scan_age_s = now_s - scan_s
        queue_age_s = now_s - enqueue_s

        self.get_logger().info(
            f"[{prefix}] "
            f"now={now_s:.6f}s, "
            f"scan_stamp={scan_s:.6f}s, "
            f"scan_age={scan_age_s:+.3f}s, "
            f"queue_age={queue_age_s:.3f}s, "
            f"pending={len(self.pending_scans)}, "
            f"frame_id={scan.header.frame_id}, "
            f"tf_hit={self.tf_hit_count}, "
            f"tf_miss={self.tf_miss_count}, "
            f"drop={self.drop_count}, "
            f"queue_expire={self.queue_expire_count}"
        )



    def _compute_static_score(
        self,
        scan: LaserScan,
        base_pose_in_map: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> float:
        if self.static_snapshot is None:
            return -float("inf")

        sensor_pose = compose_pose(base_pose_in_map, base_to_scan_pose)

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        hit_mask = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges < scan.range_max)
            & (ranges <= self.static_mode_max_usable_range)
        )

        hit_indices = np.nonzero(hit_mask)[0]
        if len(hit_indices) == 0:
            return -float("inf")

        hit_indices = hit_indices[::max(1, self.frontend.scan_match_stride)]
        if len(hit_indices) == 0:
            return -float("inf")

        hit_ranges = ranges[hit_indices]
        hit_angles = scan.angle_min + hit_indices * scan.angle_increment

        c = math.cos(sensor_pose.yaw)
        s = math.sin(sensor_pose.yaw)

        lx = hit_ranges * np.cos(hit_angles)
        ly = hit_ranges * np.sin(hit_angles)

        wx = sensor_pose.x + c * lx - s * ly
        wy = sensor_pose.y + s * lx + c * ly

        ix = np.floor((wx - self.backend.map_origin_x) / self.backend.map_resolution).astype(np.int32)
        iy = np.floor((wy - self.backend.map_origin_y) / self.backend.map_resolution).astype(np.int32)

        in_bounds = (
            (ix >= 0) & (ix < self.backend.map_width) &
            (iy >= 0) & (iy < self.backend.map_height)
        )
        if not np.any(in_bounds):
            return -float("inf")

        ix = ix[in_bounds]
        iy = iy[in_bounds]
        if len(ix) < 10:
            return -float("inf")

        d = self.static_snapshot.distance_field_m[iy, ix]
        sigma = max(self.frontend.distance_field_sigma_m, 1e-6)
        likelihood = np.exp(-0.5 * (d / sigma) ** 2)
        return float(np.mean(likelihood))

    def localize_scan_on_static_map(
        self,
        scan: LaserScan,
        odom_base_pose: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> Optional[StaticLocalizeResult]:
        if self.static_snapshot is None:
            return None

        guess_map_to_odom = self.static_last_map_to_odom
        if guess_map_to_odom is None:
            guess_map_to_odom = self.static_snapshot.map_to_odom.copy()

        guess_base_pose = compose_pose(guess_map_to_odom, odom_base_pose)
        guess_score = self._compute_static_score(scan, guess_base_pose, base_to_scan_pose)

        def search(center: Pose2D, linear_window: float, angular_window_deg: float,
                   linear_step: float, angular_step_deg: float) -> Tuple[Pose2D, float]:
            best_pose = center.copy()
            best_score = self._compute_static_score(scan, best_pose, base_to_scan_pose)

            dx_values = np.arange(-linear_window, linear_window + 1e-9, linear_step)
            dy_values = np.arange(-linear_window, linear_window + 1e-9, linear_step)
            dyaw_values = np.radians(
                np.arange(-angular_window_deg, angular_window_deg + 1e-9, angular_step_deg)
            )

            for dx in dx_values:
                for dy in dy_values:
                    for dyaw in dyaw_values:
                        cand = Pose2D(
                            center.x + float(dx),
                            center.y + float(dy),
                            normalize_angle(center.yaw + float(dyaw)),
                        )
                        corr = relative_pose(guess_base_pose, cand)
                        if math.hypot(corr.x, corr.y) > self.static_localize_max_translation_correction_m:
                            continue
                        if abs(math.degrees(corr.yaw)) > self.static_localize_max_rotation_correction_deg:
                            continue

                        score = self._compute_static_score(scan, cand, base_to_scan_pose)
                        if score > best_score:
                            best_score = score
                            best_pose = cand

            return best_pose, best_score

        coarse_pose, coarse_score = search(
            guess_base_pose,
            self.static_localize_linear_window_m,
            self.static_localize_angular_window_deg,
            self.static_localize_coarse_linear_step_m,
            self.static_localize_coarse_angular_step_deg,
        )
        fine_pose, fine_score = search(
            coarse_pose,
            self.static_localize_coarse_linear_step_m,
            self.static_localize_coarse_angular_step_deg,
            self.static_localize_fine_linear_step_m,
            self.static_localize_fine_angular_step_deg,
        )

        used_guess = False
        if (
            not math.isfinite(fine_score)
            or fine_score < self.static_localize_min_score
            or fine_score < guess_score + self.static_localize_min_score_delta
        ):
            best_pose = guess_base_pose
            best_score = guess_score if math.isfinite(guess_score) else 0.0
            used_guess = True
        else:
            best_pose = fine_pose
            best_score = fine_score

        map_to_odom = compose_pose(best_pose, inverse_pose(odom_base_pose))
        self.static_last_map_to_odom = map_to_odom.copy()

        return StaticLocalizeResult(
            base_pose_in_map=best_pose,
            map_to_odom=map_to_odom,
            score=float(best_score),
            used_guess=used_guess,
        )

    def _expire_static_dynamic_cells(self, now_ns: int) -> None:
        expired = [cell for cell, expiry_ns in self.static_dynamic_cells.items() if expiry_ns <= now_ns]
        for cell in expired:
            del self.static_dynamic_cells[cell]

    def update_static_dynamic_obstacles(
        self,
        scan: LaserScan,
        base_pose_in_map: Pose2D,
        base_to_scan_pose: Pose2D,
        now_ns: int,
    ) -> None:
        if self.static_snapshot is None:
            return

        sensor_pose = compose_pose(base_pose_in_map, base_to_scan_pose)
        expiry_ns = now_ns + int(self.dynamic_obstacle_ttl_sec * 1e9)
        angle = scan.angle_min

        for i, r in enumerate(scan.ranges):
            if is_no_information_ray(scan, i, r):
                angle += scan.angle_increment
                continue
            if not math.isfinite(r) or r < scan.range_min or r >= scan.range_max:
                angle += scan.angle_increment
                continue
            if r > self.static_mode_max_usable_range:
                angle += scan.angle_increment
                continue

            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            c = math.cos(sensor_pose.yaw)
            s = math.sin(sensor_pose.yaw)
            wx = sensor_pose.x + c * lx - s * ly
            wy = sensor_pose.y + s * lx + c * ly

            cell = self._static_world_to_grid(wx, wy)
            if cell is not None:
                ix, iy = cell
                if not self.static_snapshot.occupied_mask[iy, ix]:
                    self.static_dynamic_cells[(ix, iy)] = expiry_ns

            angle += scan.angle_increment

    def process_static_mode_scan(
        self,
        scan: LaserScan,
        odom_base_pose: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> None:
        result = self.localize_scan_on_static_map(scan, odom_base_pose, base_to_scan_pose)
        if result is None:
            return

        now_ns = self.get_clock().now().nanoseconds
        self._expire_static_dynamic_cells(now_ns)
        self.update_static_dynamic_obstacles(
            scan,
            result.base_pose_in_map,
            base_to_scan_pose,
            now_ns,
        )

        self.publish_map_to_odom_tf(scan.header.stamp, result.map_to_odom)
        self.backend.last_map_stamp = scan.header.stamp
        self.backend.map_dirty = True

    def render_static_mode_map_msg(self, stamp) -> Optional[OccupancyGrid]:
        if self.static_snapshot is None:
            return None

        now_ns = self.get_clock().now().nanoseconds
        self._expire_static_dynamic_cells(now_ns)

        log_odds = self.static_snapshot.log_odds.copy()
        observed = self.static_snapshot.observed.copy()

        for (ix, iy), expiry_ns in self.static_dynamic_cells.items():
            if expiry_ns <= now_ns:
                continue
            if 0 <= ix < self.backend.map_width and 0 <= iy < self.backend.map_height:
                log_odds[iy, ix] = max(log_odds[iy, ix], self.dynamic_obstacle_log_odds)
                observed[iy, ix] = True

        probs = 1.0 / (1.0 + np.exp(-log_odds))
        out = np.full((self.backend.map_height, self.backend.map_width), UNKNOWN, dtype=np.int8)
        out[observed] = np.clip(np.round(probs[observed] * 100.0), 0, 100).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.info.resolution = self.backend.map_resolution
        msg.info.width = self.backend.map_width
        msg.info.height = self.backend.map_height
        msg.info.origin.position.x = self.backend.map_origin_x
        msg.info.origin.position.y = self.backend.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = out.reshape(-1).tolist()
        return msg

    def debug_status(self) -> None:

        oldest_queue_age = 0.0
        oldest_scan_age = 0.0

        if self.pending_scans:
            enqueue_time_ns, oldest_scan = self.pending_scans[0]
            now = self.get_clock().now()
            oldest_queue_age = (now.nanoseconds - enqueue_time_ns) / 1e9
            oldest_scan_time = Time.from_msg(oldest_scan.header.stamp)
            oldest_scan_age = (now - oldest_scan_time).nanoseconds / 1e9
        
        active_id = self.frontend.active_submap.frame_id if self.frontend.active_submap else "None"
        prev_id = self.frontend.previous_submap.frame_id if self.frontend.previous_submap else "None"
        active_scans = self.frontend.active_submap.scan_count if self.frontend.active_submap else 0
        active_known = self.frontend.active_submap.known_cells() if self.frontend.active_submap else 0
        num_nodes = len(self.backend.graph_nodes)
        num_edges = len(self.backend.graph_edges)

        num_loop_edges = sum(1 for e in self.backend.graph_edges if e.edge_type == "loop")
        num_archived = sum(1 for n in self.backend.graph_nodes.values() if n.archived)

        msg = (
            f"mode_static={self.mode_enabled}, scan_count={self.scan_count}, tf_hit={self.tf_hit_count}, "
            f"tf_miss={self.tf_miss_count}, drop={self.drop_count}, "
            f"queue_expire={self.queue_expire_count}, pending={len(self.pending_scans)}, "
            f"oldest_queue_age={oldest_queue_age:.3f}s, "
            f"oldest_scan_age={oldest_scan_age:+.3f}s, "
            f"rollovers={self.rollover_count}, active={active_id}, prev={prev_id}, "
            f"active_scans={active_scans}, active_known={active_known}, "
            f"graph_nodes={num_nodes}, graph_edges={num_edges}, "
            f"loop_edges={num_loop_edges}, archived={num_archived}"
        )

        if self.backend.last_optimization_stats is not None:
            st = self.backend.last_optimization_stats
            msg += (
                f", opt_rmse={st.initial_rmse:.4f}->{st.final_rmse:.4f}, "
                f"opt_gain={100.0 * st.improvement_ratio:.1f}%, "
                f"opt_jump_t={st.max_translation_jump_m:.3f}m, "
                f"opt_jump_r={st.max_rotation_jump_deg:.2f}deg"
            )

        self.get_logger().info(msg)

    def scan_callback(self, scan: LaserScan) -> None:
        self.scan_count += 1

        if not scan.header.frame_id:
            self.drop_count += 1
            self.get_logger().warn("Received scan with empty frame_id, dropping.")
            return

        enqueue_time_ns = self.get_clock().now().nanoseconds
        self.pending_scans.append((enqueue_time_ns, scan))

        # Queue chỉ giữ vài scan mới nhất.
        # Nếu xử lý không kịp thì bỏ scan cũ, không cố sống trong quá khứ.
        while len(self.pending_scans) > self.max_pending_scans:
            self.pending_scans.popleft()
            self.drop_count += 1
            self.get_logger().warn(
                f"Pending scan queue full, dropping oldest scan. "
                f"pending={len(self.pending_scans)}"
            )

    def process_pending_scans(self) -> None:
        processed_this_tick = 0

        while (
            self.pending_scans
            and processed_this_tick < self.max_scans_per_process_tick
        ):
            enqueue_time_ns, scan = self.pending_scans[0]

            now = self.get_clock().now()
            now_ns = now.nanoseconds

            queue_age_sec = (now_ns - enqueue_time_ns) / 1e9

            scan_time = Time.from_msg(scan.header.stamp)
            scan_age_sec = (now - scan_time).nanoseconds / 1e9

            # 1) Scan nằm trong queue quá lâu thì bỏ
            if queue_age_sec > self.scan_queue_ttl_sec:
                self.pending_scans.popleft()
                self.drop_count += 1
                self.queue_expire_count += 1

                self.get_logger().warn(
                    f"Dropping expired scan: "
                    f"queue_age={queue_age_sec:.3f}s > ttl={self.scan_queue_ttl_sec:.3f}s, "
                    f"scan_age={scan_age_sec:+.3f}s, "
                    f"pending={len(self.pending_scans)}"
                )
                continue

            # 2) Thử lookup TF tại đúng timestamp của scan
            looked_up = self.lookup_required_transforms(scan)

            if looked_up is None:
                self.tf_miss_count += 1
                self.tf_miss_debug_count += 1

                # In debug định kỳ để biết scan và clock lệch nhau thế nào
                if self.tf_miss_debug_count % max(1, self.debug_timing_every_n) == 0:
                    self.debug_scan_timing(scan, enqueue_time_ns, "TF_MISS")

                # Nếu scan đã chờ TF quá lâu thì bỏ scan này.
                # Không cho nó đứng đầu queue chặn scan mới hơn.
                if queue_age_sec > self.max_wait_tf_sec:
                    self.pending_scans.popleft()
                    self.drop_count += 1
                    self.queue_expire_count += 1

                    self.get_logger().warn(
                        f"Dropping scan because TF not available after waiting: "
                        f"wait={queue_age_sec:.3f}s > max_wait_tf={self.max_wait_tf_sec:.3f}s, "
                        f"scan_age={scan_age_sec:+.3f}s, "
                        f"pending={len(self.pending_scans)}"
                    )
                    continue

                # Scan còn mới, có thể TF chưa kịp đến.
                # Dừng tick này, tick sau thử lại.
                break

            # 3) Có TF thì pop scan ra và xử lý
            self.pending_scans.popleft()
            self.tf_hit_count += 1
            self.processed_scan_count += 1

            if self.processed_scan_count % max(1, self.debug_timing_every_n) == 0:
                self.debug_scan_timing(scan, enqueue_time_ns, "TF_HIT")

            odom_base_pose, base_to_scan_pose = looked_up

            if self.mode_enabled:
                self.process_static_mode_scan(scan, odom_base_pose, base_to_scan_pose)
            else:
                self.process_scan(scan, odom_base_pose, base_to_scan_pose)

            processed_this_tick += 1

    def lookup_required_transforms(
        self,
        scan: LaserScan,
    ) -> Optional[Tuple[Pose2D, Pose2D]]:
        scan_time = Time.from_msg(scan.header.stamp)

        now = self.get_clock().now()
        scan_age_sec = (now - scan_time).nanoseconds / 1e9
        scan_stamp_sec = self.time_msg_to_sec(scan.header.stamp)
        now_sec = now.nanoseconds * 1e-9

        try:
            tf_odom_base = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                scan_time,
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
            odom_base_pose = pose_from_transform_msg(tf_odom_base)

        except TransformException as ex:
            # In warn không quá dày, tránh log phá nát performance
            if self.tf_miss_count % max(1, self.debug_timing_every_n) == 0:
                self.get_logger().warn(
                    f"TF lookup failed: {self.odom_frame} -> {self.base_frame}. "
                    f"now={now_sec:.6f}s, "
                    f"scan_stamp={scan_stamp_sec:.6f}s, "
                    f"scan_age={scan_age_sec:+.3f}s, "
                    f"reason={str(ex)}"
                )
            return None

        try:
            tf_base_scan = self.tf_buffer.lookup_transform(
                self.base_frame,
                scan.header.frame_id,
                scan_time,
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
            base_to_scan_pose = pose_from_transform_msg(tf_base_scan)

        except TransformException as ex:
            if self.tf_miss_count % max(1, self.debug_timing_every_n) == 0:
                self.get_logger().warn(
                    f"TF lookup failed: {self.base_frame} -> {scan.header.frame_id}. "
                    f"now={now_sec:.6f}s, "
                    f"scan_stamp={scan_stamp_sec:.6f}s, "
                    f"scan_age={scan_age_sec:+.3f}s, "
                    f"reason={str(ex)}"
                )
            return None

        return odom_base_pose, base_to_scan_pose

    def process_scan(
        self,
        scan: LaserScan,
        odom_base_pose: Pose2D,
        base_to_scan_pose: Pose2D,
    ) -> None:
        local_state, rollover_event, created_first = self.frontend.process_scan(
            scan,
            odom_base_pose,
            base_to_scan_pose,
        )

        if self.frontend.active_submap is not None and (
            created_first or not self.backend.has_submap(self.frontend.active_submap.id)
        ):
            self.backend.bootstrap_first_submap(self.frontend.active_submap, scan.header.stamp)

        if local_state is not None:
            self.backend.update_local_state(local_state, odom_base_pose, scan.header.stamp)

            latest_map_to_odom = self.backend.get_latest_map_to_odom()
            if latest_map_to_odom is not None:
                self.publish_map_to_odom_tf(scan.header.stamp, latest_map_to_odom)
        elif rollover_event is None:
            self.get_logger().debug("Dropped scan because local tracking quality was too weak.")

        if rollover_event is not None:
            self.backend.handle_rollover_event(rollover_event)

            loop_added = self.backend.try_add_loop_closure(rollover_event.finished_submap.id)
            if loop_added:
                self.get_logger().info(
                    f"Loop closure added for submap {rollover_event.finished_submap.frame_id}"
                )

            optimized = False
            if loop_added or self.backend.optimize_on_every_rollover:
                optimized = self.backend.optimize_graph()
                if optimized:
                    self.get_logger().info("Pose graph optimized and accepted.")
                elif self.backend.last_optimization_stats is not None:
                    st = self.backend.last_optimization_stats
                    self.get_logger().warn(
                        "Optimization result rejected by safety guard: "
                        f"rmse {st.initial_rmse:.4f}->{st.final_rmse:.4f}, "
                        f"gain={100.0 * st.improvement_ratio:.1f}%, "
                        f"jump_t={st.max_translation_jump_m:.3f}m, "
                        f"jump_r={st.max_rotation_jump_deg:.2f}deg"
                    )

            pruned = self.backend.maybe_prune_old_submaps()
            if pruned > 0:
                self.get_logger().info(f"Archived/pruned {pruned} old submap(s).")

            if local_state is not None:
                self.backend.update_local_state(local_state, odom_base_pose, scan.header.stamp)
                latest_map_to_odom = self.backend.get_latest_map_to_odom()
                if latest_map_to_odom is not None:
                    self.publish_map_to_odom_tf(scan.header.stamp, latest_map_to_odom)

            self.rollover_count += 1

    def publish_active_submap_if_dirty(self) -> None:
        if self.frontend.active_submap is None:
            return
        if not self.frontend.active_submap_dirty:
            return
        if self.frontend.last_active_stamp is None:
            return

        msg = self.frontend.active_submap.to_occupancy_grid_msg(self.frontend.last_active_stamp)
        self.active_submap_pub.publish(msg)
        self.frontend.active_submap_dirty = False

    def publish_map_to_odom_tf(self, stamp, map_to_odom: Pose2D) -> None:
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = self.odom_frame

        msg.transform.translation.x = float(map_to_odom.x)
        msg.transform.translation.y = float(map_to_odom.y)
        msg.transform.translation.z = 0.0
        msg.transform.rotation = yaw_to_quaternion(map_to_odom.yaw)

        self.map_to_odom_pub.publish(msg)

    def publish_global_map_if_dirty(self) -> None:
        if not self.backend.map_dirty:
            return

        stamp = self.backend.last_map_stamp
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        if self.mode_enabled:
            msg = self.render_static_mode_map_msg(stamp)
            if msg is not None:
                self.map_pub.publish(msg)
                self.backend.map_dirty = False
            return

        msg = self.backend.render_global_map_msg(stamp)
        self.map_pub.publish(msg)
        self.backend.map_dirty = False

    def destroy_node(self):
        self.get_logger().info("Shutting down slam_submap_graph_node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SlamSubmapGraphNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
