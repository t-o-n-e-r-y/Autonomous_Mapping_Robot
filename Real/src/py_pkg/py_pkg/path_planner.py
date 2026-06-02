#!/usr/bin/env python3

import math
import cv2
import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Union

from std_msgs.msg import Header
from nav_msgs.msg import GridCells, OccupancyGrid, Path
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from tf_transformations import quaternion_from_euler

from queue import PriorityQueue


GridCell = Tuple[int, int]

DIRECTIONS_OF_4: List[GridCell] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS_OF_8: List[GridCell] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


class PathPlanner:
    @staticmethod
    def grid_to_index(mapdata: OccupancyGrid, p: GridCell) -> int:
        """Convert cell coordinate (gx, gy) to flat index in OccupancyGrid.data."""
        return p[1] * mapdata.info.width + p[0]

    @staticmethod
    def get_cell_value(mapdata: OccupancyGrid, p: GridCell) -> int:
        """Return occupancy value of a grid cell."""
        return int(mapdata.data[PathPlanner.grid_to_index(mapdata, p)])

    @staticmethod
    def euclidean_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Euclidean distance between two 2D points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def grid_to_world(mapdata: OccupancyGrid, p: GridCell) -> Point:
        """
        Convert grid cell to world point.
        Assumes map origin is not rotated.
        """
        x = (p[0] + 0.5) * mapdata.info.resolution + mapdata.info.origin.position.x
        y = (p[1] + 0.5) * mapdata.info.resolution + mapdata.info.origin.position.y
        return Point(x=x, y=y, z=0.0)

    @staticmethod
    def world_to_grid(mapdata: OccupancyGrid, wp: Point) -> GridCell:
        """
        Convert world point to grid cell.
        Assumes map origin is not rotated.
        """
        x = int((wp.x - mapdata.info.origin.position.x) / mapdata.info.resolution)
        y = int((wp.y - mapdata.info.origin.position.y) / mapdata.info.resolution)
        return (x, y)

    @staticmethod
    def path_to_poses(
        mapdata: OccupancyGrid,
        path: List[GridCell],
        frame_id: str = "map",
    ) -> List[PoseStamped]:
        """
        Convert grid path to PoseStamped list in world coordinates.

        Keeps the final pose too. The final pose orientation is copied from the
        previous segment if available.
        """
        poses: List[PoseStamped] = []
        if not path:
            return poses

        for i, cell in enumerate(path):
            if len(path) == 1:
                yaw = 0.0
            elif i < len(path) - 1:
                next_cell = path[i + 1]
                yaw = math.atan2(next_cell[1] - cell[1], next_cell[0] - cell[0])
            else:
                prev_cell = path[i - 1]
                yaw = math.atan2(cell[1] - prev_cell[1], cell[0] - prev_cell[0])

            q = quaternion_from_euler(0.0, 0.0, yaw)
            poses.append(
                PoseStamped(
                    header=Header(frame_id=frame_id),
                    pose=Pose(
                        position=PathPlanner.grid_to_world(mapdata, cell),
                        orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
                    ),
                )
            )
        return poses

    @staticmethod
    def is_cell_in_bounds(mapdata: OccupancyGrid, p: GridCell) -> bool:
        x, y = p
        return (
            0 <= x < mapdata.info.width and
            0 <= y < mapdata.info.height
        )

    @staticmethod
    def is_cell_walkable(
        mapdata: OccupancyGrid,
        p: GridCell,
        walkable_threshold: int = 50,
        unknown_is_walkable: bool = False,
    ) -> bool:
        """
        A cell is walkable if:
        1. It is inside map bounds
        2. It is below threshold
        3. Unknown handling follows unknown_is_walkable
        """
        if not PathPlanner.is_cell_in_bounds(mapdata, p):
            return False

        value = PathPlanner.get_cell_value(mapdata, p)

        if value < 0:
            return unknown_is_walkable

        return value < walkable_threshold

    @staticmethod
    def neighbors(
        mapdata: OccupancyGrid,
        p: GridCell,
        directions: List[GridCell],
        must_be_walkable: bool = True,
    ) -> List[GridCell]:
        """Return neighbor cells."""
        out: List[GridCell] = []
        for dx, dy in directions:
            candidate = (p[0] + dx, p[1] + dy)
            if must_be_walkable:
                if PathPlanner.is_cell_walkable(mapdata, candidate):
                    out.append(candidate)
            else:
                if PathPlanner.is_cell_in_bounds(mapdata, candidate):
                    out.append(candidate)
        return out

    @staticmethod
    def neighbors_of_4(
        mapdata: OccupancyGrid,
        p: GridCell,
        must_be_walkable: bool = True,
    ) -> List[GridCell]:
        return PathPlanner.neighbors(mapdata, p, DIRECTIONS_OF_4, must_be_walkable)

    @staticmethod
    def neighbors_of_8(
        mapdata: OccupancyGrid,
        p: GridCell,
        must_be_walkable: bool = True,
    ) -> List[GridCell]:
        return PathPlanner.neighbors(mapdata, p, DIRECTIONS_OF_8, must_be_walkable)

    @staticmethod
    def neighbors_and_distances(
        mapdata: OccupancyGrid,
        p: GridCell,
        directions: List[GridCell],
        must_be_walkable: bool = True,
    ) -> List[Tuple[GridCell, float]]:
        """Return neighbors with motion distance cost."""
        out: List[Tuple[GridCell, float]] = []
        for dx, dy in directions:
            candidate = (p[0] + dx, p[1] + dy)

            if must_be_walkable:
                if not PathPlanner.is_cell_walkable(mapdata, candidate):
                    continue
            else:
                if not PathPlanner.is_cell_in_bounds(mapdata, candidate):
                    continue

            distance = PathPlanner.euclidean_distance((0, 0), (dx, dy))
            out.append((candidate, distance))
        return out

    @staticmethod
    def neighbors_and_distances_of_4(
        mapdata: OccupancyGrid,
        p: GridCell,
        must_be_walkable: bool = True,
    ) -> List[Tuple[GridCell, float]]:
        return PathPlanner.neighbors_and_distances(
            mapdata, p, DIRECTIONS_OF_4, must_be_walkable
        )

    @staticmethod
    def neighbors_and_distances_of_8(
        mapdata: OccupancyGrid,
        p: GridCell,
        must_be_walkable: bool = True,
    ) -> List[Tuple[GridCell, float]]:
        return PathPlanner.neighbors_and_distances(
            mapdata, p, DIRECTIONS_OF_8, must_be_walkable
        )

    @staticmethod
    def get_grid_cells(
        mapdata: OccupancyGrid,
        cells: List[GridCell],
        frame_id: str = "map",
    ) -> GridCells:
        world_cells = [PathPlanner.grid_to_world(mapdata, cell) for cell in cells]
        resolution = float(mapdata.info.resolution)
        return GridCells(
            header=Header(frame_id=frame_id),
            cell_width=resolution,
            cell_height=resolution,
            cells=world_cells,
        )

    @staticmethod
    def calc_cspace(
        mapdata: OccupancyGrid,
        include_cells: bool,
        padding: int = 5,
    ) -> Tuple[OccupancyGrid, Optional[GridCells]]:
        """
        Inflate obstacles to create configuration space.
        Unknown cells are also preserved as blocked.
        """
        width = mapdata.info.width
        height = mapdata.info.height

        # NOTE: reshape should be (height, width)
        raw = np.array(mapdata.data, dtype=np.int16).reshape((height, width))

        # Unknown mask
        unknown_mask = (raw < 0).astype(np.uint8) * 255

        kernel = np.ones((padding, padding), dtype=np.uint8)

        # Treat unknown as free temporarily before obstacle dilation
        work = raw.copy()
        work[work < 0] = 0
        work = np.clip(work, 0, 100).astype(np.uint8)

        obstacle_mask = cv2.dilate(work, kernel, iterations=1)
        cspace_data = cv2.bitwise_or(obstacle_mask, unknown_mask)

        cspace = OccupancyGrid(
            header=mapdata.header,
            info=mapdata.info,
            data=cspace_data.reshape(height * width).astype(np.int8).tolist(),
        )

        cspace_cells: Optional[GridCells] = None
        if include_cells:
            ys, xs = np.where(cspace_data > 0)
            cells = [(int(x), int(y)) for y, x in zip(ys, xs)]
            cspace_cells = PathPlanner.get_grid_cells(mapdata, cells)

        return cspace, cspace_cells

    @staticmethod
    def get_cost_map_value(cost_map: np.ndarray, p: GridCell) -> int:
        return int(cost_map[p[1], p[0]])

    @staticmethod
    def show_map(name: str, map_array: np.ndarray) -> None:
        normalized = cv2.normalize(
            map_array,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        cv2.imshow(name, normalized)
        cv2.waitKey(0)

    @staticmethod
    def calc_cost_map(mapdata: OccupancyGrid) -> np.ndarray:
        """
        Create a penalty map from obstacle clearance.
        Near obstacles -> high penalty
        Far from obstacles -> low penalty
        """
        width = mapdata.info.width
        height = mapdata.info.height

        map_array = np.array(mapdata.data, dtype=np.int16).reshape((height, width))

        # unknown coi như blocked
        blocked = map_array < 0
        blocked |= map_array > 0

        # distanceTransform:
        # non-zero = vùng cần tính khoảng cách
        # zero = vật cản / biên gần nhất
        free_mask = (~blocked).astype(np.uint8)

        dist = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)

        clearance = dist
        penalty = dist.max() - dist

        return penalty

    @staticmethod
    def get_first_walkable_neighbor(
        mapdata: OccupancyGrid,
        start: GridCell,
    ) -> GridCell:
        """
        BFS outward until a walkable cell is found.
        """
        queue: deque[GridCell] = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            if PathPlanner.is_cell_walkable(mapdata, current):
                return current

            for neighbor in PathPlanner.neighbors_of_4(mapdata, current, must_be_walkable=False):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        return start

    @staticmethod
    def a_star(
        mapdata: OccupancyGrid,
        cost_map: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[
        Union[list[tuple[int, int]], None],
        Union[float, None],
        tuple[int, int],
        tuple[int, int],
    ]:
        COST_MAP_WEIGHT = 1000

        # If the start cell is not walkable, get the first walkable neighbor instead
        if not PathPlanner.is_cell_walkable(mapdata, start):
            start = PathPlanner.get_first_walkable_neighbor(mapdata, start)

        # Likewise, if the goal cell is not walkable, get the first walkable neighbor instead
        if not PathPlanner.is_cell_walkable(mapdata, goal):
            goal = PathPlanner.get_first_walkable_neighbor(mapdata, goal)

        pq = PriorityQueue()
        counter = 0
        pq.put((0.0, counter, start))
        counter += 1

        cost_so_far = {}
        distance_cost_so_far = {}
        cost_so_far[start] = 0.0
        distance_cost_so_far[start] = 0.0
        came_from = {}
        came_from[start] = None

        while not pq.empty():
            _, _, current = pq.get()

            if current == goal:
                break

            for neighbor, distance in PathPlanner.neighbors_and_distances_of_8(
                mapdata, current
            ):
                added_cost = (
                    distance
                    + COST_MAP_WEIGHT
                    * PathPlanner.get_cost_map_value(cost_map, neighbor)
                )
                new_cost = cost_so_far[current] + added_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    distance_cost_so_far[neighbor] = (
                        distance_cost_so_far[current] + distance
                    )
                    priority = new_cost + PathPlanner.euclidean_distance(neighbor, goal)
                    pq.put((priority, counter, neighbor))
                    counter += 1
                    came_from[neighbor] = current

        path = []
        cell = goal

        while cell:
            path.insert(0, cell)

            if cell in came_from:
                cell = came_from[cell]
            else:
                return (None, None, start, goal)

        # Prevent paths that are too short
        MIN_PATH_LENGTH = 12
        if len(path) < MIN_PATH_LENGTH:
            return (None, None, start, goal)

        # Truncate the last few poses of the path
        # POSES_TO_TRUNCATE = 8
        # if len(path) > POSES_TO_TRUNCATE:
        #     path = path[:-POSES_TO_TRUNCATE]

        return (path, distance_cost_so_far[goal], start, goal)

    @staticmethod
    def path_to_message(
        mapdata: OccupancyGrid,
        path: List[GridCell],
        frame_id: str = "map",
    ) -> Path:
        """
        Convert grid path to nav_msgs/Path.
        """
        poses = PathPlanner.path_to_poses(mapdata, path, frame_id=frame_id)
        return Path(header=Header(frame_id=frame_id), poses=poses)