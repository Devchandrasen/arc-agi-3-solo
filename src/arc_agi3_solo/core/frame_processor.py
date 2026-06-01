"""Frame-segmentation and hashing utilities used by the GraphAgent.

Ported from `vendor/just-explore/agents/heuristic_agent.py` (dolphin-in-a-coma,
MIT, Copyright (c) 2025 Evgenii Rudakov). Plotting/visualization code stripped.

Responsibilities:
    - segment_frame: 4-connected color-component segmentation with twin detection
    - identify_status_bars: rule-based detection of edge-adjacent UI segments
    - hash_frame: stable 128-bit hash for a 0-15 valued grid
    - frame_segments_to_action_groups: bucket segments into priority groups
      (salient+medium, medium, salient, other, status-bar)
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class FrameProcessor:
    OFFSETS4: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
    OFFSETS8: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1),
    )

    def __init__(self) -> None:
        self.connectivity_rank = 4
        self.status_bar_mode = "rule"
        self.status_bar_distance_threshold = 3
        self.status_bar_ratio_threshold = 5
        self.status_bar_twins_threshold = 3
        self.frame_shape = (64, 64)
        self.status_bar_color = 16
        self.minimal_width = 2
        self.maximal_width = 32
        self.non_salient_color = {0, 1, 2, 3, 4, 5}
        self.salient_color = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

    def segment_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        h, w = frame.shape
        label_map = np.zeros((h, w), dtype=int) - 1
        components: List[Dict] = []
        cid = -1
        offsets = self.OFFSETS4 if self.connectivity_rank == 4 else self.OFFSETS8

        for y in range(h):
            for x in range(w):
                if label_map[y, x] != -1:
                    continue
                cid += 1
                color = int(frame[y, x])
                q = deque([(y, x)])
                label_map[y, x] = cid
                min_x = max_x = x
                min_y = max_y = y
                area = 0
                while q:
                    cy, cx = q.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)
                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h and 0 <= nx < w
                            and label_map[ny, nx] == -1
                            and frame[ny, nx] == color
                        ):
                            label_map[ny, nx] = cid
                            q.append((ny, nx))
                rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                components.append(dict(
                    bounding_box=(min_x, min_y, max_x, max_y),
                    color=color,
                    area=area,
                    is_rectangle=area == rect_area,
                ))

        for i, comp in enumerate(components):
            twins = [
                j for j, other in enumerate(components)
                if i != j
                and other["area"] == comp["area"]
                and other["is_rectangle"] == comp["is_rectangle"]
                and other["color"] == comp["color"]
            ]
            comp["number_of_twins"] = len(twins)
            comp["twin_ids"] = twins
        return label_map, components

    def identify_status_bars(
        self, segmented_frame: np.ndarray, frame_segments: List[Dict],
    ) -> Tuple[Optional[List[List[Dict]]], np.ndarray]:
        if self.status_bar_mode == "crude":
            return None, self._identify_status_bars_crude()
        if self.status_bar_mode == "rule":
            return self._identify_status_bars_with_rule(segmented_frame, frame_segments)
        raise ValueError(f"unsupported status bar mode: {self.status_bar_mode}")

    def _identify_status_bars_crude(self) -> np.ndarray:
        m = np.zeros(self.frame_shape, dtype=bool)
        t = self.status_bar_distance_threshold
        m[:t, :] = True
        m[-t:, :] = True
        m[:, :t] = True
        m[:, -t:] = True
        return m

    def _identify_status_bars_with_rule(
        self, segmented_frame: np.ndarray, frame_segments: List[Dict],
    ) -> Tuple[List[List[Dict]], np.ndarray]:
        checked: set = set()
        ids_list: List[List[int]] = []
        for i, segment in enumerate(frame_segments):
            if i in checked:
                continue
            checked.add(i)
            on_edges = self._check_segment_fully_on_edge(segment, edges=["any"])
            if not on_edges:
                continue
            directions = []
            if "left" in on_edges or "right" in on_edges:
                directions.append("vertical")
            if "top" in on_edges or "bottom" in on_edges:
                directions.append("horizontal")
            direction = "any" if len(directions) == 2 else directions[0]
            is_long = self._check_segment_ratio(segment, direction=direction)
            ids = [i]
            if not is_long:
                twin_ids = self._segment_twins_on_edge(segment, frame_segments)
                for tid in twin_ids:
                    checked.add(tid)
                if len(twin_ids) + 1 < self.status_bar_twins_threshold:
                    continue
                ids.extend(twin_ids)
            ids_list.append(ids)

        segments_list: List[List[Dict]] = []
        mask = np.zeros(segmented_frame.shape, dtype=bool)
        for ids in ids_list:
            group = []
            for sid in ids:
                mask[segmented_frame == sid] = True
                group.append(frame_segments[sid])
            segments_list.append(group)
        return segments_list, mask

    def _check_segment_fully_on_edge(self, segment: Dict, edges: Optional[List[str]] = None) -> List[str]:
        x1, y1, x2, y2 = segment["bounding_box"]
        edges = edges or ["any"]
        result: List[str] = []
        if "left" in edges or "any" in edges:
            if max(x1, x2) < self.status_bar_distance_threshold:
                result.append("left")
        if "right" in edges or "any" in edges:
            if min(x1, x2) > self.frame_shape[1] - self.status_bar_distance_threshold:
                result.append("right")
        if "top" in edges or "any" in edges:
            if max(y1, y2) < self.status_bar_distance_threshold:
                result.append("top")
        if "bottom" in edges or "any" in edges:
            if min(y1, y2) > self.frame_shape[0] - self.status_bar_distance_threshold:
                result.append("bottom")
        return result

    def _check_segment_ratio(self, segment: Dict, direction: Optional[str] = None) -> bool:
        direction = direction or "any"
        x_len = segment["bounding_box"][2] - segment["bounding_box"][0] + 1
        y_len = segment["bounding_box"][3] - segment["bounding_box"][1] + 1
        ratio = x_len / y_len
        if ratio >= self.status_bar_ratio_threshold and direction in ("any", "horizontal"):
            return True
        if ratio <= 1 / self.status_bar_ratio_threshold and direction in ("any", "vertical"):
            return True
        return False

    def _segment_twins_on_edge(
        self, segment: Dict, frame_segments: List[Dict], edges: Optional[List[str]] = None,
    ) -> List[int]:
        if edges is None:
            edges = self._check_segment_fully_on_edge(segment, edges=["any"])
            if not edges:
                return []
        twins = []
        for tid in segment["twin_ids"]:
            if self._check_segment_fully_on_edge(frame_segments[tid], edges=edges):
                twins.append(tid)
        return twins

    @staticmethod
    def hash_frame(frame: np.ndarray) -> str:
        """Stable 128-bit hash for a 0-15 valued NumPy array; shape-aware."""
        frame = np.asarray(frame, dtype=np.uint8, order="C")
        flat = frame.ravel()
        if flat.size & 1:
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
        return hashlib.blake2b(
            packed.tobytes(),
            digest_size=16,
            person=repr(frame.shape).encode(),
        ).hexdigest()

    def frame_segments_to_action_groups(
        self, frame_segments: List[Dict], n_groups: int,
    ) -> List[set]:
        """Bucket segments into priority groups by salience and size.

        Group 0 = salient + medium-size (most likely interactive UI).
        Group 1 = medium-size only.
        Group 2 = salient only.
        Group 3 = everything else except status bars.
        Group 4 = status bars.
        """
        if n_groups != 5:
            raise ValueError("only n_groups == 5 is currently supported")
        groups: List[set] = [set(), set(), set(), set(), set()]
        for sid, segment in enumerate(frame_segments):
            x_w = segment["bounding_box"][2] - segment["bounding_box"][0] + 1
            y_w = segment["bounding_box"][3] - segment["bounding_box"][1] + 1
            is_salient = segment["color"] in self.salient_color
            is_medium = self.minimal_width <= x_w <= self.maximal_width and self.minimal_width <= y_w <= self.maximal_width
            is_status_bar = segment["color"] == self.status_bar_color
            if is_salient and is_medium:
                groups[0].add(sid)
            elif is_medium:
                groups[1].add(sid)
            elif is_salient:
                groups[2].add(sid)
            elif not is_status_bar:
                groups[3].add(sid)
            else:
                groups[4].add(sid)
        return groups
