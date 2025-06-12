from collections import defaultdict

import numpy as np

from volumembo.heap import CustomHeap
from volumembo.utils import direction_to_grow, assign_clusters


class VolumeMedianFitter:
    def __init__(
        self, u: np.ndarray, lower_limit: np.ndarray, upper_limit: np.ndarray
    ) -> None:
        self.u = u
        self.N, self.M = u.shape
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.median = np.full(self.M, 1.0 / self.M)  # Start at simplex barycenter
        self.labels = assign_clusters(self.u, self.median)
        self.cluster_sizes = np.bincount(self.labels, minlength=self.M)
        self.directions = [direction_to_grow(i, self.M) for i in range(self.M)]
        self.other_labels = [
            [j for j in range(self.M) if j != i] for i in range(self.M)
        ]

        # Priority queues for all (from_cluster, to_cluster) pairs.
        # Each queue is a min-heap (list) of pids
        # Example: {(0, 2): [1, 2, ...]}
        # Meaning: point `pid` currently in cluster `from_cluster` is a candidate to move to cluster `to_cluster`,
        # and will switch clusters if the median `m` moves in the corresponding direction by at least `priority`.
        self.priority_queues: dict[tuple[int, int], list[int]] = defaultdict(list)

        self._initialize_priority_queues()

    @staticmethod
    def fit(
        u: np.ndarray, lower_limit: np.ndarray, upper_limit: np.ndarray
    ) -> np.ndarray:
        fitter = VolumeMedianFitter(u, lower_limit, upper_limit)
        return fitter.run()

    def run(
        self, return_history: bool = False
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        m_history = [self.median.copy()]
        eps = 1.0e-4
        iteration = 0
        max_iter = 100
        offset = 0
        while not self._volumes_matched():
            # print("Iteration {}: {} | {}".format(iteration, self.cluster_sizes))
            if iteration > max_iter:
                break

            cluster = self._select_cluster(offset)
            offset = (cluster + 1) % self.M

            candidates = []
            if self.cluster_sizes[cluster] < self.lower_limit[cluster]:
                # Growing logic
                # print(f"Grow cluster {cluster}")
                to_label = cluster
                for from_label in self.other_labels[to_label]:
                    candidate = self._peek(from_label, to_label)
                    if candidate is not None:
                        candidates.append(
                            (candidate, from_label, to_label)
                        )  # (pid, from, to)
                dir = self.directions[cluster]
            else:
                # Shrinking logic
                # print(f"Shrink cluster {cluster}")
                from_label = cluster
                for to_label in self.other_labels[from_label]:
                    candidate = self._peek(from_label, to_label)
                    if candidate is not None:
                        candidates.append(
                            (candidate, from_label, to_label)
                        )  # (pid, from, to)
                dir = -self.directions[cluster]

            if not candidates:
                raise RuntimeError(
                    f"No valid candidates for cluster {cluster} at iteration {iteration}"
                )

            # Recompute flip time for all candidates and pick the best one
            flip_time = float("inf")
            pid = from_label = to_label = None
            for candidate_pid, f_lbl, t_lbl in candidates:
                ft = self._compute_flip_time(candidate_pid, f_lbl, t_lbl)
                if ft < flip_time:
                    flip_time = ft
                    pid = candidate_pid
                    from_label = f_lbl
                    to_label = t_lbl

            # Update median
            self.median += (1 + eps) * flip_time * dir
            m_history.append(self.median.copy())
            # print(
            #    "m = {} ({}) | flip time = {}, dir = {}".format(
            #        self.median, np.sum(self.median), flip_time, dir
            #    )
            # )

            # Update queues
            self._remove(pid, from_label)
            self._insert_into_queues(pid, to_label)

            # Update clusters
            self.labels[pid] = to_label
            self.cluster_sizes[from_label] -= 1
            self.cluster_sizes[to_label] += 1
            # self.labels = assign_clusters(self.u, self.median)
            # self.cluster_sizes = np.bincount(self.labels, minlength=self.M)
            # print("Cluster {}\n".format(self.cluster_sizes))

            iteration += 1

        if return_history:
            return self.labels, m_history
        return self.labels

    def _compute_flip_time(self, pid: int, from_label: int, to_label: int) -> float:
        """
        Compute the flip time of a point in the direction d_i associated with cluster from_label.

        Args:
            pid (int): point ID.
            from_label (int): Current cluster assignment.
            to_label (int): Competing cluster index.

        Returns:
            float: Distance m has to be moved such that the point would flip from from_cluster to to_cluster
        """
        u_minus_m = self.u[pid] - self.median
        return (u_minus_m[from_label] - u_minus_m[to_label]) * (self.M - 1) / self.M

    def _initialize_priority_queues(self) -> None:
        for pid in range(self.N):
            from_label = self.labels[pid]
            for to_label in self.other_labels[from_label]:
                key = (from_label, to_label)
                if key not in self.priority_queues:
                    comparator = self._make_comparator(*key)
                    self.priority_queues[key] = CustomHeap(comparator)
                self.priority_queues[key].heappush(pid)

    def _insert_into_queues(self, pid: int, from_label: int) -> None:
        for to_label in self.other_labels[from_label]:
            key = (from_label, to_label)
            self.priority_queues[key].heappush(pid)

    def _make_comparator(self, from_label: int, to_label: int):
        def comparator(pid0: int, pid1: int) -> int:
            t0 = self._compute_flip_time(pid0, from_label, to_label)
            t1 = self._compute_flip_time(pid1, from_label, to_label)
            return -1 if t0 < t1 else (1 if t0 > t1 else 0)

        return comparator

    def _peek(self, from_label: int, to_label: int) -> int | None:
        heap = self.priority_queues[(from_label, to_label)]
        return heap.heappeek() if heap else None

    def _remove(self, pid: int, from_label: int) -> None:
        for to_label in self.other_labels[from_label]:
            self.priority_queues[(from_label, to_label)].heapremove(
                pid
            )  # Delete directly from heap

    def _select_cluster(self, offset: int) -> int:
        """Select the cluster to adjust next.

        Prefers clusters with large violations but biases toward round-robin cycling via offset. Only when at least two labels have maximum violation, the cycling is used. Otherwise the label with maximum violation is chosen.
        """
        # Violation magnitude: how far outside the allowed bounds (0 if inside)
        under = np.clip(self.lower_limit - self.cluster_sizes, 0, None)
        over = np.clip(self.cluster_sizes - self.upper_limit, 0, None)
        violations = np.maximum(under, over)

        if not np.any(violations):
            return offset % self.M  # Fallback

        # Rotate violations to implement soft cycling
        rotated = np.roll(violations, -offset)
        idx_in_rotated = np.argmax(rotated)
        return (offset + idx_in_rotated) % self.M

    def _volumes_matched(self) -> bool:
        return np.all(
            (self.cluster_sizes >= self.lower_limit)
            & (self.cluster_sizes <= self.upper_limit)
        )
