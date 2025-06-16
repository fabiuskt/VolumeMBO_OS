from collections.abc import Sequence

import matplotlib as mpl
import numpy as np


class SimplexPlotter:
    def __init__(self, ax: mpl.axes.Axes) -> None:
        self.ax = ax or mpl.pyplot.gca()
        self.A = np.array([0.0, 0.0])
        self.B = np.array([1.0, 0.0])
        self.C = np.array([0.5, np.sqrt(3) / 2])
        self.triangle = np.array([self.A, self.B, self.C, self.A])

        self.color0 = "deepskyblue"
        self.color1 = "gold"
        self.color2 = "magenta"

        mpl.rcParams["text.usetex"] = True

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(left=-0.075, right=1.075)
        ax.set_ylim(bottom=-0.075, top=1.075)

    def add_grid_lines(self, n: int = 10, **kwargs) -> None:
        """Draw grid lines inside the simplex.

        Args:
            n (int): Number of grid lines per direction.
            **kwargs: Additional keyword arguments for line properties.
        """
        # Set defaults only if not already provided
        kwargs.setdefault("color", "lightgray")
        kwargs.setdefault("linewidth", 0.5)
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("zorder", 0)

        # Iterate through barycentric grid values (excluding endpoints)
        for i in range(1, n):
            t = i / n

            # Lines of constant x (i.e., from B to C)
            start = self._simplex_to_cartesian(np.array([[1 - t, t, 0]]))
            end = self._simplex_to_cartesian(np.array([[1 - t, 0, t]]))
            self.ax.plot([start[0, 0], end[0, 0]], [start[0, 1], end[0, 1]], **kwargs)

            # Lines of constant y (i.e., from C to A)
            start = self._simplex_to_cartesian(np.array([[0, 1 - t, t]]))
            end = self._simplex_to_cartesian(np.array([[t, 1 - t, 0]]))
            self.ax.plot([start[0, 0], end[0, 0]], [start[0, 1], end[0, 1]], **kwargs)

            # Lines of constant z (i.e., from A to B)
            start = self._simplex_to_cartesian(np.array([[t, 0, 1 - t]]))
            end = self._simplex_to_cartesian(np.array([[0, t, 1 - t]]))
            self.ax.plot([start[0, 0], end[0, 0]], [start[0, 1], end[0, 1]], **kwargs)

    def add_ticks(
        self,
        n: int = 5,
        tick_length: float = 0.025,
        show_labels: bool = False,
        label_format: str = "{:.1f}",
        **kwargs,
    ) -> None:
        """
        Add ticks along the edges of the simplex, aligned with the barycentric axes.

        Args:
            n (int): Number of ticks per axis.
            tick_length (float): Length of tick marks.
            show_labels (bool): Whether to show numerical labels.
            label_format (str): Format string for tick labels.
            **kwargs: Additional styling passed to plot.
        """
        kwargs.setdefault("color", "black")
        kwargs.setdefault("linewidth", 2)
        kwargs.setdefault("solid_capstyle", "round")

        edges = [
            (self.A, self.B),  # edge AB
            (self.B, self.C),  # edge BC
            (self.A, self.C),  # edge AC
        ]

        directions = [
            np.array([0.5, -np.sqrt(3) / 2]),
            np.array([0.5, np.sqrt(3) / 2]),
            np.array([-1.0, 0]),
        ]

        directions[0] /= np.linalg.norm(directions[0])
        directions[1] /= np.linalg.norm(directions[1])
        directions[2] /= np.linalg.norm(directions[2])

        for i, (start, end) in enumerate(edges):
            for j in range(0, n + 1):
                t = j / n
                # Interpolate along the edge
                point_on_edge = (1 - t) * start + t * end

                # Offset and draw tick
                tick_start = point_on_edge
                tick_end = tick_start + 0.5 * tick_length * directions[i]
                self.ax.plot(
                    [tick_start[0], tick_end[0]], [tick_start[1], tick_end[1]], **kwargs
                )

                # Optional label
                if show_labels:
                    label_pos = tick_start + 1.75 * tick_length * directions[i]
                    if i == 0 or i == 1:
                        label = label_format.format(1 - t)
                    else:
                        label = label_format.format(t)
                    self.ax.text(
                        label_pos[0],
                        label_pos[1],
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    def plot_median(self, point: np.ndarray, color: str = "k", **kwargs) -> None:
        """Plot the median point in the simplex.

        Args:
            point (np.ndarray): Median point in barycentric coordinates, shape (3,).
            color (str): Color for the median point.
            **kwargs: Additional arguments for scatter plot.
        """
        self.plot_points(point, color=color, **kwargs)

        kwargs.setdefault("zorder", 2)

        # Extract and remove known scatter-only kwargs before passing to plot
        plot_kwargs = kwargs.copy()
        for bad_kw in ["s", "c"]:
            plot_kwargs.pop(bad_kw, None)

        ### Project the median point to the simplex outline
        point_cartesian = self._simplex_to_cartesian(point.reshape(1, -1)).flatten()
        # Project to each edge of the triangle
        foot_a = self._project_to_segment(point_cartesian, self.A, self.B)
        foot_b = self._project_to_segment(point_cartesian, self.B, self.C)
        foot_c = self._project_to_segment(point_cartesian, self.C, self.A)
        # Plot lines from median to the feet of the perpendiculars
        self.ax.plot(
            [point_cartesian[0], foot_a[0]],
            [point_cartesian[1], foot_a[1]],
            color=color,
            **plot_kwargs,
        )
        self.ax.plot(
            [point_cartesian[0], foot_b[0]],
            [point_cartesian[1], foot_b[1]],
            color=color,
            **plot_kwargs,
        )
        self.ax.plot(
            [point_cartesian[0], foot_c[0]],
            [point_cartesian[1], foot_c[1]],
            color=color,
            **plot_kwargs,
        )

    def plot_points(
        self, points: np.ndarray, labels: np.ndarray | None = None, **kwargs
    ) -> None:
        """Scatter points in the simplex.

        Args:
            points (np.ndarray): Points in barycentric coordinates, shape (3,) or (N, 3).
            labels (np.ndarray or None): Optional cluster labels for coloring.
            **kwargs: Additional arguments for scatter plot.
        """
        kwargs.setdefault("zorder", 3)

        points = np.atleast_2d(points)
        pts = self._mapped_points(points)

        if labels is not None:
            self.ax.scatter(
                pts[labels == 0, 0], pts[labels == 0, 1], color=self.color0, **kwargs
            )
            self.ax.scatter(
                pts[labels == 1, 0], pts[labels == 1, 1], color=self.color1, **kwargs
            )
            self.ax.scatter(
                pts[labels == 2, 0], pts[labels == 2, 1], color=self.color2, **kwargs
            )
        else:
            if "color" not in kwargs:
                kwargs["color"] = "k"
            self.ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

    def plot_simplex_outline(self, color: str = "k", **kwargs) -> None:
        """Plot the outline of the simplex triangle.

        Args:
            color (str): Color of the triangle edges. Default is "k" (black).
            **kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.plot`,
                    such as linewidth, linestyle, etc.
        """
        kwargs.setdefault("zorder", 1)
        kwargs.setdefault("solid_capstyle", "round")

        self.ax.plot(self.triangle[:, 0], self.triangle[:, 1], color=color, **kwargs)

    def plot_trace(self, history: list[np.ndarray], **kwargs) -> None:
        """Plot the trace of simplex points over time as a line on the triangle.

        Args:
            history (list of np.ndarray): A sequence of 3D simplex coordinates (e.g., barycentric medians)
                                        where each point sums to 1 and represents a position in the simplex.
            **kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.plot`,
                    such as color, linewidth, alpha, etc.
        """
        points = np.stack(history)  # shape (T, 3)
        coords = self._mapped_points(points)  # shape (T, 2)

        self.ax.plot(coords[:, 0], coords[:, 1], **kwargs)

    def set_axis_labels(
        self,
        labels: Sequence[str] = (r"$u_0$", r"$u_1$", r"$u_2$"),
        offset: float = 0.075,
        **kwargs,
    ) -> None:
        """Set axis labels centered on each edge of the simplex triangle.

        Args:
            labels (tuple of str): Labels for the axes opposite each vertex.
            offset (float): Offset outward normal to each triangle edge.
            **kwargs: Extra text styling options (e.g., fontsize).
        """
        # Compute midpoints of each edge
        mid_AB = 0.5 * (self.A + self.B)
        mid_BC = 0.5 * (self.B + self.C)
        mid_CA = 0.5 * (self.C + self.A)

        # Compute normals (perpendicular to edges, pointing outward)
        def outward_normal(p1, p2):
            v = p2 - p1
            n = np.array([-v[1], v[0]])  # 90 deg rotation
            n /= np.linalg.norm(n)
            return n

        n_AB = outward_normal(self.A, self.B)
        n_BC = outward_normal(self.B, self.C)
        n_CA = outward_normal(self.C, self.A)

        # Offset midpoints along outward normals
        pos_AB = mid_AB - offset * n_AB
        pos_BC = mid_BC - offset * n_BC
        pos_CA = mid_CA - offset * n_CA

        # Plot labels centered on edges
        self.ax.text(
            *pos_BC, labels[1], ha="center", va="center", rotation=-60, **kwargs
        )
        self.ax.text(
            *pos_CA, labels[2], ha="center", va="center", rotation=60, **kwargs
        )
        self.ax.text(*pos_AB, labels[0], ha="center", va="center", rotation=0, **kwargs)

    def set_colors(
        self,
        color0: str | None = None,
        color1: str | None = None,
        color2: str | None = None,
    ) -> None:
        """Set colors for the three simplex corners."""
        if color0 is not None:
            self.color0 = color0
        if color1 is not None:
            self.color1 = color1
        if color2 is not None:
            self.color2 = color2

    @staticmethod
    def _project_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Project point p onto segment a-b, return foot of perpendicular.
        Args:
            point (np.ndarray): Point to project, shape (2,).
            a (np.ndarray): Start of segment, shape (2,).
            b (np.ndarray): End of segment, shape (2,).
        Returns:
            np.ndarray: Foot of perpendicular from point to segment a-b, shape (2,).
        """
        ap = point - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
        return a + t * ab

    @staticmethod
    def _project_to_simplex_plane(v: np.ndarray) -> np.ndarray:
        """
        Project a 3D vector or array of vectors onto the plane x + y + z = 1.
        This does NOT clip to the triangle; values may be negative.
        """
        v = np.atleast_2d(v)
        n = np.array([1.0, 1.0, 1.0])  # normal vector
        n_dot_v = v @ n
        alpha = (n_dot_v - 1) / (n @ n)  # scalar projection
        projection = v - alpha[:, np.newaxis] * n
        return projection

    def _mapped_points(self, points: np.ndarray) -> np.ndarray:
        """Project points onto simplex along normals, then get cartesian coordinates for plotting."""
        return self._simplex_to_cartesian(self._project_to_simplex_plane(points))

    def _simplex_to_cartesian(self, point: np.ndarray) -> np.ndarray:
        """Map 3D simplex point to 2D triangle for visualization.
        The input p is a 3D point in barycentric coordinates (x, y, z) such that x + y + z = 1.
        The output is a 2D point in Cartesian coordinates.
        """
        return point[:, 0:1] * self.A + point[:, 1:2] * self.B + point[:, 2:3] * self.C
