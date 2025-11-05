import os
from datetime import datetime

import graphlearning as gl
import numpy as np
import scipy as sp

from volumembo.legacy import fit_median as fit_median_legacy
from volumembo.timer import TimingManager
from volumembo.utils import (
    bellman_ford_voronoi_initialization,
    onehot_to_labels,
)

from _volumembo import fit_median_cpp


class MBO:
    def __init__(
        self,
        *,
        labels: np.ndarray | None = None,
        data: np.ndarray | None = None,
        weight_matrix: sp.sparse.spmatrix | None = None,
        number_of_neighbors: int = 6,
        **kwargs,
    ) -> None:
        """
        :labels:
            Labels for the dataset, (N,) shape, where N is the number of points.
        :data:
            Dataset, (N, P) shape, where N is the number of points and P is the number of features/labels
        :weight_matrix:
            Precomputed weight matrix, if None, it will be computed from the data
            using k-nearest neighbors with a Gaussian kernel.
            If provided, it should be a sparse matrix of shape (N, N).
        :number_of_neighbors:
            Number of nearest neighbors to consider for the weight matrix
        :initial_clustering_method:
            Method for initial clustering, an be one of:
            - "bellman_ford": Bellman-Ford initialization
            - "random": Random initialization
            - "voronoi": Voronoi initialization
            - "laguerre": Laguerre initialization
            - "diffusion": Diffusion-based initialization
            - "diffusion_volume": Diffusion-based initialization with volume constraints
        :lower_limit:
            Lower limit for each cluster, if None, it will be set to the volume of each label
        :upper_limit:
            Upper limit for each cluster, if None, it will be set to the volume of each label
        :number_of_known_labels:
            Number of known labels per class
        :diffusion_method:
            Method for diffusion, can be one of:
            - "spectral": spectral decomposition
            - "expm_multiply": exponential matrix multiplication
            - "A_3": taylor expansion of the exponential matrix
            - "W2": normalized weight matrix squared
            - "W_W2": fourth power of the weight matrix
            - "A_minus_eig": squared weight matrix minus fraction of identity
        :diffusion_time:
            Diffusion time for the graph diffusion process
        :temperature:
            Temperature, as in Auction Dynamics by Jacobs et al., JCP 354, 288-310 (2018) (DOI: 10.1016/j.jcp.2017.10.036)
        :threshold_method:
            Method for thresholding the diffused values, can be one of:
            - "argmax": simple argmax thresholding
            - "fit_median_cpp": fit median thresholding using C++ implementation
            - "fit_median_legacy": legacy implementation of fit median thresholding
        """
        self.A_3: sp.sparse.spmatrix | None = None
        self.A_minus_eig: sp.sparse.spmatrix | None = None
        self.data: np.ndarray | None = None
        self.diffusion_method: str | None = None
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None
        self.fidelity_set: np.ndarray | None = None
        self.initial_cluster: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.lower_limit: np.ndarray | None = None
        self.number_of_known_labels: int | None = None
        self.number_of_labels: int | None = None
        self.number_of_neighbors: int | None = None
        self.number_of_points: int | None = None
        self.temperature: float | None = None
        self.upper_limit: np.ndarray | None = None
        self.volume: np.ndarray | None = None
        self.W2: sp.sparse.spmatrix | None = None
        self.W_W2: sp.sparse.spmatrix | None = None
        self.weight_matrix: sp.sparse.spmatrix | None = None

        if labels is None:
            raise ValueError("Labels must be provided.")
        else:
            self.labels = labels
            self.number_of_labels = len(np.unique(labels))

        if weight_matrix is not None and labels is not None:
            self.number_of_points = weight_matrix.shape[0]
            self.weight_matrix = weight_matrix
        elif data is not None:
            self.data = data
            self.number_of_points = data.shape[0]
            self.number_of_neighbors = number_of_neighbors
            self.weight_matrix = gl.weightmatrix.knn(
                data,
                number_of_neighbors,
                metric="euclidean",
                kernel="gaussian",
                symmetrize=True,
            )
        else:
            raise ValueError(
                "Either data or a precomputed weight_matrix must be provided."
            )

        DEFAULTS = {
            "diffusion_time": 1.0,
            "temperature": None,
            "lower_limit": None,
            "upper_limit": None,
            "number_of_known_labels": 5,
            "diffusion_method": "spectral",
            "initial_clustering_method": "diffusion",
            "threshold_method": "fit_median_cpp",
        }
        for key in kwargs:
            if key not in DEFAULTS:
                raise TypeError(f"Unexpected keyword argument: {key}")
        config = {**DEFAULTS, **kwargs}

        self.diffusion_time = config["diffusion_time"]
        self.temperature = config["temperature"]

        if labels is not None:
            # Infer number of labels from dataset
            self.number_of_points = len(labels)
        else:
            # Get number of labels from input or default
            self.number_of_known_labels = config["number_of_known_labels"]

        # Compute volume of each label
        if labels is not None:
            self.volume = np.bincount(self.labels, minlength=self.number_of_labels)
        else:
            self.volume = np.zeros(self.number_of_labels)

        # Set lower and upper limit for each cluster
        llimit = config["lower_limit"]
        if llimit is not None and len(llimit) == self.number_of_labels:
            self.lower_limit = llimit
        else:
            self.lower_limit = self.volume
        # Set upper limit for each cluster
        ulimit = config["upper_limit"]
        if ulimit is not None and len(ulimit) == self.number_of_labels:
            self.upper_limit = llimit
        else:
            self.upper_limit = self.volume

        self.number_of_known_labels = config["number_of_known_labels"]

        # Timer for performance measurement
        self.timer = TimingManager(enable=True)

        # Initialize variables
        with self.timer.time("build_matrices"):
            self._compute_diffusion_matrices()

        # Set diffusion, initial clustering, threshold methods
        self.set_diffusion_method(config["diffusion_method"])
        self.set_clustering_initialization_function(config["initial_clustering_method"])
        self.set_threshold_function(config["threshold_method"])

    def get_initial_cluster(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the initial clustering based on the specified method.
        Returns:
            tuple: Initial clustering (chi), labels, and fidelity set.
            If initial_cluster and fidelity_set are provided, they are returned directly.
            Otherwise, a new fidelity set is generated and the initial clustering is computed.
        """

        # Initialize the clustering
        if self.initial_cluster is not None and self.fidelity_set is not None:
            return (
                self.initial_cluster,
                onehot_to_labels(self.initial_cluster),
                self.fidelity_set,
            )
        else:
            fidelity_set = gl.trainsets.generate(
                self.labels, rate=self.number_of_known_labels
            )
            chi = self.cluster_initialization_function(fidelity_set)
            return chi, onehot_to_labels(chi), fidelity_set

    def make_fidelity_set(self) -> None:
        """
        Create a fidelity set based on the specified number of known labels per class.
        """
        self.fidelity_set = gl.trainsets.generate(
            self.labels, rate=self.number_of_known_labels
        )

    def print_parameters(self) -> None:
        """Print the parameters of the VolumeMBO instance."""
        if self.number_of_points is not None:
            print(f"Number of points: {self.number_of_points}")
        if self.number_of_labels is not None:
            print(f"Number of labels: {self.number_of_labels}")
        if self.number_of_neighbors is not None:
            print(f"Number of neighbors: {self.number_of_neighbors}")
        print(f"Diffusion time: {self.diffusion_time}")
        if self.temperature is not None:
            print(f"Temperature: {self.temperature}")
        print(f"Number of known labels per class: {self.number_of_known_labels}")

        if self.data is not None:
            print(f"\nDataset shape: {self.data.shape}")
        if self.labels is not None:
            print(f"Labels shape: {self.labels.shape}")
        if self.weight_matrix is not None:
            print(f"Weight matrix shape: {self.weight_matrix.shape}")
        if self.volume is not None:
            print(f"Volume: {self.volume}")
        if self.upper_limit is not None:
            print(f"Upper limit: {self.upper_limit}")
        if self.lower_limit is not None:
            print(f"Lower limit: {self.lower_limit}")

        if self.diffusion_method is not None:
            print(f"\nDiffusion method: {self.diffusion_method}")
        if self.W2 is not None:
            print(f"W2 shape: {self.W2.shape}")
        if self.A_3 is not None:
            print(f"A_3 shape: {self.A_3.shape}")
        if self.W_W2 is not None:
            print(f"W_W2 shape: {self.W_W2.shape}")
        if self.A_minus_eig is not None:
            print(f"A_minus_eig shape: {self.A_minus_eig.shape}")
        if self.eigenvectors is not None:
            print(f"Eigenvectors shape: {self.eigenvectors.shape}")
        if self.eigenvalues is not None:
            print(f"Eigenvalues shape: {self.eigenvalues.shape}")

    def run_mbo(
        self, max_iterations: int = 100, tolerance: float = 1e-6, verbose: bool = False
    ) -> float:
        """Main MBO loop for clustering
        Args:
            max_iterations (int): Maximum number of iterations to run
            tolerance (float): Tolerance for convergence
            verbose (bool): Whether to print progress and results
        Returns:
            float: Accuracy of the clustering after MBO iterations
        """

        # Get fidelity set
        if self.fidelity_set is None:
            self.fidelity_set = gl.trainsets.generate(
                self.labels, rate=self.number_of_known_labels
            )

        # Initialize the clustering
        chi = self.cluster_initialization_function(self.fidelity_set)
        self.initial_cluster = chi.copy()

        # Initial values for tracking best result
        min_energy = np.inf
        relative_energy_change = np.inf

        # Thresholding energy (used as stopping criterion)
        energy = np.inf

        # Run the MBO iteration loop
        for count in range(max_iterations):
            if not self.temperature and relative_energy_change < tolerance:
                break

            # Diffuse the current clustering
            with self.timer.time("diffusion"):
                # u = np.require(
                #    self.diffuse(chi), dtype=np.float64, requirements=["C_CONTIGUOUS"]
                # )
                u = self.diffuse(chi)

            # Add temperature noise if configured
            if self.temperature is not None:
                u += np.random.normal(0, self.temperature, size=u.shape)

            # Calculate the energy
            new_energy = np.sum(u * (1 - chi))

            relative_energy_change = (energy - new_energy) / max(
                new_energy, 1e-10
            )  # Avoid division by zero
            energy = new_energy

            # Track the best clustering (based on minimum energy)
            if energy < min_energy:
                min_energy = energy
                min_chi = chi.copy()

            # Update the clustering (one-hot encoding of the diffused values)
            with self.timer.time("threshold"):
                chi = self.threshold_function(u)
            chi = self._apply_fidelity_set(chi)

        self.new_labels = onehot_to_labels(min_chi)
        self.new_chi = min_chi
        self.new_volume = np.bincount(self.new_labels, minlength=self.number_of_labels)
        accuracy = gl.ssl.ssl_accuracy(self.labels, self.new_labels, self.fidelity_set)
        if verbose:
            print(
                f"MBO finished after {count} iterations with {accuracy} % accuracy. Energy: {min_energy}"
            )
        return accuracy

    def run(
        self,
        iterations: int = 1,
        save_results: bool = False,
        output_dir: str = "results",
        dataset_name: str = "unknown",
        enable_timing: bool = False,
    ) -> None:
        """
        Run the MBO algorithm for a specified number of iterations (different fidelity sets) and save results if requested.
        Args:
            iterations (int): Number of MBO iterations to run.
            save_results (bool): Whether to save the results to a file.
            output_dir (str): Directory to save the results.
            dataset_name (str): Name of the dataset for saving results.
        """
        # Initialize timing statistics
        self.timer.enable = enable_timing
        self.timer.reset("run")
        self.timer.reset("diffusion")
        self.timer.reset("threshold")

        accuracies = []

        # Run the MBO iterations
        for iteration in range(iterations):
            self.make_fidelity_set()
            with self.timer.time("run"):
                acc = self.run_mbo()
            accuracies.append(acc)

        # Compute mean and standard deviation of accuracies
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"Accuracy: ({mean_acc:.4f} ± {std_acc:.4f}) %\n")

        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(
                output_dir, f"{dataset_name}_run_{timestamp}.npz"
            )

            # Compute mean times of timed sections (diffusion and thresholding)
            timing_summary = self.timer.summary()
            print(timing_summary)

            save_kwargs = {
                "dataset": dataset_name,
                "iterations": iterations,
                "accuracies": np.array(accuracies),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "timing_summary": timing_summary,
            }

            if (
                self.data is not None and self.data.shape[0] < 5000
            ):  # arbitrary limit for saving data
                save_kwargs["data"] = self.data
                save_kwargs["labels"] = self.labels
                save_kwargs["initial_cluster"] = onehot_to_labels(self.initial_cluster)

            np.savez_compressed(result_file, **save_kwargs)
            print(f"Saved results to {result_file}")

    def set_diffusion_method(self, method: str) -> None:
        """Prepares a diffusion function to apply to a given chi.
        Args:
            method (str): Method for diffusion, can be one of:
                - "spectral": spectral decomposition
                - "expm_multiply": exponential matrix multiplication
                - "A_3": taylor expansion of the exponential matrix
                - "W2": normalized weight matrix squared
                - "W_W2": fourth power of the weight matrix
                - "A_minus_eig": squared weight matrix minus fraction of identity
        """
        self.diffusion_method = method

        if method == "spectral":
            exp_vals = np.exp(-self.diffusion_time * self.eigenvalues)

            def spectral_diffuse(chi: np.ndarray) -> np.ndarray:
                return self.eigenvectors @ (
                    exp_vals[:, None] * (self.eigenvectors.T @ chi)
                )

            self.diffuse = spectral_diffuse

        elif method == "expm_multiply":

            def diffuse_heat_expm(chi: np.ndarray) -> np.ndarray:
                return sp.sparse.linalg.expm_multiply(
                    -self.diffusion_time
                    * (sp.sparse.identity(self.number_of_points) - self.W2),
                    chi,
                )

            self.diffuse = diffuse_heat_expm

        else:
            matrix = {
                "A_3": self.A_3,
                "W2": self.W2,
                "W_W2": self.W_W2,
                "A_minus_eig": self.A_minus_eig,
            }.get(method)

            if matrix is None:
                raise ValueError(f"Unknown diffusion method: {method}")

            def matrix_diffuse(chi: np.ndarray) -> np.ndarray:
                return matrix @ chi

            self.diffuse = matrix_diffuse

    def set_clustering_initialization_function(self, method: str) -> None:
        """
        Returns a clustering initialization function based on the specified method.
        Args:
            method (str): Method for clustering initialization, can be one of:
                - "bellman_ford": Bellman-Ford initialization
                - "random": Random initialization
                - "voronoi": Voronoi initialization
                - "laguerre": Laguerre initialization
                - "diffusion": Diffusion-based initialization
                - "diffusion_volume": Diffusion-based initialization with volume constraints
        """
        if method == "bellman_ford":

            def bellman_ford_init(fidelity_set: np.ndarray) -> np.ndarray:
                voronoi, label = bellman_ford_voronoi_initialization(
                    self.number_of_points, fidelity_set, self.labels, self.weight_matrix
                )
                chi = self._labels_to_onehot(label)
                return self._apply_fidelity_set(chi)

            self.cluster_initialization_function = bellman_ford_init

        elif method == "random":

            def random_init(fidelity_set: np.ndarray) -> np.ndarray:
                # Randomly assign labels to all points
                label = np.random.randint(
                    0, self.number_of_labels, size=self.number_of_points
                )
                chi = self._labels_to_onehot(label)
                return self._apply_fidelity_set(chi)

            self.cluster_initialization_function = random_init

        elif method == "voronoi":

            def voronoi_init(fidelity_set: np.ndarray) -> np.ndarray:
                # Create a sparse matrix with diagonal elements
                q_W = self.weight_matrix + sp.sparse.eye(self.number_of_points)

                # Construct a graph from the weight matrix
                q_graph = gl.graph(q_W)

                # Compute distances and labels using Dijkstra's algorithm
                distance, phase = q_graph.dijkstra(fidelity_set, return_cp=True)

                # Map the indices of closest points to their labels
                labels = self.labels[phase]

                chi = self._labels_to_onehot(labels)
                return self._apply_fidelity_set(chi)

            self.cluster_initialization_function = voronoi_init

        elif method == "laguerre":

            def laguerre_init(fidelity_set: np.ndarray) -> np.ndarray:
                # Create a sparse matrix with diagonal elements
                q_W = self.weight_matrix + sp.sparse.eye(self.number_of_points)

                # Apply the negative logarithm to the weight matrix
                q_W.data = -np.log(q_W.data)

                # Construct a graph from the weight matrix
                q_graph = gl.graph(q_W)

                # Compute distances and labels using Dijkstra's algorithm
                distances = np.zeros((self.number_of_points, self.number_of_labels))
                fidelity_labels = self.labels[fidelity_set]
                for index in range(self.number_of_labels):
                    indices = fidelity_set[fidelity_labels == index]
                    distances[:, index] = -q_graph.dijkstra(indices, return_cp=False)
                median_0 = self.number_of_labels * [1 / self.number_of_labels]
                distances = np.nan_to_num(distances, neginf=-1e4)
                median, labels, _ = fit_median_legacy(
                    self.number_of_labels, self.volume, self.volume, distances, median_0
                )
                chi = self._labels_to_onehot(labels)
                return self._apply_fidelity_set(chi)

            self.cluster_initialization_function = laguerre_init

        elif method == "diffusion":

            def diffusion_init(fidelity_set: np.ndarray) -> np.ndarray:
                delta = np.zeros((self.number_of_points, self.number_of_labels))
                delta[fidelity_set, self.labels[fidelity_set]] = self.number_of_points
                diffused = self.diffuse(delta)
                labels = onehot_to_labels(diffused)
                chi = self._labels_to_onehot(labels)
                return self._apply_fidelity_set(chi)

            self.cluster_initialization_function = diffusion_init

        elif method == "diffusion_volume":

            def diffusion_volume_init(fidelity_set: np.ndarray) -> np.ndarray:
                delta = np.zeros((self.number_of_points, self.number_of_labels))
                for i in range(len(fidelity_set)):
                    delta[fidelity_set[i], self.labels[fidelity_set[i]]] = (
                        self.number_of_points
                    )
                diffused = self.diffuse(delta)
                median_0 = self.number_of_labels * [1 / self.number_of_labels]
                median, labels, _ = fit_median_legacy(
                    self.number_of_labels, self.volume, self.volume, diffused, median_0
                )
                chi = self._labels_to_onehot(labels)
                return self._apply_fidelity_set(chi)

            self.cluster_initialization_function = diffusion_volume_init

        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def set_threshold_function(self, method: str) -> None:
        """
        Returns a threshold function based on the specified method.
        Args:
            method (str): Method for thresholding, can be "argmax", "fit_median_cpp", or "fit_median_legacy"
        """
        threshold_methods = {
            "argmax": lambda u: self._diffused_to_onehot(u),
            "fit_median_legacy": lambda u: self._labels_to_onehot(
                fit_median_legacy(
                    self.number_of_labels,
                    self.volume,
                    self.volume,
                    u,
                    np.full(self.number_of_labels, 1 / self.number_of_labels),
                )[1]
            ),
            "fit_median_cpp": lambda u: self._diffused_to_onehot(
                u - fit_median_cpp(u, self.volume, self.volume)
            ),
        }
        self.threshold_function = threshold_methods.get(method)
        if self.threshold_function is None:
            raise ValueError(f"Unknown threshold method: {method}")

    @staticmethod
    def _diffused_to_onehot(u: np.ndarray) -> np.ndarray:
        """
        Convert diffused values to one-hot encoding.
        Args:
            u (np.ndarray): Diffused values of shape (N, number_of_labels).
        Returns:
            np.ndarray: One-hot encoded matrix of shape (N, number_of_labels).
        """
        onehot = np.zeros_like(u, dtype=np.float32)
        onehot[np.arange(u.shape[0]), np.argmax(u, axis=1)] = 1.0
        return onehot

    def _apply_fidelity_set(self, chi: np.ndarray) -> np.ndarray:
        """
        Enforce fidelity constraints by overwriting chi at known indices.

        Args:
            chi (np.ndarray): One-hot encoded label matrix of shape (N, P)

        Returns:
            np.ndarray: Modified chi with known labels re-inserted
        """
        if self.fidelity_set is None or len(self.fidelity_set) == 0:
            return chi  # Nothing to apply

        indices = self.fidelity_set
        labels = self.labels[indices]

        # Efficient one-hot encoding for subset
        chi[indices] = 0.0
        chi[indices, labels] = 1.0
        return chi

    def _compute_diffusion_matrices(self) -> None:
        # Compute diagonal elements of weightmatrix
        D = np.sum(self.weight_matrix, axis=1).A1

        # Compute the normalized weight matrix D^{-1}W
        W_normalized = sp.sparse.diags(1 / D, 0).dot(self.weight_matrix)

        # Compute the matrices once and store them as member variables
        identity = sp.sparse.identity(self.number_of_points)
        self.W2 = (W_normalized.T @ W_normalized).tocsr()

        # Diffusion candidate: taylor expansion of exponential matrix
        self.A_3 = (1 / (1 + self.diffusion_time + (self.diffusion_time**2) / 2)) * (
            identity
            + self.diffusion_time * W_normalized
            + (self.diffusion_time**2) / 2 * self.W2
        ).tocsr()

        # Fourth power of weight matrix
        self.W_W2 = (self.W2 @ self.W2).tocsr()

        # Squared weight matrix minus fraction of identity
        self.A_minus_eig = (-0.1 * identity + self.W2).tocsr()

        # Diffusion candidate:spectral decomposition exponential matrix
        # First K eigenvalues and eigenvectors of Graph Laplace
        # K = int(0.5 * np.log(self.number_of_points))
        K = min(50, self.number_of_points - 2)
        print(f"Computing eigenvalues and eigenvectors for K = {K}")
        vals, vec = sp.sparse.linalg.eigs(W_normalized, k=K)
        self.eigenvectors, _ = np.linalg.qr(vec.real)  # Orthonormalize
        self.eigenvalues = 1 - vals.real
        # print("Eigenvalues:", self.eigenvalues)
        # print("Vec.T @ Vec ≈ Identity?", np.allclose(self.eigenvectors.T @ self.eigenvectors, np.eye(K), atol=1e-1))
        # print("error:", np.linalg.norm(self.eigenvectors.T @ self.eigenvectors - np.eye(K)))
        # print("exp(-t * λ):", np.exp(-self.diffusion_time * self.eigenvalues))

    def _labels_to_onehot(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert integer labels to one-hot encoding.

        Args:
            labels (np.ndarray): Array of shape (N,) containing integer labels.
            num_classes (int): Number of unique classes (phases).

        Returns:
            np.ndarray: One-hot encoded matrix of shape (N, number_of_labels).
        """
        onehot = np.zeros(
            (self.number_of_points, self.number_of_labels), dtype=np.float32
        )
        onehot[np.arange(self.number_of_points), labels] = 1.0
        return onehot

    def _fit_median_priority_queue(self, u: np.ndarray) -> np.ndarray:
        labels = VolumeMedianFitter.fit(u, self.lower_limit, self.upper_limit)
        return self._labels_to_onehot(labels)
