#pragma once

#include "volumembo/priority_queue.hpp"

#include <map>
#include <optional>
#include <vector>

namespace volumembo {

using PID = unsigned int;   // point ID type
using Label = unsigned int; // cluster label type

class VolumeMedianFitter;

struct FlipTimeComparator
{
  const VolumeMedianFitter*
    parent; // Pointer to the parent VolumeMedianFitter instance
  const Label
    from_label; // Label of the cluster from which the flip is considered
  const Label to_label; // Label of the cluster to which the flip is considered

  /**
   * @brief Constructor for FlipTimeComparator
   *
   * @param parent Pointer to the VolumeMedianFitter instance
   * @param from Label of the cluster from which the flip is considered
   * @param to Label of the cluster to which the flip is considered
   */
  FlipTimeComparator(const VolumeMedianFitter* parent, Label from, Label to);

  /**
   * @brief Compare two point IDs based on their flip time
   *
   * @param a First point ID
   * @param b Second point ID
   * @return true if flip time of a is greater than that of b, false otherwise
   */
  bool operator()(int a, int b) const;
};

class VolumeMedianFitter
{
public:
  /**
   * @brief Constructor for VolumeMedianFitter
   *
   * @param u_ A 2D vector representing the data points (shape N × M)
   * @param lower_limit_ A vector of lower limits for each cluster (size M)
   * @param upper_limit_ A vector of upper limits for each cluster (size M)
   */
  VolumeMedianFitter(const std::vector<std::vector<double>>& u_,
                     const std::vector<unsigned int>& lower_limit_,
                     const std::vector<unsigned int>& upper_limit_);

  /**
   * @brief Fit the median using the priority queue approach
   *
   * @return A vector of doubles representing the fitted median
   */
  std::vector<double> fit();

  friend struct FlipTimeComparator;

private:
  const std::vector<std::vector<double>>& u;    // shape N × M
  const std::vector<unsigned int>& lower_limit; // size M
  const std::vector<unsigned int>& upper_limit; // size M

  const unsigned int N, M;    // N = number of points, M = number of clusters
  std::vector<double> median; // current median in simplex
  std::vector<Label> labels;  // length N
  std::vector<unsigned int> cluster_sizes;            // length M
  const std::vector<std::vector<double>> directions;  // size M of dim M
  const std::vector<std::vector<Label>> other_labels; // size M of size M-1

  /**
   * @brief Priority queues for managing point IDs based on flip times
   *
   * The priority queues are organized by pairs of labels (from_label,
   * to_label). Each queue contains point IDs that can potentially flip from one
   * label to another.
   */
  std::map<std::pair<Label, Label>,
           ModifiablePriorityQueue<PID, FlipTimeComparator>>
    priority_queues;

  /**
   * @brief Assign clusters based on the current median
   *
   * This method updates the labels of each point based on the current median.
   * It computes the difference between each point and the median, and assigns
   * the point to the cluster whose median is closest to the point.
   */
  void assign_clusters();

  /**
   * @brief Compute the flip time for a point ID between two labels
   *
   * @param pid The point ID for which the flip time is computed
   * @param from_label The label from which the flip is considered
   * @param to_label The label to which the flip is considered
   *
   * @return The computed flip time as a double
   */
  double compute_flip_time(PID pid, Label from_label, Label to_label) const;

  /**
   * @brief Compute the difference between the data point and the current median
   *
   * @param index The index of the data point
   *
   * @return A vector of doubles representing the difference between the data
   * point and the current median
   */
  std::vector<double> compute_u_minus_m(size_t index) const;

  /**
   * @brief Initialize the priority queues for all label pairs
   *
   * This method sets up the priority queues for each pair of labels
   * (from_label, to_label) and populates them with point IDs based on their
   * current labels.
   */
  void initialize_priority_queues();

  /**
   * @brief Insert a point ID into the priority queues for all label pairs
   *
   * @param pid The point ID to insert
   * @param from_label The label from which the flip is considered
   */
  void insert_into_queues(PID pid, Label from_label);

  /**
   * @brief Peek at the top element of the priority queue for a given label pair
   *
   * @param from_label The label from which the flip is considered
   * @param to_label The label to which the flip is considered
   *
   * @return An optional containing the point ID if the queue is not empty,
   * otherwise std::nullopt
   */
  std::optional<PID> peek(Label from_label, Label to_label);

  /**
   * @brief Precompute the directions for each cluster
   *
   * @param M The number of clusters
   *
   * @return A vector of vectors, where each inner vector contains the direction
   * for a cluster
   */
  static std::vector<std::vector<double>> precompute_directions(unsigned int M);

  /**
   * @brief Precompute the other labels for each cluster (0: 1-M-1, 1: 0, 2-M-1,
  2: 0-1, 3-M-1, etc.)
   *
   * @param M The number of clusters
   *
   * @return A vector of vectors, where each inner vector contains the labels
  */
  static std::vector<std::vector<Label>> precompute_other_labels(
    unsigned int M);

  /**
   * @brief Remove a point ID from all priority queues associated with its label
   *
   * @param pid The point ID to remove
   * @param from_label The label from which the point ID is being removed
   */
  void remove(PID pid, Label from_label);

  /**
   * @brief Select a label to modify based on the current cluster sizes and
   * limits
   *
   * @param offset Offset to adjust the selection process
   *
   * @return The label of the selected cluster
   */
  Label select_label(int offset) const;

  /**
   * @brief Check if the volumes of all clusters match their limits
   *
   * @return true if all clusters' sizes are within their limits, false
   * otherwise
   */
  bool volumes_matched() const;
};

} // namespace volumembo
