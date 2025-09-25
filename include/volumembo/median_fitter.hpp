#pragma once

#include <volumembo/priority_queue.hpp>
#include <volumembo/span2d.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <utility>
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
   * @param u_ A 2D span of doubles representing the data points (shape N × M)
   * @param lower_limit_ A vector of lower limits for each cluster (size M)
   * @param upper_limit_ A vector of upper limits for each cluster (size M)
   */
  VolumeMedianFitter(Span2D<const double> u_,
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
  Span2D<const double> u;                       // shape N × M
  const std::vector<unsigned int>& lower_limit; // size M
  const std::vector<unsigned int>& upper_limit; // size M

  const unsigned int N, M;    // N = number of points, M = number of clusters
  std::vector<double> median; // current median in simplex
  std::vector<Label> labels;  // length N
  std::vector<unsigned int> cluster_sizes;            // length M
  const std::vector<std::vector<double>> directions;  // size M of dim M
  const std::vector<std::vector<Label>> other_labels; // size M of size M-1

  /**
   * @brief Specifies whether the current flip logic is in growth or shrinkage
   * mode. Used in flip-chain construction to determine direction of movement
   * and which cluster (frozen vs complement) is donor or receiver.
   */
  enum class Mode
  {
    Grow,  ///< Flip logic grows the initially selected cluster (receives
           ///< point).
    Shrink ///< Flip logic shrinks the initially selected cluster (gives away
           ///< point).
  };

  /**
   * @brief FlipEvent structure to hold information about a potential flip event
   */
  struct FlipEvent
  {
    double t = std::numeric_limits<double>::infinity();
    std::vector<double> dir;
    PID pid;
    Label donor;
    Label receiver;
  };

  /**
   * @brief Structure to manage frozen hyperplanes during flip chain building
   *
   * This structure keeps track of the set of frozen hyperplanes and their
   * complement. It provides methods to generate cross pairs of labels and to
   * update the frozen set.
   */
  struct FrozenHyperplanes
  {
    Mode mode;
    unsigned int M;
    std::vector<Label> frozen;
    std::vector<Label> complement;
    std::vector<std::vector<double>> directions;
    std::vector<double> dir;
    int sign = 1;

    FrozenHyperplanes()
      : mode(Mode::Grow)
      , frozen({})
      , directions()
      , M(0)
    {
    }

    FrozenHyperplanes(Mode mode_,
                      std::vector<Label> frozen_labels,
                      const std::vector<std::vector<double>>& directions_,
                      unsigned int M_)
      : mode(mode_)
      , frozen(std::move(frozen_labels))
      , directions(directions_)
      , dir(M_)
      , M(M_)
    {
      // Initialize complement
      complement.reserve(M - frozen.size());
      for (Label i = 0; i < static_cast<Label>(M); ++i) {
        if (std::find(frozen.begin(), frozen.end(), i) == frozen.end()) {
          complement.push_back(i);
        }
      }

      // Set sign based on mode
      sign = (mode == Mode::Grow) ? 1 : -1;

      // Initialize direction
      for (Label label : frozen) {
        for (std::size_t j = 0; j < M; ++j) {
          dir[j] += static_cast<double>(sign) * directions[label][j];
        }
      }
    }

    /**
     * @brief Insert new label into frozen set and update complement
     *
     * @param label The label to be added to the frozen set
     */
    void freeze(Label label)
    {
      frozen.push_back(label);
      // Remove from complement
      auto it = std::find(complement.begin(), complement.end(), label);
      if (it != complement.end()) {
        complement.erase(it);
      }

      // Update direction
      for (std::size_t j = 0; j < M; ++j) {
        dir[j] += static_cast<double>(sign) * directions[label][j];
      }
    }

    /**
     * @brief Generate a vector of pairs of labels from a given subset of frozen
     * hyperplanes
     *
     * @return A vector of pairs of labels representing all unique pairs formed
     * by
     */
    std::vector<std::pair<Label, Label>> generate_cross_pairs() const
    {
      const std::vector<Label>& from =
        (mode == Mode::Grow) ? complement : frozen;
      const std::vector<Label>& to = (mode == Mode::Grow) ? frozen : complement;

      std::vector<std::pair<Label, Label>> result;
      for (Label f : from) {
        for (Label t : to) {
          result.emplace_back(f, t);
        }
      }
      return result;
    }

    /**
     * @brief Get the current direction vector
     *
     * @return A vector of doubles representing the direction
     */
    std::vector<double> get_direction() const { return dir; }
  };

  /**
   * @brief Structure to represent a chain of flip events
   *
   * This structure holds a sequence of flip events that represent a series of
   * flips to be applied to the clusters.
   */
  struct FlipChain
  {
    std::vector<FlipEvent> chain;
    FrozenHyperplanes frozen_hyperplanes;
    Mode mode;
    std::vector<double> median;
    const unsigned int M;

    /**
     * @brief Constructor for FlipChain without frozen hyperplanes
     */
    FlipChain(const FlipEvent& event,
              const std::vector<double>& median_,
              Mode mode_,
              unsigned int M_)
      : chain{ event }
      , median(median_)
      , mode(mode_)
      , M(M_)
    {
      // Update median
      for (unsigned int i = 0; i < M; ++i) {
        median[i] += event.t * event.dir[i];
      }
    }

    /**
     * @brief Constructor for FlipChain with frozen hyperplanes
     */
    FlipChain(const FlipEvent& event,
              const std::vector<double>& median_,
              Mode mode_,
              Label from,
              Label to,
              const std::vector<std::vector<double>>& directions,
              unsigned int M_)
      : chain{ event }
      , median(median_)
      , frozen_hyperplanes(mode_,
                           std::vector<Label>{ from, to },
                           directions,
                           M_)
      , mode(mode_)
      , M(M_)
    {
      update_median(event);
    }

    /**
     * @brief Add a flip event to the chain and update the median
     */
    void add_event(const FlipEvent& event)
    {
      chain.push_back(event);
      update_median(event);
    }

    /**
     * @brief Freeze a label in the frozen hyperplanes set
     */
    void freeze(Label label) { frozen_hyperplanes.freeze(label); }

    /**
     * @brief Generate a vector of pairs of labels from the frozen hyperplanes
     *
     * @return A vector of pairs of labels representing all unique pairs formed
     * by
     */
    std::vector<std::pair<Label, Label>> generate_cross_pairs() const
    {
      return frozen_hyperplanes.generate_cross_pairs();
    }

    /**
     * @brief Get the current direction vector from the frozen hyperplanes
     *
     * @return A vector of doubles representing the direction
     */
    std::vector<double> get_direction() const
    {
      return frozen_hyperplanes.get_direction();
    }

    /**
     * @brief Get the current median
     *
     * @return A vector of doubles representing the current median
     */
    std::vector<double> get_median() const { return median; }

    /**
     * @brief Set frozen hyperplanes for the flip chain
     *
     * @param frozen A vector of labels representing the frozen hyperplanes
     * @param directions A vector of direction vectors for each cluster
     * @param M The number of clusters
     */
    void set_frozen_hyperplanes(
      std::vector<Label> frozen,
      const std::vector<std::vector<double>>& directions)
    {
      frozen_hyperplanes =
        FrozenHyperplanes(mode, std::move(frozen), directions, M);
    }

    /**
     * @brief Update the median based on a flip event
     *
     * @param event The flip event used to update the median
     */
    void update_median(const FlipEvent& event)
    {
      for (unsigned int i = 0; i < M; ++i) {
        median[i] += event.t * event.dir[i];
      }
    }
  };

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
   * @brief Apply a sequence of flip events to update the median and clusters
   *
   * @param flip_chain A vector of flip events representing the sequence of
   * flips to apply
   */
  void apply_flip_chain(const FlipChain& flip_chain);

  /**
   * @brief Assign clusters based on the current median
   *
   * This method updates the labels of each point based on the current median.
   * It computes the difference between each point and the median, and assigns
   * the point to the cluster whose median is closest to the point.
   */
  void assign_clusters();

  /**
   * @brief Build a flip chain starting from a given direction
   *
   * @param flip_chain The current flip chain to which new events will be added
   * @param frozen_hyperplanes A set of labels representing the frozen
   * hyperplanes
   *
   * @return true if a valid flip chain was built, false otherwise
   */
  bool build_flip_chain(FlipChain& flip_chain, unsigned int recursion_level);

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
  std::vector<double> compute_u_minus_m(std::size_t index) const;

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
   * @brief Check if the mismatch in cluster sizes is reduced by the flip chain
   *
   * @param flip_chain The flip chain to evaluate
   *
   * @return true if the mismatch is reduced, false otherwise
   */
  bool mismatch_reduced(const FlipChain& flip_chain) const;

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
   * @brief Check if the volumes of all clusters match their limits
   *
   * @return true if all clusters' sizes are within their limits, false
   * otherwise
   */
  bool volumes_matched() const;
};

} // namespace volumembo
