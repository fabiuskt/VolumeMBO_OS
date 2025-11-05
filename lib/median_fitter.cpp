#include <volumembo/median_fitter.hpp>

#include <volumembo/priority_queue.hpp>
#include <volumembo/span2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace volumembo {

FlipTimeComparator::FlipTimeComparator(const VolumeMedianFitter* parent,
                                       Label from,
                                       Label to)
  : parent(parent)
  , from_label(from)
  , to_label(to)
{
}

bool
FlipTimeComparator::operator()(int a, int b) const
{
  return parent->compute_flip_time(a, from_label, to_label) >
         parent->compute_flip_time(b, from_label, to_label);
}

bool
FlipTimeComparator::operator()(int a,
                               int b,
                               const std::vector<double>& median_override) const
{
  return parent->compute_flip_time(a, from_label, to_label, median_override) >
         parent->compute_flip_time(b, from_label, to_label, median_override);
}

VolumeMedianFitter::VolumeMedianFitter(
  Span2D<const double> u_,
  const std::vector<unsigned int>& lower_limit_,
  const std::vector<unsigned int>& upper_limit_)
  : u(u_)
  , lower_limit(lower_limit_)
  , upper_limit(upper_limit_)
  , N(static_cast<unsigned int>(u.rows()))
  , P(static_cast<unsigned int>(u.cols()))
  , median(P, 1.0 / static_cast<double>(P))
  , labels(N)
  , cluster_sizes(P, 0)
  , directions(precompute_directions(P))
  , other_labels(precompute_other_labels(P))
{
  // Initialize labels and cluster sizes
  assign_clusters();

  // Initialize priority queues
  initialize_priority_queues();
}

void
VolumeMedianFitter::apply_flip_event(const FlipEvent& event)
{
  // Update priority queues
  remove(event.pid, event.donor);
  insert_into_queues(event.pid, event.receiver);

  // Update cluster labels and sizes
  labels[event.pid] = event.receiver;
  cluster_sizes[event.donor]--;
  cluster_sizes[event.receiver]++;
}

void
VolumeMedianFitter::apply_flip_tree(const FlipTree& flip_tree)
{
  for (const std::size_t& index : flip_tree.valid_flips) {
    apply_flip_event(flip_tree.flips[index]);

    // Update median
    median = flip_tree.get_median();
  }
}

void
VolumeMedianFitter::assign_clusters()
{
  cluster_sizes.assign(P, 0);

  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> u_minus_m = compute_u_minus_m(i);
    for (unsigned int j = 0; j < P; ++j) {
      u_minus_m[j] -= median[j];
    }

    auto max_it = std::max_element(u_minus_m.begin(), u_minus_m.end());
    Label max_index =
      static_cast<Label>(std::distance(u_minus_m.begin(), max_it));

    labels[i] = max_index;
    ++cluster_sizes[max_index];
  }
}

bool
VolumeMedianFitter::build_flip_tree(FlipTree& flip_tree,
                                    unsigned int recursion_level)
{
  // Limit recursion depth to avoid infinite loops
  if (recursion_level >
      P - 3) { // flip tree already has one entry when entering this function
    return false;
  }

  // Get pairs of possible flip labels
  std::vector<std::pair<Label, Label>> label_pairs =
    flip_tree.generate_cross_pairs();

  // Loop through possible pairs of flip labels
  FlipEvent best;
  for (const auto& [donor, receiver] : label_pairs) {

    if (auto pid_opt = peek(donor, receiver)) {
      double t =
        compute_flip_time(*pid_opt, donor, receiver, flip_tree.get_median());
      if (t < best.t && std::isfinite(t)) {
        best.t = t;
        best.dir = flip_tree.get_direction();
        best.pid = *pid_opt;
        best.donor = donor;
        best.receiver = receiver;
      }
    }
  }

  flip_tree.add_event(best,
                      flip_tree.mode == Mode::Shrink
                        ? can_receive(best.receiver)
                        : can_donate(best.donor));

  if (path_found(flip_tree)) {
    return true;
  } else {
    flip_tree.mode == Mode::Grow ? flip_tree.freeze(best.donor)
                                 : flip_tree.freeze(best.receiver);
    return build_flip_tree(flip_tree, recursion_level + 1);
  }
}

double
VolumeMedianFitter::compute_flip_time(PID pid,
                                      Label from_label,
                                      Label to_label) const
{
  std::vector<double> u_minus_m = compute_u_minus_m(pid);

  return (u_minus_m[from_label] - u_minus_m[to_label]) *
         static_cast<double>(P - 1) / static_cast<double>(P);
}

double
VolumeMedianFitter::compute_flip_time(
  PID pid,
  Label from_label,
  Label to_label,
  const std::vector<double>& median_override) const
{
  std::vector<double> u_minus_m = compute_u_minus_m(pid, median_override);

  return (u_minus_m[from_label] - u_minus_m[to_label]) *
         static_cast<double>(P - 1) / static_cast<double>(P);
}

std::vector<double>
VolumeMedianFitter::compute_u_minus_m(std::size_t index) const
{
  std::vector<double> u_minus_m(P);
  for (unsigned int j = 0; j < P; ++j) {
    u_minus_m[j] = u(index, j) - median[j];
  }
  return u_minus_m;
}

std::vector<double>
VolumeMedianFitter::compute_u_minus_m(
  std::size_t index,
  const std::vector<double>& median_override) const
{
  std::vector<double> u_minus_m(P);
  for (unsigned int j = 0; j < P; ++j) {
    u_minus_m[j] = u(index, j) - median_override[j];
  }
  return u_minus_m;
}

std::vector<double>
VolumeMedianFitter::fit()
{
  const unsigned int max_iter = N;
  unsigned int iteration = 0;

  while (!volumes_matched()) {
    // Exit if maximum iterations reached
    if (iteration > max_iter) {
      throw std::runtime_error("fit() failed: max iteration reached " +
                               std::to_string(iteration));
    }

    // Flag to indicate if a flip was performed in this iteration
    bool flip_performed = false;

    for (Label i = 0; i < P; ++i) {

      // Determine if cluster i needs to grow or shrink
      Mode mode;

      unsigned int size_i = cluster_sizes[i];
      unsigned int lower_i = lower_limit[i];
      unsigned int upper_i = upper_limit[i];

      if (size_i < lower_i) { // i receives a point
        mode = Mode::Grow;
      } else if (size_i > upper_i) { // i donates a point
        mode = Mode::Shrink;
      } else {
        continue; // Cluster size is within limits, skip to next cluster
      }

      FlipEvent best;
      FrozenHyperplanes frozen_hyperplanes(
        mode,
        { i },
        directions,
        P); // Careful, here this acts only as helper for generating the pairs
            // and direction

      // Get pairs of possible flip labels
      std::vector<std::pair<Label, Label>> label_pairs =
        frozen_hyperplanes.generate_cross_pairs();

      for (const auto& [donor, receiver] : label_pairs) {
        if (auto pid_opt = peek(donor, receiver)) {
          double t = compute_flip_time(*pid_opt, donor, receiver);
          if (t < best.t && std::isfinite(t)) {
            best.t = t;
            best.dir = frozen_hyperplanes.get_direction();
            best.pid = *pid_opt;
            best.donor = donor;
            best.receiver = receiver;
          }
        }
      }

      // Initialize flip tree
      FlipTree flip_tree = { best, median, mode, P };

      unsigned int size_donor = cluster_sizes[best.donor];
      unsigned int lower_donor = lower_limit[best.donor];
      unsigned int size_receiver = cluster_sizes[best.receiver];
      unsigned int upper_receiver = upper_limit[best.receiver];

      if ((mode == Mode::Grow && size_donor > lower_donor) ||
          (mode == Mode::Shrink && size_receiver < upper_receiver)) {

        apply_flip_event(best);
        median = flip_tree.get_median();
        flip_performed = true;
        break;
      } else if (P > 2) {

        flip_tree.set_frozen_hyperplanes({ best.donor, best.receiver },
                                         directions);

        bool flip_tree_build = build_flip_tree(flip_tree, 0);
        if (flip_tree_build) {
          apply_flip_tree(flip_tree);
          flip_performed = true;
          break;
        }
      }
    }
    if (!flip_performed) {
      // If no flip was performed in this iteration, we are stuck
      throw std::runtime_error(
        "fit() failed: no valid flip found at iteration " +
        std::to_string(iteration));
    }

    ++iteration;
  }

  return median;
}

void
VolumeMedianFitter::initialize_priority_queues()
{
  // Initialize priority queues for each pair of labels
  for (Label i = 0; i < P; ++i) {
    for (Label j = 0; j < P; ++j) {
      if (i == j)
        continue;
      auto key = std::make_pair(i, j);
      FlipTimeComparator cmp(this, i, j);
      priority_queues.emplace(
        key, ModifiablePriorityQueue<PID, FlipTimeComparator>(std::move(cmp)));
    }
  }

  // Populate the priority queues with point IDs based on their labels
  for (PID pid = 0; pid < N; ++pid) {
    Label from_label = labels[pid];
    for (Label to_label : other_labels[from_label]) {
      priority_queues.at({ from_label, to_label }).push(pid);
    }
  }
}

void
VolumeMedianFitter::insert_into_queues(PID pid, Label from_label)
{
  for (Label to_label : other_labels[from_label]) {
    priority_queues.at({ from_label, to_label }).push(pid);
  }
}

bool
VolumeMedianFitter::path_found(FlipTree& flip_tree) const
{
  auto& flips = flip_tree.flips;
  auto& path = flip_tree.valid_flips;

  if (path.empty()) {
    return false;
  }

  Label current_cluster = (flip_tree.mode == Mode::Shrink)
                            ? flips[path.back()].donor
                            : flips[path.back()].receiver;

  // Check if current cluster is already the root
  if ((flip_tree.mode == Mode::Shrink && current_cluster == flips[0].donor) ||
      (flip_tree.mode == Mode::Grow && current_cluster == flips[0].receiver)) {
    return true;
  }

  // Keep walking up the tree
  while (true) {
    auto it =
      std::find_if(flips.begin(), flips.end(), [&](const FlipEvent& ev) {
        return (flip_tree.mode == Mode::Shrink) ? ev.receiver == current_cluster
                                                : ev.donor == current_cluster;
      });

    if (it == flips.end()) {
      return false; // Root not yet reached, keep building tree
    }

    std::size_t index = std::distance(flips.begin(), it);

    // Avoid duplicates
    if (std::find(path.begin(), path.end(), index) != path.end()) {
      return false; // Already added
    }

    path.push_back(index);

    current_cluster =
      (flip_tree.mode == Mode::Shrink) ? it->donor : it->receiver;

    // Check if we reached a root
    if ((flip_tree.mode == Mode::Shrink && current_cluster == flips[0].donor) ||
        (flip_tree.mode == Mode::Grow &&
         current_cluster == flips[0].receiver)) {
      return true;
    }
  }
}

std::optional<PID>
VolumeMedianFitter::peek(Label from_label, Label to_label)
{
  auto& pq = priority_queues.at({ from_label, to_label });
  if (pq.empty())
    return std::nullopt;
  return pq.peek();
}

std::vector<std::vector<double>>
VolumeMedianFitter::precompute_directions(unsigned int P)
{
  std::vector<std::vector<double>> directions;
  directions.reserve(P);

  for (unsigned int i = 0; i < P; ++i) {
    std::vector<double> d(P, 1.0 / static_cast<double>(P - 1));
    d[i] = -1.0;
    directions.push_back(std::move(d));
  }

  return directions;
}

std::vector<std::vector<Label>>
VolumeMedianFitter::precompute_other_labels(unsigned int P)
{
  std::vector<std::vector<Label>> other_labels;
  other_labels.resize(P);

  for (Label i = 0; i < P; ++i) {
    other_labels[i].reserve(P - 1);
    for (Label j = 0; j < P; ++j) {
      if (j != i) {
        other_labels[i].push_back(j);
      }
    }
  }

  return other_labels;
}

void
VolumeMedianFitter::remove(PID pid, Label from_label)
{
  for (Label to_label : other_labels[from_label]) {
    priority_queues.at({ from_label, to_label }).remove(pid);
  }
}

bool
VolumeMedianFitter::volumes_matched() const
{
  for (Label i = 0; i < P; ++i) {
    // Check if the cluster size is within the limits
    if (cluster_sizes[i] < lower_limit[i] || cluster_sizes[i] > upper_limit[i])
      return false;
  }
  return true;
}
}
