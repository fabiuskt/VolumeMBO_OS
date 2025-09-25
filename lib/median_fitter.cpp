#include <volumembo/median_fitter.hpp>

#include <volumembo/priority_queue.hpp>
#include <volumembo/span2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
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
  , M(static_cast<unsigned int>(u.cols()))
  , median(M, 1.0 / static_cast<double>(M))
  , labels(N)
  , cluster_sizes(M, 0)
  , directions(precompute_directions(M))
  , other_labels(precompute_other_labels(M))
{
  // Initialize labels and cluster sizes
  assign_clusters();

  // Initialize priority queues
  initialize_priority_queues();
}

void
VolumeMedianFitter::apply_flip_chain(const FlipChain& flip_chain)
{
  for (const FlipEvent& event : flip_chain.chain) {

    // Update priority queues
    remove(event.pid, event.donor);
    insert_into_queues(event.pid, event.receiver);

    // Update cluster labels and sizes
    labels[event.pid] = event.receiver;
    cluster_sizes[event.donor]--;
    cluster_sizes[event.receiver]++;

    // Update median
    median = flip_chain.get_median();
  }
}

void
VolumeMedianFitter::assign_clusters()
{
  cluster_sizes.assign(M, 0);

  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> u_minus_m = compute_u_minus_m(i);
    for (unsigned int j = 0; j < M; ++j) {
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
VolumeMedianFitter::build_flip_chain(FlipChain& flip_chain,
                                     unsigned int recursion_level)
{
  // Limit recursion depth to avoid infinite loops
  if (recursion_level >
      M - 3) { // flip chain already has one entry when entering this function
    printf("Recursion limit reached\n");
    fflush(stdout);
    return false;
  }

  // Get pairs of possible flip labels
  std::vector<std::pair<Label, Label>> label_pairs =
    flip_chain.generate_cross_pairs();

  // Loop through possible pairs of flip labels
  FlipEvent best;
  for (const auto& [donor, receiver] : label_pairs) {
    printf("Checking pair (%u, %u)\n", donor, receiver);
    fflush(stdout);
    // throw std::runtime_error(
    //   "INVALID: The priority queues have not been updated wiile building the
    //   " "lip chain and the flip time computation does not take the
    //   intermediate " "median values into account. Major bug.");
    if (auto pid_opt = peek(donor, receiver)) {
      double t =
        compute_flip_time(*pid_opt, donor, receiver, flip_chain.get_median());
      if (t < best.t && std::isfinite(t)) {
        best.t = t;
        best.dir = flip_chain.get_direction();
        best.pid = *pid_opt;
        best.donor = donor;
        best.receiver = receiver;
      }
    }
  }
  printf("Best: (t = %g, PID = %d, %d → %d)\n",
         best.t,
         best.pid,
         best.donor,
         best.receiver);
  fflush(stdout);

  flip_chain.add_event(best);

  if (mismatch_reduced(flip_chain)) {
    return true;
  } else {
    flip_chain.mode == Mode::Grow
      ? flip_chain.freeze(best.donor)
      : flip_chain.freeze(best.receiver); // Why donor?
    return build_flip_chain(flip_chain, recursion_level + 1);
  }
}

double
VolumeMedianFitter::compute_flip_time(PID pid,
                                      Label from_label,
                                      Label to_label) const
{
  std::vector<double> u_minus_m = compute_u_minus_m(pid);

  return (u_minus_m[from_label] - u_minus_m[to_label]) *
         static_cast<double>(M - 1) / static_cast<double>(M);
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
         static_cast<double>(M - 1) / static_cast<double>(M);
}

std::vector<double>
VolumeMedianFitter::compute_u_minus_m(std::size_t index) const
{
  std::vector<double> u_minus_m(M);
  for (unsigned int j = 0; j < M; ++j) {
    u_minus_m[j] = u(index, j) - median[j];
  }
  return u_minus_m;
}

std::vector<double>
VolumeMedianFitter::compute_u_minus_m(
  std::size_t index,
  const std::vector<double>& median_override) const
{
  std::vector<double> u_minus_m(M);
  for (unsigned int j = 0; j < M; ++j) {
    u_minus_m[j] = u(index, j) - median_override[j];
  }
  return u_minus_m;
}

std::vector<double>
VolumeMedianFitter::fit()
{
  printf("fit()\n");
  fflush(stdout);
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

    for (Label i = 0; i < M; ++i) {
      printf("\t Cluster %d:\n", i);
      fflush(stdout);
      // Determine if cluster i needs to grow or shrink
      Mode mode;

      unsigned int size_i = cluster_sizes[i];
      unsigned int lower_i = lower_limit[i];
      unsigned int upper_i = upper_limit[i];
      for (std::size_t k = 0; k < M; ++k) {
        printf("Cluster sizes[%zu] = %u (limits: %u - %u)\n",
               k,
               cluster_sizes[k],
               lower_limit[k],
               upper_limit[k]);
        fflush(stdout);
      }
      if (size_i < lower_i) { // i receives a point
        printf("Mode: Grow\n");
        fflush(stdout);
        mode = Mode::Grow;
      } else if (size_i > upper_i) { // i donates a point
        printf("Mode: Shrink\n");
        fflush(stdout);
        mode = Mode::Shrink;
      } else {
        printf("%d already within targets.\n", i);
        fflush(stdout);
        continue; // Cluster size is within limits, skip to next cluster
      }

      FlipEvent best;
      FrozenHyperplanes frozen_hyperplanes(
        mode,
        { i },
        directions,
        M); // Careful, here this acts only as helper for generating the pairs
            // and direction

      // Get pairs of possible flip labels
      std::vector<std::pair<Label, Label>> label_pairs =
        frozen_hyperplanes.generate_cross_pairs();

      for (const auto& [donor, receiver] : label_pairs) {
        printf("Checking pair (%u, %u)\n", donor, receiver);
        fflush(stdout);
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
      printf("Best: (t = %g, PID = %d, %d → %d)\n",
             best.t,
             best.pid,
             best.donor,
             best.receiver);
      fflush(stdout);

      // Initialize flip chain
      FlipChain flip_chain = { best, median, mode, M };
      unsigned int size_donor = cluster_sizes[best.donor];
      unsigned int lower_donor = lower_limit[best.donor];
      unsigned int size_receiver = cluster_sizes[best.receiver];
      unsigned int upper_receiver = upper_limit[best.receiver];

      if ((mode == Mode::Grow && size_donor > lower_donor) ||
          (mode == Mode::Shrink && size_receiver < upper_receiver)) {

        apply_flip_chain(flip_chain);
        flip_performed = true;
        break;
      } else if (M > 2) {
        printf("Build flip chain:\n");
        fflush(stdout);
        flip_chain.set_frozen_hyperplanes({ best.donor, best.receiver },
                                          directions);

        bool flip_chain_build = build_flip_chain(flip_chain, 0);
        if (flip_chain_build) {
          apply_flip_chain(flip_chain);
          flip_performed = true;
          break;
        }
      }

      if (i == M - 1 && !flip_performed) {
        print_flip_chain(flip_chain);
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
  printf("\n");
  fflush(stdout);
  return median;
}

void
VolumeMedianFitter::initialize_priority_queues()
{
  // Initialize priority queues for each pair of labels
  for (Label i = 0; i < M; ++i) {
    for (Label j = 0; j < M; ++j) {
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

bool
VolumeMedianFitter::mismatch_reduced(const FlipChain& flip_chain) const
{
  auto mismatch = [&](const std::vector<unsigned int>& sizes) -> unsigned int {
    unsigned int total = 0;
    for (std::size_t i = 0; i < M; ++i) {
      if (sizes[i] < lower_limit[i]) {
        total += lower_limit[i] - sizes[i];
      } else if (sizes[i] > upper_limit[i]) {
        total += sizes[i] - upper_limit[i];
      }
    }
    return total;
  };

  // Compute current mismatch
  auto current_sizes = cluster_sizes;
  unsigned int mismatch_before = mismatch(current_sizes);

  // Apply flip chain hypothetically
  for (const auto& ev : flip_chain.chain) {
    current_sizes[ev.donor]--;
    current_sizes[ev.receiver]++;
  }

  // Compute mismatch after
  unsigned int mismatch_after = mismatch(current_sizes);

  // Return true if mismatch is reduced
  return mismatch_after < mismatch_before;
}

void
VolumeMedianFitter::insert_into_queues(PID pid, Label from_label)
{
  for (Label to_label : other_labels[from_label]) {
    priority_queues.at({ from_label, to_label }).push(pid);
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
VolumeMedianFitter::precompute_directions(unsigned int M)
{
  std::vector<std::vector<double>> directions;
  directions.reserve(M);

  for (unsigned int i = 0; i < M; ++i) {
    std::vector<double> d(M, 1.0 / static_cast<double>(M - 1));
    d[i] = -1.0;
    directions.push_back(std::move(d));
  }

  return directions;
}

std::vector<std::vector<Label>>
VolumeMedianFitter::precompute_other_labels(unsigned int M)
{
  std::vector<std::vector<Label>> other_labels;
  other_labels.resize(M);

  for (Label i = 0; i < M; ++i) {
    other_labels[i].reserve(M - 1);
    for (Label j = 0; j < M; ++j) {
      if (j != i) {
        other_labels[i].push_back(j);
      }
    }
  }

  return other_labels;
}

void
VolumeMedianFitter::print_flip_chain(const FlipChain& flip_chain)
{
  for (const FlipEvent& event : flip_chain.chain) {
    std::cout << "  PID " << event.pid << ": " << event.donor << " → "
              << event.receiver << ", t = " << event.t << ", dir = [";
    for (std::size_t i = 0; i < event.dir.size(); ++i) {
      std::cout << event.dir[i];
      if (i < event.dir.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::flush;
  }
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
  for (Label i = 0; i < M; ++i) {
    // Check if the cluster size is within the limits
    if (cluster_sizes[i] < lower_limit[i] || cluster_sizes[i] > upper_limit[i])
      return false;
  }
  return true;
}
}
