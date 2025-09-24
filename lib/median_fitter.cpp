#include <volumembo/median_fitter.hpp>

#include <volumembo/priority_queue.hpp>
#include <volumembo/span2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <sstream>

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
VolumeMedianFitter::apply_flip_chain(const std::vector<Event>& chain)
{
  for (const Event& event : chain) {
    // --- Update median ---
    for (unsigned int j = 0; j < M; ++j) {
      median[j] += event.t * event.dir[j];
    }

    // --- Update priority queues ---
    remove(event.pid, event.from);
    insert_into_queues(event.pid, event.to);

    // --- Update cluster labels and sizes ---
    labels[event.pid] = event.to;
    cluster_sizes[event.from]--;
    cluster_sizes[event.to]++;

    std::cout << "  PID " << event.pid << ": " << event.from << " → "
              << event.to << ", t = " << event.t << ", dir = [";
    for (std::size_t i = 0; i < event.dir.size(); ++i) {
      std::cout << event.dir[i];
      if (i < event.dir.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]\n";
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
VolumeMedianFitter::build_flip_chain(std::vector<Event>& flip_chain,
                                     FrozenHyperplanes& frozen_hyperplanes)
{
  // Get pairs of possible flip labels
  std::vector<std::pair<Label, Label>> label_pairs =
    frozen_hyperplanes.generate_cross_pairs();

  // Loop through possible pairs of flip labels
  Event best;
  for (const auto& [donor, receiver] : label_pairs) {
    printf("Checking pair (%u, %u)\n", donor, receiver);
    if (auto pid_opt = peek(donor, receiver)) {
      double t = compute_flip_time(*pid_opt, donor, receiver);
      if (t < best.t && std::isfinite(t)) {
        best.t = t;
        best.dir = frozen_hyperplanes.get_direction();
        best.pid = *pid_opt;
        best.from = donor;
        best.to = receiver;
      }
    }
  }
  flip_chain.push_back(best);

  if (mismatch_reduced(flip_chain)) {
    printf("Mismatched reduced.\n");
    return true;
  } else {
    printf("Mismatched not reduced.\n");
    frozen_hyperplanes.freeze(best.from);
    return build_flip_chain(flip_chain, frozen_hyperplanes);
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
VolumeMedianFitter::fit()
{
  constexpr unsigned int max_iter = 100000;
  unsigned int iteration = 0;

  while (!volumes_matched()) {
    // Exit if maximum iterations reached
    if (iteration > max_iter)
      break;

    for (Label i = 0; i < M; ++i) {
      // Start with simple direction
      std::vector<double> dir = directions[i];

      unsigned int size_i = cluster_sizes[i];
      unsigned int lower_i = lower_limit[i];
      unsigned int upper_i = upper_limit[i];

      // --- Grow direction (+i) ---
      if (size_i < lower_i) { // i receives a point
        Mode mode = Mode::Grow;
        Event best;
        for (Label donor : other_labels[i]) {
          if (auto pid_opt = peek(donor, i)) {
            double t = compute_flip_time(*pid_opt, donor, i);
            if (t < best.t) {
              best.t = t;
              best.dir = dir;
              best.pid = *pid_opt;
              best.from = donor;
              best.to = i;
            }
          }
        }
        unsigned int size_donor = cluster_sizes[best.from];
        unsigned int lower_donor = lower_limit[best.from];

        if (size_donor > lower_donor) {

          // Update median
          for (unsigned int j = 0; j < M; ++j) {
            median[j] += best.t * best.dir[j];
          }

          // Update queues
          remove(best.pid, best.from);
          insert_into_queues(best.pid, best.to);

          // Update clusters
          labels[best.pid] = best.to;
          cluster_sizes[best.from]--;
          cluster_sizes[best.to]++;
          break;
        } else if (M > 2) {

          // Initialize flip chain
          std::vector<Event> flip_chain = { best };

          // Define initial set of frozen hyperplanes
          FrozenHyperplanes frozen_hyperplanes(
            mode, { best.from, best.to }, directions, M);

          bool flip_chain_build =
            build_flip_chain(flip_chain, frozen_hyperplanes);
          if (flip_chain_build) {
            printf("test 1\n");

            apply_flip_chain(flip_chain);
            break;
          } else {
            throw std::runtime_error(
              "fit() failed: no valid flip or chain found at iteration " +
              std::to_string(iteration));
          }
        }
      }
      /*
       // --- Shrink direction (-i) ---
       if (size_i > upper_i) { // i gives a point
         Mode mode = Mode::Shrink;
         std::vector<double> dir_shrink = dir;
         for (unsigned int j = 0; j < M; ++j) {
           dir_shrink[j] = -dir[j];
         }
         Event best;
         for (Label receiver : other_labels[i]) {
           if (auto pid_opt = peek(i, receiver)) {
             double t = compute_flip_time(*pid_opt, i, receiver);
             if (t < best.t) {
               best.t = t;
               best.dir = dir_shrink;
               best.pid = *pid_opt;
               best.from = i;
               best.to = receiver;
             }
           }
         }
         unsigned int size_receiver = cluster_sizes[best.to];
         unsigned int upper_receiver = upper_limit[best.to];

         if (size_receiver < upper_receiver) {
           // Perform flip (i.e., update clusters and priority queues)

           // Update median
           for (unsigned int j = 0; j < M; ++j) {
             median[j] += best.t * best.dir[j];
           }

           // Update queues
           remove(best.pid, best.from);
           insert_into_queues(best.pid, best.to);

           // Update clusters
           labels[best.pid] = best.to;
           cluster_sizes[best.from]--;
           cluster_sizes[best.to]++;
           break;
         } else if (M > 2) {
           // Push event to chain, define set of frozen hyperplanes, call the
           // flip chain building function recursively

           // Initialize flip chain
           std::vector<Event> flip_chain = { best };

           // Define initial set of frozen hyperplanes
           FrozenHyperplanes frozen_hyperplanes(
             mode, { best.from, best.to }, directions, M);
           printf("test 0\n");

           bool flip_chain_build =
             build_flip_chain(flip_chain, frozen_hyperplanes);
           if (flip_chain_build) {
             printf("test 1\n");

             // Perform flip chain (i.e., update clusters and priority queues)
             // and continue
             apply_flip_chain(flip_chain);
             break;
           } else {
             throw std::runtime_error(
               "fit() failed: no valid flip or chain found at iteration " +
               std::to_string(iteration));
           }
         }
       }
   */
    }

    ++iteration;
  }
  std::cerr << "Iterations:" << iteration << "\n" << std::flush;

  printf("m = (");
  for (unsigned int i = 0; i < M; ++i) {
    printf("%g, ", median[i]);
  }
  printf(")\n");
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
VolumeMedianFitter::mismatch_reduced(const std::vector<Event>& flip_chain) const
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
  for (const auto& ev : flip_chain) {
    current_sizes[ev.from]--;
    current_sizes[ev.to]++;
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
