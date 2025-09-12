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

std::vector<double>
VolumeMedianFitter::fit()
{

  struct Event
  {
    double t = std::numeric_limits<double>::infinity();
    PID pid = 0;
    Label from = 0;
    Label to = 0;
    std::vector<double> dir;
  };

  constexpr unsigned int max_iter = 100000;
  unsigned int iteration = 0;

  while (!volumes_matched()) {
    if (iteration > max_iter)
      break;

    Event best;

    for (Label i = 0; i < M; ++i) {
      unsigned int size_i = cluster_sizes[i];
      unsigned int lower_i = lower_limit[i];
      unsigned int upper_i = upper_limit[i];

      const std::vector<double> dir = directions[i];

      // --- Shrink direction (-i) ---
      if (size_i > lower_i) { // i can give away a point
        for (Label receiver : other_labels[i]) {
          unsigned int size_receiver = cluster_sizes[receiver];
          unsigned int lower_receiver = lower_limit[receiver];
          unsigned int upper_receiver = upper_limit[receiver];

          if (size_receiver < upper_receiver &&
              (size_receiver < lower_receiver || size_i > upper_i)) {
            if (auto pid_opt = peek(i, receiver)) {
              double t = compute_flip_time(*pid_opt, i, receiver);
              if (t < best.t) {
                std::vector<double> dir_shrink = dir;
                for (double& v : dir_shrink)
                  v = -v;

                best.t = t;
                best.pid = *pid_opt;
                best.from = i;
                best.to = receiver;
                best.dir = dir_shrink;
              }
            }
          }
        }
      }

      // --- Grow direction (+i) ---
      if (size_i < upper_i) { // i can receive a point
        for (Label donor : other_labels[i]) {
          unsigned int size_donor = cluster_sizes[donor];
          unsigned int lower_donor = lower_limit[donor];
          unsigned int upper_donor = upper_limit[donor];

          if (size_donor > lower_donor &&
              (size_donor > upper_donor || size_i < lower_i)) {
            if (auto pid_opt = peek(donor, i)) {
              double t = compute_flip_time(*pid_opt, donor, i);
              if (t < best.t) {
                best.t = t;
                best.pid = *pid_opt;
                best.from = donor;
                best.to = i;
                best.dir = dir; // grow direction = +i
              }
            }
          }
        }
      }
    }

    if (!std::isfinite(best.t)) {
      throw std::runtime_error(
        "fit() failed: no valid flip found at iteration " +
        std::to_string(iteration));
    }

    // Update median
    for (unsigned int j = 0; j < M; ++j) {
      median[j] += best.t * best.dir[j];
    }

    // Update queues
    remove(best.pid, best.from);
    insert_into_queues(best.pid, best.to);

    // ---- DEBUG ----
    long long mismatch = 0;
    for (Label k = 0; k < M; ++k) {
      mismatch +=
        std::llabs((long long)cluster_sizes[k] - (long long)lower_limit[k]);
    }

    std::ostringstream oss;
    oss << "Iteration " << iteration << ", mismatch=" << mismatch << "\n";
    for (Label k = 0; k < M; ++k) {
      int size = (int)cluster_sizes[k];
      int lower = (int)lower_limit[k];
      int upper = (int)upper_limit[k];
      oss << "  Cluster " << k << ": size=" << size << " (range [" << lower
          << "," << upper << "])\n";
    }

    int from_sz = (int)cluster_sizes[best.from];
    int to_sz = (int)cluster_sizes[best.to];
    int from_lo = (int)lower_limit[best.from];
    int from_up = (int)upper_limit[best.from];
    int to_lo = (int)lower_limit[best.to];
    int to_up = (int)upper_limit[best.to];

    oss << "  >> Flip event: " << best.from << " → " << best.to
        << " (pid=" << best.pid << ", t=" << best.t << ")\n"
        << "     from_size " << from_sz << " -> " << (from_sz - 1)
        << " (range [" << from_lo << "," << from_up << "])\n"
        << "     to_size   " << to_sz << " -> " << (to_sz + 1) << " (range ["
        << to_lo << "," << to_up << "])\n"
        << "--------------------------------------\n";

    std::cerr << oss.str() << std::flush; // single write

    // Update clusters
    labels[best.pid] = best.to;
    cluster_sizes[best.from]--;
    cluster_sizes[best.to]++;

    ++iteration;
  }
  std::cerr << "Iterations:" << iteration << std::flush;
  return median;
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
