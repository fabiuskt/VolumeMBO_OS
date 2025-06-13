#include "volumembo/median_fitter.hpp"

#include "volumembo/priority_queue.hpp"
#include "volumembo/span2d.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
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
  const double eps = 1.0e-4;
  const int max_iter = 100;
  unsigned int iteration = 0;
  int offset = 0;

  while (!volumes_matched()) {
    if (iteration > max_iter)
      break;

    Label cluster = select_label(offset);
    offset = (cluster + 1) % M;

    std::vector<std::tuple<PID, Label, Label>> candidates;
    bool grow = cluster_sizes[cluster] < lower_limit[cluster];
    std::vector<double> dir = directions[cluster];

    if (grow) {
      Label to_label = cluster;
      for (Label from_label : other_labels[to_label]) {
        auto pid_opt = peek(from_label, to_label);
        if (pid_opt) {
          candidates.emplace_back(*pid_opt, from_label, to_label);
        }
      }
    } else {
      for (double& val : dir)
        val = -val;
      Label from_label = cluster;
      for (Label to_label : other_labels[from_label]) {
        auto pid_opt = peek(from_label, to_label);
        if (pid_opt) {
          candidates.emplace_back(*pid_opt, from_label, to_label);
        }
      }
    }
    if (candidates.size() < 1) {
      throw std::runtime_error("Not enough valid candidates for cluster " +
                               std::to_string(cluster) + " at iteration " +
                               std::to_string(iteration));
    }

    auto [pid, from_label, to_label] = candidates[0];
    double flip_time = compute_flip_time(pid, from_label, to_label);

    // Optional: if you have 2 or more, pick the best one
    if (candidates.size() > 1) {
      auto [pid_alt, from_alt, to_alt] = candidates[1];
      double ft_alt = compute_flip_time(pid_alt, from_alt, to_alt);
      if (ft_alt < flip_time) {
        pid = pid_alt;
        from_label = from_alt;
        to_label = to_alt;
        flip_time = ft_alt;
      }
    }

    // Update median
    for (unsigned int j = 0; j < M; ++j) {
      median[j] += (1.0 + eps) * flip_time * dir[j];
    }
    // Update queues
    remove(pid, from_label);
    insert_into_queues(pid, to_label);

    // Update clusters
    labels[pid] = to_label;
    cluster_sizes[from_label]--;
    cluster_sizes[to_label]++;

    ++iteration;
  }

  return median;
}

void
VolumeMedianFitter::assign_clusters()
{
  cluster_sizes.assign(M, 0);

  for (size_t i = 0; i < N; ++i) {
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
VolumeMedianFitter::compute_u_minus_m(size_t index) const
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

Label
VolumeMedianFitter::select_label(int offset) const
{
  std::vector<unsigned int> violations(M);

  for (Label i = 0; i < M; ++i) {
    unsigned int size = cluster_sizes[i];
    unsigned int under = (lower_limit[i] > size) ? (lower_limit[i] - size) : 0;
    unsigned int over = (size > upper_limit[i]) ? (size - upper_limit[i]) : 0;
    violations[i] = std::max(under, over);
  }

  auto max_violation = *std::max_element(violations.begin(), violations.end());

  if (max_violation == 0) {
    // No violations, fallback to round-robin cycling
    return static_cast<Label>(offset % M);
  }

  // Rotate violations left by offset
  std::vector<unsigned int> rotated(M);
  for (Label i = 0; i < M; ++i) {
    rotated[i] = violations[(i + offset) % M];
  }

  auto max_it = std::max_element(rotated.begin(), rotated.end());
  Label idx_in_rotated =
    static_cast<Label>(std::distance(rotated.begin(), max_it));

  return static_cast<Label>((offset + idx_in_rotated) % M);
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
