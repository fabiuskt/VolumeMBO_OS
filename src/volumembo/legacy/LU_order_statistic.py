import matplotlib.pyplot as plt
import numpy as np
import collections as cl

# from volumembo.array_inherit import c_arr
# from volumembo.priority_queue import delete, heapify, heappush, heappop
from .array_inherit import c_arr
from .priority_queue import delete, heapify, heappush, heappop


def e(i, m):
    v = np.zeros(m)
    v[i] = 1.0
    return v


def n(i, j, m):
    return (e(i, m) - e(j, m)) / np.sqrt(2)


def compare_along_normal(x, y, normal):
    # comperator = -1 -> "<"
    # comperator = 0  -> "="
    # comperator = 1  -> ">"
    angle = (x[0] - y[0]).dot(normal)
    if angle < 0:
        return -1
    if angle > 0:
        return 1
    else:
        return 0


def comperator(i, j, m):
    return lambda x, y: compare_along_normal(x, y, e(i, m) - e(j, m))


def divide_into_clusters(median, samples):
    size = len(samples)
    cluster = np.ones(size, dtype=int) * (-1)
    for i in range(size):
        x = samples[i, :]
        cluster[i] = np.argmax(x - median)
    return cluster


def count_clusters(cluster, m):
    count_up = m * [0]
    for c in cluster:
        count_up[int(c)] += 1
    return count_up


def project_to_boundary(point, m):
    boundary_points = []
    for i in range(3):
        c = point[int((i + 2) % 3)] / 2
        boundary_point = point + c * (
            e(i, m) + e(int((i + 1) % 3), m) - 2 * e(int((i + 2) % 3), m)
        )
        boundary_points.append(boundary_point)
    return boundary_points


def draw_triangle_neu(median, neu, next_point, samples, clustering, m):
    ax = plt.figure().add_subplot(projection="3d")
    for boundary_point in project_to_boundary(median, m):
        ax.plot(
            [median[0], boundary_point[0]],
            [median[1], boundary_point[1]],
            [median[2], boundary_point[2]],
        )
    for boundary_point in project_to_boundary(neu, m):
        ax.plot(
            [neu[0], boundary_point[0]],
            [neu[1], boundary_point[1]],
            [neu[2], boundary_point[2]],
        )

    co = np.array(["red", "green", "purple"])

    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=co[clustering])
    ax.scatter(median[0], median[1], median[2], c="black")
    ax.scatter(neu[0], neu[1], neu[2], c="orange")

    ax.scatter(next_point[0], next_point[1], next_point[2], c="black", marker="*")

    X = np.arange(0.0, 1, 0.01)
    Y = np.arange(0.0, 1, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = 1 - X - Y
    Z[Z < 0] = np.inf
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.view_init(45, 45)
    plt.show()


def draw_triangle(median, samples, clustering, m):
    ax = plt.figure().add_subplot(projection="3d")
    boundary_vectors = []
    for boundary_point in project_to_boundary(median, m):
        ax.plot(
            [median[0], boundary_point[0]],
            [median[1], boundary_point[1]],
            [median[2], boundary_point[2]],
        )
    co = np.array(["red", "green", "purple"])
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=co[clustering])
    ax.scatter(median[0], median[1], median[2], c="black")

    X = np.arange(0.0, 1, 0.01)
    Y = np.arange(0.0, 1, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = 1 - X - Y
    Z[Z < 0] = np.inf
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.view_init(45, 45)
    plt.show()


def fit_median(m, L, U, samples, median):
    size = len(samples)
    ########clustering#########
    clustering = divide_into_clusters(median, samples)
    count = count_clusters(clustering, m)
    P = []
    for i in range(m):
        V = count[i]
        if V > U[i]:
            V = U[i]
        elif V < L[i]:
            V = L[i]
        P.append(V)

    sum = 0
    for p in P:
        sum += p

    if sum < size:
        difference = size - sum
        for i in range(m):
            if P[i] < U[i]:
                if difference <= U[i] - P[i]:
                    P[i] += difference
                    difference = 0
                else:
                    difference -= U[i] - P[i]
                    P[i] = U[i]
            if difference == 0:
                break

    if sum > size:
        difference = sum - size
        for i in range(m):
            if P[i] > L[i]:
                if difference <= P[i] - L[i]:
                    P[i] -= difference
                    difference = 0
                else:
                    difference += P[i] - L[i]
                    P[i] = L[i]
            if difference == 0:
                break
    #########sorting###########

    sorted_along_normal = [[[] for k in range(m)] for i in range(m)]
    indices_along_normal = [[[] for k in range(m)] for i in range(m)]
    eti_along_normal = [[[] for k in range(m)] for i in range(m)]
    for i in range(size):
        for j in range(m):
            for k in range(m):
                indices_along_normal[k][j].append(-1)
            if clustering[i] != j:
                indices_along_normal[clustering[i]][j][-1] = len(
                    sorted_along_normal[clustering[i]][j]
                )
                sorted_along_normal[clustering[i]][j].append([samples[i, :], i])
                eti_along_normal[clustering[i]][j].append(i)

    # resorted_boundaries= [[[],[],[]],[[],[],[]],[[],[],[]]]

    for i in range(m):
        for j in range(m):
            if i != j:
                heapify(
                    sorted_along_normal[i][j],
                    comperator(i, j, m),
                    indices_along_normal[i][j],
                    eti_along_normal[i][j],
                )

    ####################main step##################
    while P != count:
        # print(count)
        predecessors = m * [-1]
        diffs = np.array(count) - np.array(P)
        index = np.argmin(diffs)
        ins = [index]
        outs = []
        for j in range(m):
            if j != index:
                outs.append(j)
        direction = 1 / (m - 1) * np.ones(m) - (1 + 1 / (m - 1)) * e(index, m)
        while diffs[index] <= 0:
            minimum = np.inf
            min_in = -1
            min_out = -1
            for i in ins:
                for o in outs:
                    if sorted_along_normal[o][i]:
                        # next point in normal direction
                        next_point = sorted_along_normal[o][i][0][0]
                        # calculate distance to hypersurface
                        d = np.abs((next_point - median).dot(n(o, i, m)))
                        if d < minimum:
                            minimum = d
                            min_in = i
                            min_out = o

            # calculate according time
            time = minimum / direction.dot(n(min_out, min_in, m))
            # move
            median_neu = median + time * direction
            # draw_triangle_neu(median, median_neu,sorted_along_normal[min_out][min_in][0][0])
            median = median_neu
            # save
            predecessors[min_out] = min_in
            # set next direction
            index = min_out
            direction += 1 / (m - 1) * np.ones(m) - (1 + 1 / (m - 1)) * e(index, m)
            ins.append(index)
            outs.remove(index)

        # change along predecessor line
        while diffs[index] >= 0:
            predecessor = predecessors[index]
            popped, poppedindex = heappop(
                sorted_along_normal[index][predecessor],
                comperator(index, predecessor, m),
                indices_along_normal[index][predecessor],
                eti_along_normal[index][predecessor],
            )
            clustering[popped[1]] = predecessor
            count[predecessor] += 1
            count[index] -= 1

            for k in range(m):
                if k != index and k != predecessor:
                    # for i, element in enumerate(sorted_along_normal[index][k]):
                    #    if element[1] == popped[1]:
                    #        location = i
                    delete(
                        sorted_along_normal[index][k],
                        indices_along_normal[index][k][poppedindex],
                        comperator(index, k, m),
                        indices_along_normal[index][k],
                        eti_along_normal[index][k],
                    )
                if k != predecessor:
                    heappush(
                        sorted_along_normal[predecessor][k],
                        popped,
                        comperator(predecessor, k, m),
                        indices_along_normal[predecessor][k],
                        poppedindex,
                        eti_along_normal[predecessor][k],
                    )
            index = predecessor

    ################################## order statistic for P is found #################
    P = np.array(P)
    L = np.array(L)
    U = np.array(U)

    Jplus = np.array(range(m))[P > L]
    Jminus = np.array(range(m))[P < U]
    if len(Jminus) == 0:
        return median, clustering, P
    max_in_minus = -np.inf
    max_in_minus_indices = []
    for i in Jminus:
        if median[i] > max_in_minus:
            max_in_minus_indices = [i]
            max_in_minus = median[i]
        elif median[i] == max_in_minus:
            max_in_minus_indices.append(i)

    while any(median[max_in_minus_indices[0]] > median[Jplus]):
        direction = len(max_in_minus_indices) / (m - 1) * np.ones(m)
        for i in max_in_minus_indices:
            direction[i] -= 1 + 1 / (m - 1)

        min_in_plus = np.inf

        ins = max_in_minus_indices.copy()
        outs = []
        for j in range(m):
            if not j in ins:
                outs.append(j)
        optimal = False
        while min_in_plus > median[max_in_minus_indices[0]]:
            if len(ins) == m:
                optimal = True
                break
            minimum = np.inf
            min_in = -1
            min_out = -1
            for i in ins:
                for o in outs:
                    # next point in normal direction
                    next_point = sorted_along_normal[o][i][0][0]
                    # calculate distance to hypersurface
                    d = np.abs((next_point - median).dot(n(o, i, m)))
                    if d < minimum:
                        minimum = d
                        min_in = i
                        min_out = o

            # calculate according time
            time = minimum / direction.dot(n(min_out, min_in, m))

            max_non_T_minus = -np.inf
            max_non_T_minus_indices = []
            for i in Jminus:
                if median[i] != median[max_in_minus_indices[0]] and not i in ins:
                    if median[i] > max_non_T_minus:
                        max_non_T_minus_indices = [i]
                        max_non_T_minus = median[i]
                    elif median[i] == max_non_T_minus:
                        max_non_T_minus_indices.append(i)

            time_until_equality = np.inf
            if len(max_non_T_minus_indices) > 0:
                time_until_equality = (
                    median[max_in_minus_indices[0]] - max_non_T_minus
                ) / (1 + 1 / (m - 1))

            if time_until_equality < time:
                time = time_until_equality
                median = median + time * direction
                for i in max_non_T_minus_indices:
                    ins.append(i)
                    outs.remove(i)
                    max_in_minus_indices.append(i)
                    direction += 1 / (m - 1) * np.ones(m) - (1 + 1 / (m - 1)) * e(i, m)

            else:
                predecessors[min_out] = min_in
                median = median + time * direction
                ins.append(min_out)
                outs.remove(min_out)
                direction += 1 / (m - 1) * np.ones(m) - (1 + 1 / (m - 1)) * e(
                    min_out, m
                )
                if P[min_out] > L[min_out]:
                    min_in_plus = median[min_out]

        index = min_out
        # change along predecessor line
        while not index in max_in_minus_indices and not optimal:
            predecessor = predecessors[index]
            popped, poppedindex = heappop(
                sorted_along_normal[index][predecessor],
                comperator(index, predecessor, m),
                indices_along_normal[index][predecessor],
                eti_along_normal[index][predecessor],
            )
            clustering[popped[1]] = predecessor
            P[predecessor] += 1
            P[index] -= 1

            for k in range(m):
                if k != index and k != predecessor:
                    # for i, element in enumerate(sorted_along_normal[index][k]):
                    #    if element[1] == popped[1]:
                    #        location = i
                    delete(
                        sorted_along_normal[index][k],
                        indices_along_normal[index][k][poppedindex],
                        comperator(index, k, m),
                        indices_along_normal[index][k],
                        eti_along_normal[index][k],
                    )
                if k != predecessor:
                    heappush(
                        sorted_along_normal[predecessor][k],
                        popped,
                        comperator(predecessor, k, m),
                        indices_along_normal[predecessor][k],
                        poppedindex,
                        eti_along_normal[predecessor][k],
                    )
            index = predecessor

        Jplus = np.array(range(m))[P > L]
        Jminus = np.array(range(m))[P < U]
        if len(Jminus) == 0:
            return median, clustering
        max_in_minus = -np.inf
        max_in_minus_indices = []
        for i in Jminus:
            if median[i] > max_in_minus:
                max_in_minus_indices = [i]
                max_in_minus = median[i]
            elif median[i] == max_in_minus:
                max_in_minus_indices.append(i)

    return median, clustering, P


def energy(samples, clustering):
    sum = 0
    for i in range(len(clustering)):
        sum += samples[i][clustering[i]]
    return sum


def test():
    size = 60

    m = 4

    # P = [5,5,5]
    P = m * [size // m]
    L = [1, 20, 4, 6]
    U = [30, 30, 30, 30]

    samples = (
        np.random.uniform(0, 1, m * size) + np.array([0.3, 0, 0.6, 0] * size)
    ).reshape((size, m))
    # samples = (np.random.uniform(0,1,m*size)).reshape((size,m))
    for i in range(size):
        samples[i, :] /= np.sum(samples[i, :])

    # median = np.array(m*[1/m])
    median = np.array(m * [1 / m])
    median, clustering, P = fit_median(m, L, U, samples, median)
    print(energy(samples, clustering), P, median)
    max_energy = -np.inf
    max_P = []
    max_med = []
    for a in range(L[0], U[0] + 1):
        for b in range(L[1], U[1] + 1):
            for c in range(L[2], U[2] + 1):
                for d in range(L[3], U[3] + 1):
                    if a + b + c + d == size:
                        P = [a, b, c, d]
                        median, clustering, P = fit_median(m, P, P, samples, median)
                        en = energy(samples, clustering)
                        if en > max_energy:
                            max_energy = en
                            max_P = P
                            max_med = median

    print(max_energy, max_P, max_med)
