import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# parameter
N_SAMPLE = 400
N_KNN = 10
MAX_EDGE_LEN = 30.0
show_animation = True


class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index


def prm_planning(start_x, start_y, goal_x, goal_y,
                 obstacles, robot_radius, rng=None):

    sample_x, sample_y = sample_points(
        start_x, start_y, goal_x, goal_y,
        robot_radius, obstacles, rng
    )

    if show_animation:
        plt.plot(sample_x, sample_y, ".b", alpha=0.5)

    road_map = generate_road_map(
        sample_x, sample_y, robot_radius, obstacles
    )

    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y,
        road_map, sample_x, sample_y
    )

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacles):
    dx = gx - sx
    dy = gy - sy
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    steps = max(int(d / rr), 1)

    for i in range(steps + 1):
        t = i / steps
        x = sx + t * dx
        y = sy + t * dy

        for (ox, oy, r) in obstacles:
            if math.hypot(ox - x, oy - y) <= (r + rr):
                return True

    return False


def generate_road_map(sample_x, sample_y, rr, obstacles):
    road_map = []
    n_sample = len(sample_x)

    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for i, (ix, iy) in enumerate(zip(sample_x, sample_y)):
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)

        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacles):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    while True:
        if not open_set:
            print("Cannot find path")
            return [], []

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        if show_animation and len(closed_set) % 5 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("Goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        del open_set[c_id]
        closed_set[c_id] = current

        for n_id in road_map[c_id]:
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)

            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue

            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index

    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def sample_points(sx, sy, gx, gy, rr, obstacles, rng):
    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = rng.uniform(-2, 15)
        ty = rng.uniform(-2, 14)

        collision = False
        for (ox, oy, r) in obstacles:
            if math.hypot(tx - ox, ty - oy) <= (r + rr):
                collision = True
                break

        if not collision:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.extend([sx, gx])
    sample_y.extend([sy, gy])

    return sample_x, sample_y


def main(rng=None):
    print("PRM start!!")

    sx, sy = 0.0, 0.0
    gx, gy = 6.0, 10.0
    robot_size = 0.5

    obstacles = [
        (1, 10, 1), (3, 10, 1),
        (3, 8, 1), (3, 6, 1),
        (5, 5, 1), (7, 5, 1), (9, 5, 1),
        (9, 3, 1),
        (8, 10, 1),
        (14, 8, 2)
    ]

    if show_animation:
        fig, ax = plt.subplots()

        for (ox, oy, r) in obstacles:
            circle = plt.Circle((ox, oy), r,
                                color='black',
                                fill=False,
                                linewidth=2)
            ax.add_patch(circle)

        ax.plot(sx, sy, "xr", markersize=10)
        ax.plot(gx, gy, "xr", markersize=10)

        ax.set_xlim(-2, 16)
        ax.set_ylim(-2, 15)

        ax.set_aspect('equal')
        ax.grid(True)

    rx, ry = prm_planning(
        sx, sy, gx, gy, obstacles, robot_size
    )

    if rx:
        plt.plot(rx, ry, "-r", linewidth=2)
    else:
        print("Path not found!")

    plt.show()


if __name__ == '__main__':
    main()
