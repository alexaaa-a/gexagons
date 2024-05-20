import numpy as np
from itertools import product
import plotly.express as px
import random
import math

# создание карты
def create_map(max_dist: int):
    N = max_dist * 2 + 1
    data = np.zeros((N ** 3, 4), dtype=int)
    k = 0

    for i in product([str(x) for x in range(N)], repeat=3):
        data[k] = [int(i[0]), int(i[1]), int(i[2]), 1]
        k += 1

    return data


MAX_DIST = 20
map_data = create_map(MAX_DIST)

# получение индексов
def get_hex_idxs(coords_map):
    c = 20
    sums = np.sum(coords_map, axis=1)
    idxs = np.where(sums == 3 * c)[0] + 41

    return idxs


hexagon_idxs = get_hex_idxs(map_data)


hexagon_data = map_data[hexagon_idxs]
N = MAX_DIST * 2 + 1
for hexagon in hexagon_data:
    center = [20, 20, 20]
    dist = max(abs(hexagon[0] - 20), abs(hexagon[1] - 20), abs(hexagon[2] - 20))
    hexagon[-1] = dist

new_map = create_map(MAX_DIST)
hex_idxs = get_hex_idxs(new_map)
hex_data = new_map[hex_idxs]


ALL_MOVEMENTS = np.array([
    [0, 1, -1],
    [1, 0, -1],
    [1, -1, 0],
    [0, -1, 1],
    [-1, 0, 1],
    [-1, 1, 0]
])

river_movements = ALL_MOVEMENTS[0:2]

RIVERS_COUNT = 40

MAX_RIVER_LENGTH = 10

RIVER_TYPE = 10

np.random.seed(42)
random.seed(42)

river_map = np.copy(hex_data)

for _ in range(RIVERS_COUNT):
    river_length = random.randint(1, MAX_RIVER_LENGTH)
    start_index = random.randint(0, hex_data.shape[0])
    start_point = river_map[start_index]
    river_map[start_index][3] = RIVER_TYPE

    for _ in range(river_length):
        movement = random.choice(river_movements)
        new_point = np.array([start_point[0] + movement[0], start_point[1] + movement[1], start_point[2] + movement[2], 1], int)
        idx = new_point[0] * N ** 2 + new_point[1] * N + new_point[2]
        new_index = np.where(hex_idxs == idx)

        if len(new_index[0]) > 0:
            river_map[new_index[0][0]][3] = RIVER_TYPE
            start_point = new_point
        else:
            continue

hills_map = np.copy(river_map)

HILLS_COUNT = 10

HILLS_TYPE = 100

for _ in range(HILLS_COUNT):
    start_index = random.randint(0, hills_map.shape[0])
    start_point = hills_map[start_index]
    hills_map[start_index][3] = HILLS_TYPE

    for movement in ALL_MOVEMENTS:
        new_point = np.array(
            [start_point[0] + movement[0], start_point[1] + movement[1], start_point[2] + movement[2], 1], int)
        idx = new_point[0] * N ** 2 + new_point[1] * N + new_point[2]
        new_index = np.where(hex_idxs == idx)

        if len(new_index[0]) > 0:
            hills_map[new_index[0][0]][3] = HILLS_TYPE
        else:
            continue

ROUTE_TYPE = 1000

PLAIN_TYPE = 1

types_map = {
    PLAIN_TYPE: "Plain",
    HILLS_TYPE: "Hill",
    RIVER_TYPE: "River"
}
vfunc = np.vectorize(types_map.get)
colors = vfunc(hills_map[:, 3])

fig = px.scatter_3d(
    hills_map,
    x=hills_map[:, 0],
    y=hills_map[:, 1],
    z=hills_map[:, 2],
    hover_name=hex_idxs,
    color=colors,
    color_discrete_sequence=["#387C44", "#3EA99F", "gray"],
)
# fig.show()

def visualize_solve(route_path):
    route_map = np.copy(hills_map)

    for cell_idx in route_path:
        new_idx = np.where(hex_idxs == cell_idx)
        route_map[new_idx[0][0]][-1] = ROUTE_TYPE

    vfunc = np.vectorize(types_map.get)

    colors = vfunc(route_map[:, 3])

    fig = px.scatter_3d(
        route_map,
        x=route_map[:, 0],
        y=route_map[:, 1],
        z=route_map[:, 2],
        hover_name=hex_idxs,
        color=colors,
        color_discrete_sequence=["#387C44", "#3EA99F", "gray", "maroon"],
    )
    fig.show()

back_from_idx = 52_940
back_to_idx = 45_860

def count_dist(a_idx, b_idx):
    a_data = new_map[a_idx]
    b_data = new_map[b_idx]

    return math.sqrt(sum((a_data - b_data)**2))

init_dist = count_dist(back_from_idx, back_to_idx)

# бэктрекинг
def backtracking(curr_idx, path, prev_dist, prev_movement_idx):
    curr_point = np.where(hex_idxs == curr_idx)
    curr_data = hills_map[curr_point[0][0]][:3]

    path.append(curr_idx)

    if curr_idx == back_to_idx:
        return path

    for i, movement in enumerate(ALL_MOVEMENTS):
        if i == prev_movement_idx:
            continue

        new_data = [curr_data[0] + movement[0], curr_data[1] + movement[1], curr_data[2] + movement[2], i]
        new_idx = new_data[0] * N ** 2 + new_data[1] * N + new_data[2]

        if new_idx in path:
            continue

        new_dist = count_dist(new_idx, back_to_idx)

        if (prev_dist - new_dist) / init_dist > 2 or new_dist > prev_dist:
            continue
        else:
            prev_movement_idx = i

        result = backtracking(new_idx, path.copy(), new_dist, i)
        if result is not None:
            return result

    path.pop()

back_path = backtracking(back_from_idx, [back_from_idx], init_dist, -1)

# visualize_solve(back_path)


greedy_path = [back_from_idx]
curr_point = np.where(hex_idxs == back_from_idx)[0][0]

curr_dist = init_dist
curr_data = hills_map[curr_point][:3]

# жадные алгоритмы
def count_greedy_path(idx_from, idx_to):
    greedy_path = [idx_from]
    curr_idx = idx_from
    init_dist = count_dist(idx_from, idx_to)
    curr_dist = init_dist

    while curr_dist > 0:
        curr_point = np.where(hex_idxs == curr_idx)
        curr_data = hills_map[curr_point[0][0]][:3]

        possible_moves = []

        for movement in ALL_MOVEMENTS:
            new_data = [curr_data[0] + movement[0], curr_data[1] + movement[1], curr_data[2] + movement[2]]
            new_idx = new_data[0] * N ** 2 + new_data[1] * N + new_data[2]

            if new_idx in greedy_path:
                continue

            new_point = np.where(hex_idxs == new_idx)
            if new_point[0].size == 0:
                continue

            new_landscape = hills_map[new_point[0][0]][3]

            if new_landscape == RIVER_TYPE or new_landscape == HILLS_TYPE:
                continue

            new_dist = count_dist(new_idx, idx_to)
            possible_moves.append((new_idx, new_dist))

        if not possible_moves:
            break

        best_move = min(possible_moves, key=lambda x: x[1])
        greedy_path.append(best_move[0])
        curr_idx = best_move[0]
        curr_dist = best_move[1]

    return greedy_path


def visualise_greedy_solve(idx_from, idx_to):
    greedy_path = count_greedy_path(idx_from, idx_to)

    visualize_solve(greedy_path)

# print(visualise_greedy_solve(29540, 46220))


