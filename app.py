import numpy as np
from scipy.spatial.distance import pdist, squareform
from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import random

app = Flask(__name__)
CORS(app)  # 允许跨域请求


class WindFarmLayoutEvaluator:
    def __init__(self, width, height, n_turbines, obstacles):
        self.width = width
        self.height = height
        self.n_turbines = n_turbines
        self.obstacles = obstacles

        # 场地与风机物理参数 (对应原代码 ws)
        self.R = 38.5
        self.k = 0.0750
        self.vRated = 14
        self.vCin = 3.5
        self.vCout = 20
        self.PRated = 1500.0
        self.CT = 0.8
        self.energy_base = 7315.38  # 理论无尾流能量基准

    def check_constraint(self, layout):
        if len(layout) == 0: return False
        if len(layout) != self.n_turbines: return False  # 确保风机数量正确

        # 1. 检查间距约束 (D <= 8*R)
        D = squareform(pdist(layout))
        np.fill_diagonal(D, np.inf)
        if np.any(D <= 8 * self.R): return False

        # 2. 检查边界
        if np.any(layout < 0) or np.any(layout[:, 0] > self.width) or np.any(layout[:, 1] > self.height):
            return False

        # 3. 检查障碍物
        for pt in layout:
            for ob in self.obstacles:
                if ob['xmin'] < pt[0] < ob['xmax'] and ob['ymin'] < pt[1] < ob['ymax']:
                    return False
        return True

    def evaluate(self, layout):
        if not self.check_constraint(layout):
            return float('inf'), -1

        n = len(layout)

        # 简化版尾流计算(用于快速演示)
        TotalVdef = np.zeros(n)
        for i in range(n):
            downstream_deficits_sq_sum = 0
            for j in range(n):
                if i == j: continue
                dx = layout[i, 0] - layout[j, 0]
                # 假设风从左向右吹(0度), 只有上游风机(dx>0)对下游有影响
                if dx > 0:
                    dist = np.hypot(dx, layout[i, 1] - layout[j, 1])
                    if dist < 2000:  # 假设尾流影响2km
                        a = 1 - np.sqrt(1 - self.CT)
                        b = self.k / self.R
                        deficit = a / ((1 + b * dist) ** 2)
                        downstream_deficits_sq_sum += deficit ** 2
            TotalVdef[i] = np.sqrt(downstream_deficits_sq_sum)

        wfRatio = np.mean(1 - TotalVdef)
        if wfRatio <= 0: wfRatio = 0.01

        # 度电成本
        ct, cs, m, r, y, com = 750000, 8000000, 30, 0.03, 20, 20000
        A = (ct * n + cs * math.floor(n / m)) * (0.666667 + 0.333333 * math.exp(-0.00174 * n * n)) + com * n
        B = ((1.0 - (1.0 + r) ** (-y)) / r)
        C = (8760.0 * self.energy_base * wfRatio * n)
        EnergyCost = (A / B / C) + (0.1 / n)

        return EnergyCost, wfRatio


def run_ga_algorithm(width, height, n_turbines, obstacles):
    evaluator = WindFarmLayoutEvaluator(width, height, n_turbines, obstacles)

    interval = 8.001 * evaluator.R
    xs = np.arange(0, width, interval)
    ys = np.arange(0, height, interval)
    grid_x, grid_y = np.meshgrid(xs, ys)
    raw_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    valid_grid_mask = np.ones(len(raw_grid), dtype=bool)
    for i, pt in enumerate(raw_grid):
        for ob in obstacles:
            if ob['xmin'] < pt[0] < ob['xmax'] and ob['ymin'] < pt[1] < ob['ymax']:
                valid_grid_mask[i] = False
                break
    valid_grid = raw_grid[valid_grid_mask]
    max_turbs = len(valid_grid)

    if n_turbines > max_turbs:
        return {"error": f"风机数量过多，此区域最多容纳 {max_turbs} 台风机。"}, 0, 0

    num_pop, tour_size, mut_rate, cross_rate, generations = 20, 4, 0.05, 0.40, 30

    pops = np.array([np.random.choice(max_turbs, n_turbines, replace=False) for _ in range(num_pop)])
    fits = np.ones(num_pop) * float('inf')

    best_fit_overall = float('inf')
    best_indices_overall = pops[0]

    for gen in range(generations):
        for p_idx, p_indices in enumerate(pops):
            layout = valid_grid[p_indices]
            cost, _ = evaluator.evaluate(layout)
            fits[p_idx] = cost

        best_gen_idx = np.argmin(fits)
        if fits[best_gen_idx] < best_fit_overall:
            best_fit_overall = fits[best_gen_idx]
            best_indices_overall = pops[best_gen_idx]

        winners_indices = []
        available_indices = list(range(num_pop))
        random.shuffle(available_indices)
        for i in range(0, num_pop, tour_size):
            tourney_indices = available_indices[i:i + tour_size]
            if not tourney_indices: continue
            tourney_fits = fits[tourney_indices]
            winner_in_tourney = tourney_indices[np.argmin(tourney_fits)]
            winners_indices.append(winner_in_tourney)

        new_pops = np.zeros_like(pops)
        for i in range(len(winners_indices)):
            new_pops[i] = pops[winners_indices[i]]

        for i in range(len(winners_indices), num_pop):
            p1_idx, p2_idx = random.sample(winners_indices, 2)
            p1, p2 = pops[p1_idx], pops[p2_idx]

            common = np.intersect1d(p1, p2)
            unique_p1 = np.setdiff1d(p1, common)
            unique_p2 = np.setdiff1d(p2, common)

            cross_num = min(round(cross_rate * len(unique_p1)), len(unique_p2))

            to_replace_from_p1 = np.random.choice(unique_p1, cross_num, replace=False)
            new_vals_from_p2 = np.random.choice(unique_p2, cross_num, replace=False)

            child = np.setdiff1d(p1, to_replace_from_p1)
            child = np.union1d(child, new_vals_from_p2)

            if random.random() < mut_rate:
                available_grid_indices = np.setdiff1d(np.arange(max_turbs), child)
                if available_grid_indices.size > 0:
                    to_mutate_idx = random.choice(child)
                    new_gene = random.choice(available_grid_indices)
                    child = np.setdiff1d(child, [to_mutate_idx])
                    child = np.union1d(child, [new_gene])

            new_pops[i] = child
        pops = new_pops

    best_layout = valid_grid[best_indices_overall]
    best_cost, best_ratio = evaluator.evaluate(best_layout)

    return best_layout.tolist(), best_cost, best_ratio


@app.route('/run_optimization', methods=['POST'])
def handle_run_optimization():
    data = request.json
    width = data.get('width', 7000)
    height = data.get('height', 14000)
    turbines = data.get('turbines', 25)
    obstacles = data.get('obstacles', [])

    layout, cost, ratio = run_ga_algorithm(width, height, turbines, obstacles)

    if isinstance(layout, dict) and "error" in layout:
        return jsonify({"status": "error", "message": layout["error"]})

    return jsonify({
        'status': 'success',
        'layout': layout,
        'cost': round(cost, 5) if cost != float('inf') else 'N/A',
        'ratio': round(ratio, 4) if ratio != -1 else 'N/A'
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)