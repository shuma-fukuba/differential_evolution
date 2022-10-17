###### optimizer.py #####
#                                           Last Update:  2020/4/13
#
# 遺伝的アルゴリズムの詳細アルゴリズムファイル
# インスタンスはoptとして生成

# 他ファイル,モジュールのインポート
import numpy as np

from configuration import Configuration
from function import Function

# 遺伝的アルゴリズム（実数）クラス


class GeneticAlgorithm:
    """parent genetic algorithm"""

    def __init__(self, config: Configuration, function: Function) -> None:
        self.config = config  # 設定
        self.function = function  # 関数
        self.pop = []  # 個体群

    def initialize_solutions(self):
        """initialize solutions"""
        for i in range(self.config.max_pop):
            self.pop.append(Solution(self.config, self.function))
            self.get_fitness(self.pop[i])

    # 次世代個体群生成
    def get_next_population(self):
        self.get_elite()
        while len(self.pop) < self.config.max_pop:
            p1, p2 = self.select_parent()
            o1, o2 = self.generate_offspring(p1, p2)
            self.get_fitness(o1)
            self.get_fitness(o2)
            self.pop.append(o1)
            self.pop.append(o2)

    # 淘汰（親個体の選択）
    def select_parent(self):
        if self.config.selection == "RW":
            return self.RW_selection(), self.RW_selection()
        elif self.config.selection == "TS":
            return self.TS_selection(), self.TS_selection()
        else:
            print("Error in select_parent")

    # ルーレット選択
    def RW_selection(self):
        f_sum = 0.
        for v in self.pop:
            f_sum += v.f
        choice = f_sum * self.config.rd.rand()
        f_sum = 0.
        for v in self.pop:
            f_sum += v.f
            if f_sum >= choice:
                return v
        return None

    # トーナメント選択（トーナメントサイズ Nt=1）
    def TS_selection(self):
        i, j = self.config.rd.randint(
            0, len(self.pop)), self.config.rd.randint(0, len(self.pop))

        if self.pop[i].f < self.pop[j].f:
            return self.pop[i]
        else:
            return self.pop[j]

    # エリート戦略
    def get_elite(self):
        for i, i_pop in enumerate(self.pop):
            for j, j_pop in enumerate(self.pop):
                if i != j and i_pop.f > j_pop.f:
                    tmp = i_pop
                    i_pop = j_pop
                    j_pop = tmp
        cnt = (int)(len(self.pop) * self.config.p_elite)
        for i in range(cnt):
            self.pop.pop(0)

    # 子個体の生成
    def generate_offspring(self, p1, p2):
        solution_1, solution_2 = self.apply_x_over(Solution(self.config, self.function, parent=p1), Solution(
            self.config, self.function, parent=p2))
        solution_1.mutation()
        solution_2.mutation()
        return solution_1, solution_2

    # 交叉（一点交叉）
    def apply_x_over(self, solution_1, solution_2):
        if self.config.rd.rand() >= self.config.p_x:
            return solution_1, solution_2
        x_point = self.config.rd.randint(0, self.config.prob_dimension)
        for i in range(x_point):
            tmp = solution_1.x[i]
            solution_1.x[i] = solution_2.x[i]
            solution_2.x[i] = tmp
        return solution_1, solution_2

    # 評価値fの計算
    def get_fitness(self, solution):
        solution.f = self.function.do_evaluate(solution.x)


# 個体のクラス
class Solution:

    def __init__(self, config: Configuration, function: Function, parent=None):
        self.config, self.function, self.x, self.f = \
            config, function, [], 0.
        # 個体の初期化
        if parent is None:
            self.x = [self.config.rd.uniform(
                self.function.axis_range[0], self.function.axis_range[1])
                for _ in range(self.config.prob_dimension)]
        # 親個体のコピー
        else:
            self.x = [parent.x[i] for i in range(self.config.prob_dimension)]
        # リスト -> ndarray
        self.x = np.array(self.x)

    # 突然変異（摂動）
    def mutation(self):
        dx = 0.05   # 摂動の微小量
        for i in range(self.config.prob_dimension):
            if self.config.rd.rand() < self.config.p_m:
                # 摂動
                self.x[i] += self.config.rd.uniform(-dx, dx) * (
                    self.function.axis_range[1] - self.function.axis_range[0])
                # 定義域外の探索防止
                self.x[i] = np.clip(
                    self.x[i], self.function.axis_range[0], self.function.axis_range[1])
