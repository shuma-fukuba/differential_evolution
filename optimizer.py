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
    def __init__(self, config: Configuration, function: Function):
        self.config = config  # 設定
        self.function = function  # 関数
        self.pop = []  # 個体群

    def initializeSolutions(self):
        for i in range(self.config.max_pop):
            self.pop.append(Solution(self.config, self.function))
            self.getFitness(self.pop[i])

    # 次世代個体群生成
    def getNextPopulation(self):
        self.getElite()
        while len(self.pop) < self.config.max_pop:
            p1, p2 = self.selectParent()
            o1, o2 = self.generateOffspring(p1, p2)
            self.getFitness(o1)
            self.getFitness(o2)
            self.pop.append(o1)
            self.pop.append(o2)

    # 淘汰（親個体の選択）
    def selectParent(self):
        if self.config.selection == "RW":
            return self.RWselection(), self.RWselection()
        elif self.config.selection == "TS":
            return self.TSselection(), self.TSselection()
        else:
            print("Error in selectParent")

    # ルーレット選択
    def RWselection(self):
        Fsum = 0.
        for i, v in enumerate(self.pop):
            Fsum += v.f
        choice = Fsum * self.config.rd.rand()
        Fsum = 0.
        for i, v in enumerate(self.pop):
            Fsum += v.f
            if Fsum >= choice:
                return v
        return None

    # トーナメント選択（トーナメントサイズ Nt=1）
    def TSselection(self):
        i, j = self.config.rd.randint(
            0, len(self.pop)), self.config.rd.randint(0, len(self.pop))

        if self.pop[i].f < self.pop[j].f:
            return self.pop[i]
        else:
            return self.pop[j]

    # エリート戦略
    def getElite(self):
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
    def generateOffspring(self, p1, p2):
        o1, o2 = self.applyXover(Solution(self.config, self.function, parent=p1), Solution(
            self.config, self.function, parent=p2))
        o1.mutation()
        o2.mutation()
        return o1, o2

    # 交叉（一点交叉）
    def applyXover(self, o1, o2):
        if self.config.rd.rand() >= self.config.p_x:
            return o1, o2
        x_point = self.config.rd.randint(0, self.config.prob_dimension)
        for i in range(x_point):
            tmp = o1.x[i]
            o1.x[i] = o2.x[i]
            o2.x[i] = tmp
        return o1, o2

    # 評価値fの計算
    def getFitness(self, solution):
        solution.f = self.function.do_evaluate(solution.x)


# 個体のクラス
class Solution:

    def __init__(self, config: Configuration, function: Function, parent=None):
        self.config, self.function, self.x, self.f = config, function, [], 0.
        # 個体の初期化
        if parent is None:
            self.x = [self.config.rd.uniform(
                self.function.axis_range[0], self.function.axis_range[1]) for _ in range(self.config.prob_dimension)]
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
