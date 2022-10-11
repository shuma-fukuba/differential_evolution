###### optimizer.py #####
#                                           Last Update:  2020/4/13
#
# 遺伝的アルゴリズムの詳細アルゴリズムファイル
# インスタンスはoptとして生成

# 他ファイル,モジュールのインポート
import function as fc
import numpy as np

# 遺伝的アルゴリズム（実数）クラス
class GeneticAlgorithm:

    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc):
        self.cnf = cnf      # 設定
        self.fnc = fnc      # 関数
        self.pop = []       # 個体群

    """ インスタンスメソッド """
    # 初期化
    def initializeSolutions(self):
        for i in range(self.cnf.max_pop):
            self.pop.append(Solution(self.cnf, self.fnc))
            self.getFitness(self.pop[i])

    # 次世代個体群生成
    def getNextPopulation(self):
        self.getElite()
        while len(self.pop) < self.cnf.max_pop:
            p1, p2 = self.selectParent()
            o1, o2 = self.generateOffspring(p1, p2)
            self.getFitness(o1)
            self.getFitness(o2)
            self.pop.append(o1)
            self.pop.append(o2)

    # 淘汰（親個体の選択）
    def selectParent(self):
        if self.cnf.selection == "RW":
            return self.RWselection(), self.RWselection()
        elif self.cnf.selection == "TS":
            return self.TSselection(), self.TSselection()
        else:
            print("Error in selectParent")

    # ルーレット選択
    def RWselection(self):
        Fsum = 0.
        for i in range(len(self.pop)):
            Fsum += self.pop[i].f
        choice = Fsum * self.cnf.rd.rand()
        Fsum = 0.
        for i in range(len(self.pop)):
            Fsum += self.pop[i].f
            if Fsum >= choice:
                return self.pop[i]
        return None

    # トーナメント選択（トーナメントサイズ Nt=1）
    def TSselection(self):
        i, j = self.cnf.rd.randint(0, len(self.pop)), self.cnf.rd.randint(0, len(self.pop))

        if self.pop[i].f < self.pop[j].f:
            return self.pop[i]
        else:
            return self.pop[j]

    # エリート戦略
    def getElite(self):
        for i in range(len(self.pop)):
            for j in range(len(self.pop)):
                if i != j and self.pop[i].f > self.pop[j].f:
                    tmp = self.pop[i]
                    self.pop[i] = self.pop[j]
                    self.pop[j] = tmp
        cnt = (int)(len(self.pop) * self.cnf.p_elite)
        for i in range(cnt):
            self.pop.pop(0)

    # 子個体の生成
    def generateOffspring(self, p1, p2):
        o1, o2 = self.applyXover(Solution(self.cnf, self.fnc, parent=p1), Solution(self.cnf, self.fnc, parent=p2))
        o1.mutation()
        o2.mutation()
        return o1, o2

    # 交叉（一点交叉）
    def applyXover(self, o1, o2):
        if self.cnf.rd.rand() >= self.cnf.p_x:
            return o1, o2
        xpoint = self.cnf.rd.randint(0, self.cnf.prob_dim)
        for i in range(xpoint):
            tmp = o1.x[i]
            o1.x[i] = o2.x[i]
            o2.x[i] = tmp
        return o1, o2

    # 評価値fの計算
    def getFitness(self, solution):
        solution.f = self.fnc.doEvaluate(solution.x)


#個体のクラス
class Solution:
    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc, parent=None):
        self.cnf, self.fnc, self.x, self.f = cnf, fnc, [], 0.
        # 個体の初期化
        if parent == None:
            self.x = [self.cnf.rd.uniform(self.fnc.axis_range[0], self.fnc.axis_range[1]) for i in range(self.cnf.prob_dim)]
        # 親個体のコピー
        else:
            self.x = [parent.x[i] for i in range(self.cnf.prob_dim)]
        # リスト -> ndarray
        self.x = np.array(self.x)

    """ インスタンスメソッド """
    # 突然変異（摂動）
    def mutation(self):
        dx = 0.05   # 摂動の微小量
        for i in range(self.cnf.prob_dim):
            if self.cnf.rd.rand() < self.cnf.p_m :
                # 摂動
                self.x[i] += self.cnf.rd.uniform(-dx,dx) * (self.fnc.axis_range[1] - self.fnc.axis_range[0])
                # 定義域外の探索防止
                self.x[i] = np.clip(self.x[i], self.fnc.axis_range[0], self.fnc.axis_range[1])
