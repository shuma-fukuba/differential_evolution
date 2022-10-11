###### function.py #####
#                                           Last Update:  2020/4/21
#
# 関数設定用ファイル
# インスタンスはfncとして生成

# 他ファイル,モジュールのインポート
import sys
import math  as mt
import numpy as np

# functionクラス
class Function:

    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, prob_name, prob_dim):
        self.prob_name   = prob_name    # 関数名
        self.prob_dim    = prob_dim     # 次元数
        self.axis_range  = []           # 各軸の定義域
        self.evaluate    = None         # 代替関数
        self.total_evals = 0            # 総評価回数

        # 関数名分岐処理
        if self.prob_name == "F1":
            self.evaluate = self.F1
            self.axis_range  = [-100,100]
        elif self.prob_name == "F5":
            self.evaluate = self.F5
            self.axis_range  = [-100, 100]
        else: # 関数不在時
            print("Error: Do not exist Function {} (function.py)".format(prob_name))
            sys.exit()

        # 関数名出力
        print("\t\t[ Problem : {} ]\t\t".format(prob_name))

    """ インスタンスメソッド """
    # 評価実行
    def doEvaluate(self, x):
        self.total_evals += 1
        return self.evaluate(x)

    # 評価回数リセット
    def resetTotalEvals(self):
        self.total_evals = 0

    # F1 Sphere : f(x) = Σ(x[i]^2)
    def F1(self, x):
        if not len(x) == self.prob_dim:
            print("Error: Solution X is not a {}-d vector".format(self.prob_dim))
            return None
        ret = np.sum(x**2)
        return ret

    # F5 Griewank
    def F5(self, x):
        if not len(x) == self.prob_dim:
            print("Error: Solution X is not a {}-d vector".format(self.prob_dim))
            return None
        sum_1 = 0.
        prod_1 = 1.
        for i in range(self.prob_dim):
            sum_1 += x[i] * x[i]
            prod_1 *= np.cos(x[i] / np.sqrt(i + 1))
        ret = 1 - prod_1 + sum_1 / 4000
        return ret

