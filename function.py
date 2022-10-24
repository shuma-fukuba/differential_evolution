###### function.py #####
# Last Update:  2020/4/21
#
# 関数設定用ファイル
# インスタンスはfncとして生成

import numpy as np

from error import FunctionError


class Function:

    def __init__(self, prob_name, prob_dimension):
        self.prob_name = prob_name  # 関数名
        self.prob_dimension = prob_dimension  # 次元数
        self.axis_range = []  # 各軸の定義域
        self.evaluate = None  # 代替関数
        self.total_evals = 0  # 総評価回数

        # 関数名分岐処理
        if self.prob_name == "F1":
            self.evaluate = self.F1
            self.axis_range = [-100, 100]
        elif self.prob_name == "F5":
            self.evaluate = self.F5
            self.axis_range = [-100, 100]
        else:  # 関数不在時
            raise FunctionError(f"Error: Do not exist Function {prob_name} (function.py)")

        # 関数名出力
        print(f"\t\t[ Problem : {prob_name} ]\t\t")

    # 評価実行
    def do_evaluate(self, x):
        self.total_evals += 1
        return self.evaluate(x)

    # 評価回数リセット
    def reset_total_evals(self):
        self.total_evals = 0

    # F1 Sphere : f(x) = Σ(x[i]^2)
    def F1(self, x):
        if not len(x) == self.prob_dimension:
            raise FunctionError(f"Error: Solution X is not a {self.prob_dimension}-d vector")
        ret = np.sum([item ** 2 for item in x])
        return ret

    # F5 Griewank
    def F5(self, x):
        if not len(x) == self.prob_dimension:
            raise FunctionError(f"Error: Solution X is not a {self.prob_dimension}-d vector")
        sum_1 = 0.
        prod_1 = 1.
        for i in range(self.prob_dimension):
            sum_1 += x[i] * x[i]
            prod_1 *= np.cos(x[i] / np.sqrt(i + 1))
        ret = 1 - prod_1 + sum_1 / 4000
        return ret
