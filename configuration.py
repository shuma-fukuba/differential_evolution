###### configuration.py #####
#                                           Last Update:  2020/4/13
#
# 各種設定用ファイル
# インスタンスはcnfとして生成

import numpy as np

# configurationクラス
class Configuration:

    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self):

        # 入出力設定
        self.path_out   = "./"                  # 出力先フォルダ
        self.log_name   = "_result_" + "GA"     # ログの出力先フォルダ(path_outの直下)
        self.log_out    = True                  # ログ出力の有無

        # GAの設定
        self.max_pop    = 50                    # 個体数
        #self.max_gen    = 600                   # 最大世代数(今回はmax_evalsで制限)
        self.p_x        = 0.8                   # 交叉率
        self.p_m        = 0.04                  # 突然変異率
        self.p_elite    = 0.5                   # 淘汰率
        self.selection  = "RW"                  # 選択の方法

        # 問題設定
        self.prob_dim  = 20                     # 問題の次元数
        self.prob_name = ["F1","F5"]            # 解く問題

        # 実験環境
        self.max_trial = 30                     # 試行回数
        self.max_evals = 100000                 # 評価回数(max_pop × max_gen)

    """ インスタンスメソッド """
    # ランダムシード値設定
    def setRandomSeed(self, seed=1):
        # シード値の固定
        self.seed = seed
        self.rd = np.random
        self.rd.seed(self.seed)
