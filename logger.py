###### logger.py #####
#                                           Last Update:  2020/4/13
#
# ログ取得用ファイル
# インスタンスはlogとして生成

# 他ファイル,モジュールのインポート
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configuration import Configuration
from function import Function

""" データロガー """
# [出力] trialsフォルダ（trialX.csv）・標準出力
# loggerクラス


class Logger:

    def __init__(self, config: Configuration, prob_name):
        self.answer: int
        self.dat = []  # dat:データ一時保管場所
        self.config = config
        # パスの設定
        self.path_out = config.path_out
        self.path_out += '/{0}/{1}/'.format(self.config.log_name, prob_name)
        self.path_trial = self.path_out + 'trials'

        if self.config.log_out:
            if not os.path.isdir(self.path_trial):  # trialディレクトリがなければ作成
                os.makedirs(self.path_trial)

    # 個体群ログ取得

    def logging(self, pop, evals, do_console=False):
        # 最良個体番号取得
        self.answer = 0
        for i, v in enumerate(pop):
            if v.f < pop[self.answer].f:
                self.answer = i
        # 最良個体のコンソール表示
        if do_console:
            print([evals, pop[self.answer].f])
        sls = [evals, pop[self.answer].f]  # 最良個体情報(evals, f)
        sls.extend(pop[self.answer].x)  # 最良個体情報(x)
        self.dat.append(sls)  # データ合体

    # csvファイル出力
    def out_log(self, pop, evals):
        if self.config.log_out:
            # ヘッダー作成
            head = "evals,fx," + \
                ','.join(["x{}".format(i) for i in range(self.config.prob_dimension)])
            np.savetxt(self.path_trial + '/trial{}.csv'.format(self.config.seed),
                       np.array(self.dat), delimiter=',', header=head)  # 出力
        print(" trial: {:03}\tevals: {}\tfx: {}".format(
            self.config.seed, evals, pop[self.answer].f))  # コンソール表示
        self.dat = []  # 一時データ解放


""" データ処理 """
# [出力] all_trials.csv・statics_Fn.csv・statics_Fn.png
# statisticsクラス


class Statistics:
    def __init__(self, config: Configuration, function: Function, path_out, path_dat):
        self.path_out = path_out
        self.path_dat = path_dat
        self.config = config
        self.function = function

    """ インスタンスメソッド """
    # 統計作成

    def out_statistics(self):
        if self.config.log_out:
            # csvファイルの読み取り 全試行一覧表示
            df = None
            for i in range(self.config.max_trial):
                dat = pd.read_csv(
                    self.path_dat+'/trial{}.csv'.format(i+1), index_col=0)
                if i == 0:
                    df = pd.DataFrame(
                        {'trial{}'.format(i+1): np.array(dat['fx'])}, index=dat.index)
                else:
                    df['trial{}'.format(i+1)] = np.array(dat['fx'])
            df.to_csv(self.path_out + "all_trials.csv")

            # 統計処理（最小値・最大値・四分位数・平均）
            _min, _max, _q25, _med, _q75, _ave = [], [], [], [], [], []
            for i, idx in enumerate(df.index):
                dat = np.array(df.loc[idx])
                res = np.percentile(dat, [25, 50, 75])
                _min.append(dat.min())
                _max.append(dat.max())
                _q25.append(res[0])
                _med.append(res[1])
                _q75.append(res[2])
                _ave.append(dat.mean())

            # データフレームの作成
            _out = pd.DataFrame({
                'min': np.array(_min),
                'q25': np.array(_q25),
                'med': np.array(_med),
                'q75': np.array(_q75),
                'max': np.array(_max),
                'ave': np.array(_ave)
            }, index=df.index)

            # csv出力
            _out.to_csv(self.path_out + "statistics_" +
                        self.function.prob_name + ".csv")

            # 図の作成
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(_out.index, _med, color='orange',
                    linestyle='solid', label=self.config.log_name)
            ax.fill_between(_out.index, _q25, _q75,
                            facecolor='orange', alpha=0.1)
            ax.set_ylabel('F(x)')
            ax.set_xlabel('Evaluations')
            ax.set_xlim([0, self.config.max_evals])

            # png出力
            fig.savefig(self.path_out + "statistics_" +
                        self.function.prob_name + ".png", dpi=300)

            # グラフのクローズ
            plt.close()
