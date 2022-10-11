###### logger.py #####
#                                           Last Update:  2020/4/13
#
# ログ取得用ファイル
# インスタンスはlogとして生成

# 他ファイル,モジュールのインポート
import numpy                as np
import os
import pandas               as pd
import matplotlib.pyplot    as plt

""" データロガー """
# [出力] trialsフォルダ（trialX.csv）・標準出力
# loggerクラス
class Logger:
        
    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, prob_name):
        self.dat, self.cnf = [], cnf    # dat:データ一時保管場所
        # パスの設定
        self.path_out = cnf.path_out
        self.path_out += '/{0}/{1}/'.format(self.cnf.log_name, prob_name)
        self.path_trial = self.path_out + 'trials'

        if self.cnf.log_out:
            if not os.path.isdir(self.path_trial): # trialディレクトリがなければ作成
                os.makedirs(self.path_trial)

    """ インスタンスメソッド """
    # 個体群ログ取得
    def logging(self, pop, evals, do_console = False):
        # 最良個体番号取得
        self.ans = 0
        for i in range(len(pop)):
            if pop[i].f < pop[self.ans].f:
                self.ans = i
        # 最良個体のコンソール表示
        if do_console:
            print([evals, pop[self.ans].f])
        sls = [evals, pop[self.ans].f]          # 最良個体情報(evals, f)
        sls.extend(pop[self.ans].x)             # 最良個体情報(x)
        self.dat.append(sls)                    # データ合体

    # csvファイル出力
    def outLog(self, pop, evals):
        if self.cnf.log_out:
            head = "evals,fx," + ','.join(["x{}".format(i) for i in range(self.cnf.prob_dim)])                                  # ヘッダー作成
            np.savetxt(self.path_trial +'/trial{}.csv'.format(self.cnf.seed), np.array(self.dat), delimiter=',', header = head) # 出力
        print(" trial: {:03}\tevals: {}\tfx: {}".format(self.cnf.seed, evals, pop[self.ans].f))                               # コンソール表示
        self.dat = []                                    # 一時データ解放


""" データ処理 """
# [出力] all_trials.csv・statics_Fn.csv・statics_Fn.png
# statisticsクラス    
class Statistics:
    def __init__(self, cnf, fnc, path_out, path_dat):
        self.path_out = path_out
        self.path_dat = path_dat
        self.cnf      = cnf
        self.fnc      = fnc

    """ インスタンスメソッド """
    # 統計作成
    def outStatistics(self):
        if self.cnf.log_out:
            # csvファイルの読み取り 全試行一覧表示
            df = None
            for i in range(self.cnf.max_trial):
                dat = pd.read_csv(self.path_dat+'/trial{}.csv'.format(i+1), index_col = 0)
                if i == 0:
                    df = pd.DataFrame({'trial{}'.format(i+1) : np.array(dat['fx'])}, index = dat.index)
                else:
                    df['trial{}'.format(i+1)] = np.array(dat['fx'])
            df.to_csv(self.path_out + "all_trials.csv")

            # 統計処理（最小値・最大値・四分位数・平均）
            _min, _max, _q25, _med, _q75, _ave = [], [], [], [], [], []
            for i in range(len(df.index)):
                dat = np.array(df.loc[df.index[i]])
                res = np.percentile(dat, [25, 50, 75])
                _min.append(dat.min())
                _max.append(dat.max())
                _q25.append(res[0])
                _med.append(res[1])
                _q75.append(res[2])
                _ave.append(dat.mean())

            # データフレームの作成
            _out = pd.DataFrame({
                'min' : np.array(_min),
                'q25' : np.array(_q25),
                'med' : np.array(_med),
                'q75' : np.array(_q75),
                'max' : np.array(_max),
                'ave' : np.array(_ave)
                },index = df.index)

            # csv出力
            _out.to_csv(self.path_out + "statistics_" + self.fnc.prob_name + ".csv")

            # 図の作成
            fig = plt.figure(figsize=(10, 4))
            ax  = fig.add_subplot(1,1,1)
            ax.plot(_out.index, _med , color='orange',  linestyle='solid', label=self.cnf.log_name)
            ax.fill_between(_out.index, _q25, _q75, facecolor='orange', alpha=0.1)
            ax.set_ylabel('F(x)')
            ax.set_xlabel('Evaluations')
            ax.set_xlim([0,self.cnf.max_evals])

            # png出力
            fig.savefig(self.path_out + "statistics_" + self.fnc.prob_name + ".png" , dpi=300)

            # グラフのクローズ
            plt.close()
