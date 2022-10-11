###### main.py #####
#                                           Last Update:  2020/4/13
#
# プログラム実行用ファイル
# F5キーで実行

# 他ファイルのインポート
import numpy            as np
import configuration    as cf
import function         as fc
import optimizer        as op
import logger           as lg

# ---------- logger.py Loggerクラスのメソッド ----------
#  ・loggging(pop, total_evals)
#     [total_evals, pop.x, pop.f] を配列で格納
#  ・outlog(pop, total_evals)
#     格納した情報をcsv出力し，最終的な解の評価値を標準出力
# -----------------------------------------------------

# 実行処理
def run(opt, cnf, fnc, log):
    opt.initializeSolutions()                   # 初期化
    log.logging(opt.pop, fnc.total_evals)       # 初期個体群ログ
    while fnc.total_evals < cnf.max_evals:      # 評価回数上限まで実行
        opt.getNextPopulation()                 # 次世代個体群生成
        log.logging(opt.pop, fnc.total_evals)   # 次世代個体群ログ
    log.outLog(opt.pop, fnc.total_evals)        # ログ出力(trial'n'.csv)


""" main (最初に実行) """
if __name__ == '__main__':
    cnf = cf.Configuration()                                        # configurationインスタンス生成
    for i in range(len(cnf.prob_name)):                             # 関数の個数だけ探索
        log = lg.Logger(cnf, cnf.prob_name[i])                      # loggerインスタンス生成
        fnc = fc.Function(cnf.prob_name[i], cnf.prob_dim)           # 探索する関数のインスタンス生成
        for j in range(cnf.max_trial):                              # 試行回数まで実行
            fnc.resetTotalEvals()                                   # 総評価回数(functionクラス内変数)リセット
            cnf.setRandomSeed(seed=j+1)                             # ランダムシード値設定
            opt = op.GeneticAlgorithm(cnf, fnc)                     # optimizerインスタンス生成
            run(opt, cnf, fnc, log)                                 # 探索実行
        sts = lg.Statistics(cnf, fnc, log.path_out, log.path_trial) # 関数ごとの統計を作成
        sts.outStatistics()                                         # 統計出力(all_trials.csv, statistics.csv)