###### main.py #####
# Last Update:  2020/4/13
#
# プログラム実行用ファイル
# F5キーで実行

# 他ファイルのインポート
from configuration import Configuration
from differential_evolution import DifferentialEvolution
from function import Function
from optimizer import GeneticAlgorithm
from logger import Logger, Statistics

# ---------- logger.py Loggerクラスのメソッド ----------
#  ・logging(pop, total_evals)
#     [total_evals, pop.x, pop.f] を配列で格納
#  ・outlog(pop, total_evals)
#     格納した情報をcsv出力し，最終的な解の評価値を標準出力
# -----------------------------------------------------

# 実行処理


def run(optimizer: GeneticAlgorithm,
        config: Configuration,
        function: Function,
        log: Logger) -> None:
    optimizer.initialize_solutions()  #  初期化
    log.logging(optimizer.solutions, function.total_evals)  # 初期個体群ログ
    optimizer.optimize(log)  # 次世代個体群生成
    log.logging(optimizer.solutions, function.total_evals)  # 次世代個体群ログ
    log.out_log(optimizer.solutions, function.total_evals)  # ログ出力(trial'n'.csv)


if __name__ == '__main__':
    # configurationインスタンス生成
    config = Configuration()
    for i, name in enumerate(config.prob_name):  # 関数の個数だけ探索
        # loggerインスタンス生成
        log = Logger(config, name)
        # 探索する関数のインスタンス生成
        function = Function(name, config.prob_dimension)
        for j in range(config.max_trial):  # 試行回数まで実行
            # 総評価回数(functionクラス内変数)リセット
            function.reset_total_evals()
            # ランダムシード値設定
            config.set_random_seed(seed=j+1)
            # optimizerインスタンス生成
            # optimizer = GeneticAlgorithm(config, function)
            optimizer = DifferentialEvolution(config=config, function=function)
            # 探索実行
            run(optimizer, config, function, log)
        statistics = Statistics(config, function, log.path_out, log.path_trial)  # 関数ごとの統計を作成
        # 統計出力(all_trials.csv, statistics.csv)
        statistics.out_statistics()
