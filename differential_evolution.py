"""
import parent class and other classes being necessary to annotation
"""
from random import paretovariate
from optimizer import GeneticAlgorithm, Solution


class DifferentialEvolution(GeneticAlgorithm):
    """differential evolution"""

    def get_next_population(self):
        self.get_elite()
        while len(self.solutions) < self.config.max_pop:
            # 突然変異
            for target in self.solutions:
                parent_1, parent_2 = self.select_parent()
                offspring = self.mutation(target, parent_1, parent_2)
                # 交叉
                offspring = self.crossover(offspring, parent_1)  # NOTE: parent_1でいいか確認
                self.get_fitness(offspring)
                superior = self.evaluate_offspring(offspring=offspring, parent=parent_1)
                self.solutions.append(superior)
        # TODO: すべての要素に対し交叉と突然変異を適用したら選択アルゴリズム

    def mutation(self,
                 target_solution: Solution,
                 solution_2: Solution,
                 solution_3: Solution) -> Solution:
        """選別した3つの個体を使って突然変異させる(パラメータはxのハズ...)
        """
        self.get_fitness(self, target_solution)
        target_solution.x += \
            self.config.F * (solution_2.x - solution_3.x)
        return target_solution

    def crossover(self, solution_1: Solution, solution_2: Solution)-> Solution:
        """
        2つのsolution v, xを受け取る
        乱数を生成(0以上1以下)
        CRとは交叉率、configurationクラスにある
        乱数がCR以下もしくはj_rand=jのとき xの要素を採用(0 <= j_rand <= np.rand.randint(0, len(pop)))
        それ以外はvの要素を採用
        こうして新しいsolutionを生成する
        最終的にvを生成した子個体で書き換え、親個体xと比較して評価する
        """
        # 0から19までの整数乱数を生成
        # config.prob_dimensionはlen(solution.x)に同じ
        j_rand = self.config.rd.randint(self.config.prob_dimension)
        new_solution = []
        for j in range(self.config.prob_dimension):
            cr = self.config.rd.rand()
            if cr <= self.config.p_x or j == j_rand:
                new_solution.append(solution_2.x[j])
            else:
                new_solution.append(solution_1.x[j])

        child_solution = Solution(config=self.config, function=self.function)
        child_solution.x = new_solution
        return child_solution

    def evaluate_offspring(self, offspring: Solution, parent: Solution):
        """子要素と親要素の評価値を比較し、優れているほうを返す"""
        if offspring.f <= parent.f:
            # 親を削除
            return offspring
        else:
            return parent