"""
import parent class and other classes being necessary to annotation
"""
import random
import numpy as np
from optimizer import GeneticAlgorithm, Solution


class DifferentialEvolution(GeneticAlgorithm):
    """differential evolution"""

    # 次世代個体群生成
    def get_next_population(self):
        # self.get_elite()
        # 突然変異
        for i, solution in enumerate(self.solutions):
            """
            突然変異：iではない3個体をランダムに選択
            交叉：iとmutationを交叉
            """
            # 突然変異
            # solution_1, solution_2, solution_3 = self.select_parent()
            """
            solutionに対して突然変異を起こさせる
            任意の個体を２つ選んで、new solutionを更新させる
            """
            # best1で取ってみる
            best_solution = self.get_best_solution()
            random_parent_1, random_parent_2 = random.sample(self.solutions, 2)
            mutated_solution = self.mutation(target_solution=best_solution,
                                             solution_2=random_parent_1,
                                             solution_3=random_parent_2)

            # 交叉
            new_solution = self.crossover(solution, mutated_solution)
            # 選択
            self.evaluate_offspring(offspring=new_solution, parent=solution)

    def mutation(self,
                 target_solution: Solution,
                 solution_2: Solution,
                 solution_3: Solution) -> Solution:
        """選別した3つの個体を使って突然変異させる
        """
        self.get_fitness(target_solution)
        mutated = Solution(config=self.config, function=self.function)

        mutated.f = target_solution.f
        mutated.x = target_solution.x + self.config.F * \
            (np.array(solution_2.x) - np.array(solution_3.x))
        self.get_fitness(mutated)
        return mutated

    def crossover(self, solution_1: Solution, solution_2: Solution) -> Solution:
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
        self.get_fitness(child_solution)
        return child_solution

    def evaluate_offspring(self, offspring: Solution, parent: Solution):
        """子要素と親要素の評価値を比較し、優れているほうを残す"""
        if offspring.f <= parent.f:
            # 親を削除
            self.solutions.append(offspring)
            self.solutions.remove(parent)
        # 親が優れている場合は何もしないでok
