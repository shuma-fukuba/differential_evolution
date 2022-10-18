"""
import parent class and other classes being necessary to annotation
"""
from random import paretovariate
from optimizer import GeneticAlgorithm, Solution


class DifferentialEvolution(GeneticAlgorithm):
    """differential evolution"""

    def get_next_population(self):
        self.get_elite()
            # 突然変異
        for i, solution in enumerate(self.solutions):
            """
            突然変異：iではない3個体をランダムに選択
            交叉：iとmutationを交叉
            """
            # 突然変異
            solution_1, solution_2, solution_3 = \
                self.config.rd.sample(
                    [j for j, _ in enumerate(self.solutions) if j != i], 3)
            mutated_solution = self.mutation(
                solution_1, solution_2, solution_3)
            
            # 交叉
            new_solution = self.crossover(solution, mutated_solution)
            # 選択
            self.evaluate_offspring(offspring=new_solution, parent=solution)
        # TODO: debug 10/18


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
        return child_solution

    def evaluate_offspring(self, offspring: Solution, parent: Solution):
        """子要素と親要素の評価値を比較し、優れているほうを残す"""
        if offspring.f <= parent.f:
            # 親を削除
            self.solutions.append(offspring)
            self.solutions.remove(parent)
        # 親が優れている場合は何もしないでok