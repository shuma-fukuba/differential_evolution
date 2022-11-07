"""
import parent class and other classes being necessary to annotation
"""
import random
import math
import numpy as np
from optimizer import GeneticAlgorithm, Solution
from configuration import Configuration
from function import Function
from logger import Logger


class DifferentialEvolution(GeneticAlgorithm):
    def __init__(self, config: Configuration, function: Function) -> None:
        super().__init__(config, function)
        '''JADEの初期値'''
        self.m_cr = 0.5
        self.m_f = 0.5
        self.archive = []
        self.C = 0.5  # パラメータ調整

    def optimize(self, log: Logger):
        s_f = []
        s_cr = []
        while self.function.total_evals < self.config.max_evals:
            self.get_next_population(s_f, s_cr)
            log.logging(self.solutions, self.function.total_evals)

    # 次世代個体群生成
    def get_next_population(self, s_f, s_cr):
        for i, solution in enumerate(self.solutions):
            cr_i = self.config.rd.normal(loc=self.m_cr, scale=0.1)
            pbest_solution = self.get_pbest_solution()
            g1_solution = self.get_random_solution_g1(
                pbest_solution=pbest_solution)
            g2_solution = self.get_random_solution_g2(pbest_solution=pbest_solution,
                                                      g1_solution=g1_solution)
            f_i = self.get_f_i()
            #　 TODO: 突然変異
            mutated_solution = self.jade_mutation(current_solution=solution,
                                                  pbest_solution=pbest_solution,
                                                  g1_solution=g1_solution,
                                                  g2_solution=g2_solution,
                                                  f_i=f_i)

            # 交叉
            new_solution = self.crossover(solution, mutated_solution)

            # 選択
            # TODO: 交叉後の評価値が良ければ、パラメータを更新
            if new_solution.f <= solution.f:
                self.solutions.remove(solution)
                self.solutions.append(new_solution)
                self.archive.append(solution)

                s_cr.append(cr_i)
                s_f.append(f_i)
            # xi, g + 1 = 交叉後
            # 交叉前の個体をアーカイブに投入
            # criをs_ｃｒに、Fiをs_fに入れる
        self.randomly_delete_from_archive()
        self.m_cr = (1 - self.C) * self.m_cr + self.C * np.mean(s_cr)
        self.m_f = (1 - self.C) * self.m_f + self.C * \
            (np.sum(np.array(s_f) ** 2) / sum(s_f))

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

        # 定義域外の探索防止
        mutated.prevent_out_dimension_search()
        self.get_fitness(mutated)
        return mutated

    def jade_mutation(self, current_solution: Solution,
                      pbest_solution: Solution,
                      g1_solution: Solution,
                      g2_solution: Solution,
                      f_i) -> Solution:
        # F_iはコーシー分布に従って生成される。1以上なら切り捨てられ、0以下なら再生される
        mutated = Solution(config=self.config, function=self.function)
        mutated.x = current_solution.x \
            + f_i * (np.array(pbest_solution.x) - np.array(current_solution.x)) \
            + f_i * (np.array(g1_solution.x) - np.array(g2_solution.x))
        self.get_fitness(mutated)
        return mutated

    def crossover(self, solution_1: Solution, solution_2: Solution) -> Solution:
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

    def get_best_solution(self) -> Solution:
        return min(self.solutions, key=lambda x: x.f)

    def get_f_i(self):
        u = self.config.rd.random()
        f_i = self.m_f + 0.1 * (math.tan(math.pi * (u - 0.5)))
        if f_i <= 0:
            return self.get_f_i()
        elif f_i >= 1:
            return math.floor(f_i)
        return f_i
        # return random.gauss(self.m_f, 0.1)

    def get_pbest_solution(self):
        p = self.config.rd.random()  # 0 <= p <= 1
        pbest_nums = math.ceil(len(self.solutions) * p)
        pbests = sorted(self.solutions, key=lambda x: x.f)[:pbest_nums]
        return random.choice(pbests)

    def get_random_solution_g1(self, pbest_solution: Solution):
        merged_solutions = list(set(self.solutions) | set(self.archive))
        g1_solution = self.config.rd.choice(merged_solutions)
        if g1_solution == pbest_solution:
            return self.get_random_solution_g1(pbest_solution=pbest_solution)
        else:
            return g1_solution

    def get_random_solution_g2(self, pbest_solution: Solution, g1_solution: Solution) -> Solution:
        merged_solutions = list(set(self.solutions) | set(self.archive))
        g2_solution = self.config.rd.choice(merged_solutions)
        if g2_solution == pbest_solution or g2_solution == g1_solution:
            return self.get_random_solution_g2(pbest_solution, g1_solution)
        else:
            return g2_solution

    def randomly_delete_from_archive(self):
        solutions_len = len(self.solutions)
        archive_len = len(self.archive)
        if solutions_len < archive_len:
            delete_num = archive_len - solutions_len
            for _ in range(delete_num):
                self.archive.remove(self.config.rd.choice(self.archive))
