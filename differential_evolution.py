"""
import parent class and other classes being necessary to annotation
"""
from optimizer import GeneticAlgorithm, Solution
from configuration import Configuration
from function import Function


class DifferentialEvolution(GeneticAlgorithm):
    """differential evolution"""

    def __init__(self, config: Configuration, function: Function) -> None:
        super().__init__(config, function)

    def apply_x_over(self, solution_1, solution_2):
        if self.config.rd.rand() >= self.config.p_x:
            return solution_1, solution_2
        x_point = self.config.rd.randint(0, self.config.prob_name)
        for i in range(x_point):
            solution_1.x[i], solution_2.x[i] = \
                solution_2.x[i], solution_1.x[i]
        return solution_1, solution_2

    def generate_offspring(self):
        """
        solutions全体から適当な3つのsolutionを選び取る(solution_1, solution_2, solution_3)
        mutated_solution_1 = self.mutation(solution_1,)
        これをすべての個体に対して行う
        個体群はself.popにあるから、ここからランダムに個体を選び取る
        """
        pass

    def mutation(self,
                 target_solution: Solution,
                 solution_2: Solution,
                 solution_3: Solution) -> Solution:
        """選別した3つの個体を使って突然変異させる(パラメータはxのハズ...)"""
        self.get_fitness(self, target_solution)
        target_solution.x += \
            target_solution.f * (solution_2.x - solution_3.x)
        return target_solution

    def differential_crossing_over(self):
        """"""
        pass