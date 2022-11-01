from differential_evolution import DifferentialEvolution


class Jade(DifferentialEvolution):
    def get_next_population(self):
        '''
        パラメータを更新していく
        初期値は設定しておいて、
        '''