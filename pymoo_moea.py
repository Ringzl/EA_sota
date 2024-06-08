import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.optimize import minimize


from pymoo.problems.many.wfg import WFG1, WFG4
from pymoo.indicators.hv import HV

from pymoo.indicators.igd import IGD



if __name__ == "__main__":
    problem = WFG1(n_var=10, n_obj=3)
    alg = NSGA2(pop_size=100)

    ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
    alg = MOEAD(
        ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7
    )

    # ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    # alg = RVEA(ref_dirs)
    
    res = minimize(problem,
                   alg,
                   ('n_gen', 1000),
                   seed=1,
                   verbose=False)
    
    refpoint = np.array([2 * i + 1 for i in range(1, 3+1)])
    ind = HV(ref_point=refpoint)
    
    igd = IGD(problem.pareto_front())
    print(igd(res.F))
    print(ind(res.F)/np.prod(refpoint))
    