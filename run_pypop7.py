import numpy as np  # for numerical computation, which is also the computing engine of pypop7

# 2. Define your own objective/cost function for the optimization problem at hand:
#   the below example is Rosenbrock, the notorious test function from the optimization community
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

# define the fitness (cost) function and also its settings
ndim_problem = 4
problem = {'fitness_function': func,  # cost function
           'ndim_problem': ndim_problem,  # dimension
           'lower_boundary': -100.0*np.ones((ndim_problem,)),  # search boundary
           'upper_boundary': 100.0*np.ones((ndim_problem,))}

# 3. Run one or more black-box optimizers on the given optimization problem:
#   here we choose LM-MA-ES owing to its low complexity and metric-learning ability for LSO
#   https://pypop.readthedocs.io/en/latest/es/lmmaes.html
# from pypop7.optimizers.es.lmmaes import LMMAES
from pypop7.optimizers.de.shade import SHADE

# define all the necessary algorithm options (which differ among different optimizers)
options = {'fitness_threshold': 1e-10,  # terminate when the best-so-far fitness is lower than this threshold
           'max_runtime': 3600,  # 1 hours (terminate when the actual runtime exceeds it)
           'seed_rng': 0,  # seed of random number generation (which must be explicitly set for repeatability)
           'x': 4.0*np.ones((ndim_problem,)),  # initial mean of search (mutation/sampling) distribution
           'sigma': 3.0,  # initial global step-size of search distribution (not necessarily optimal)
           'verbose': 500}

lmmaes = SHADE(problem, options)  # initialize the optimizer
results = lmmaes.optimize()  # run its (time-consuming) search process
print(results)