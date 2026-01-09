from .algorithms.classic_pso import classic_pso
from .algorithms.parallel_jax_pso import parallel_jax_pso
from .algorithms.vectorized_pso import vectorized_pso
from .benchmarks.ackley import ACKLEY_BOUNDS, ackley_fn
from .benchmarks.alpine import ALPINE_BOUNDS, alpine_fn
from .benchmarks.griewank import GRIEWANK_BOUNDS, griewank_fn
from .benchmarks.rastrigin import RASTRIGIN_BOUNDS, rastrigin_fn
from .benchmarks.rosenbrock import ROSENBROCK_BOUNDS, rosenbrock_fn
from .benchmarks.salomon import SALOMON_BOUNDS, salomon_fn
from .benchmarks.schwefel import SCHWEFEL_BOUNDS, schwefel_fn
from .benchmarks.sphere import SPHERE_BOUNDS, sphere_fn
from .benchmarks.styblinski_tang import STYBLINSKI_TANG_BOUNDS, styblinski_tang_fn

config = {
    'dims': [2, 10, 30, 50, 100, 250, 500, 750, 1000],
    'params': {
        'num_dims': None,
        'num_particles': 100, # 100
        'max_iters': 500, # 500
        'c1': 2,
        'c2': 2,
        'w': 0.7,
        'seed': 42,
    },
    'num_subswarms': 10,
    'num_runs': 10,
    'algorithms': {
        'Pure PSO': classic_pso,
        'Vectorized PSO': vectorized_pso,
        'JAX PSO': parallel_jax_pso,
    },
    'benchmarks': {
        'Ackley': (ackley_fn, ACKLEY_BOUNDS),
        'Alpine': (alpine_fn, ALPINE_BOUNDS),
        'Griewank': (griewank_fn, GRIEWANK_BOUNDS),
        'Rastrigin': (rastrigin_fn, RASTRIGIN_BOUNDS),
        'Rosenbrock': (rosenbrock_fn, ROSENBROCK_BOUNDS),
        'Salomon': (salomon_fn, SALOMON_BOUNDS),
        'Schwefel': (schwefel_fn, SCHWEFEL_BOUNDS),
        'Sphere': (sphere_fn, SPHERE_BOUNDS),
        'Styblinski-Tang': (styblinski_tang_fn, STYBLINSKI_TANG_BOUNDS),
    },
}
