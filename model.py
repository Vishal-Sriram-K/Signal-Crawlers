import numpy as np

class IntersectionModel:
    def __init__(
        self,
        lambdas,
        saturation_flows,
        C,
        L,
        g_min=5.0,
        g_max=None,
    ):
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.s = np.asarray(saturation_flows, dtype=float)
        assert self.lambdas.shape == (4,)
        assert self.s.shape == (4,)

        self.C = float(C)
        self.L = float(L)
        self.G_total = self.C - self.L

        if np.isscalar(g_min):
            self.g_min = np.full(4, float(g_min))
        else:
            self.g_min = np.asarray(g_min, dtype=float)

        if g_max is None:
            self.g_max = np.full(4, 0.8 * self.G_total)
        else:
            if np.isscalar(g_max):
                self.g_max = np.full(4, float(g_max))
            else:
                self.g_max = np.asarray(g_max, dtype=float)

        self.eps = 1e-6
    
    def total_delay(self, g):

        g = np.asarray(g, dtype=float)

        g = np.clip(g, self.g_min, self.g_max)

        mu = self.s * (g / self.C)

        slack = np.maximum(mu - self.lambdas, self.eps)

        Wq = self.lambdas / slack
        D = self.lambdas * Wq

        return float(np.sum(D))

    def capacity_slack(self, g):

        g = np.asarray(g, dtype=float)
        mu = self.s * (g / self.C)
        return mu - self.lambdas
    
    def random_feasible_g(self, seed=None):
        rng = np.random.default_rng(seed)
        g = np.full(4, self.G_total / 4.0)
        g += rng.normal(scale=1.0, size=4)
        g = project_onto_simplex_with_bounds(
            g,
            total=self.G_total,
            lower=self.g_min,
            upper=self.g_max,
        )
        return g
    
def project_onto_simplex_with_bounds(x, total, lower, upper, max_iter=10):
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    g = np.clip(x, lower, upper)

    for _ in range(max_iter):
        current_sum = np.sum(g)
        diff = total - current_sum
        if abs(diff) < 1e-8:
            break
        if diff > 0:
            free = g < upper - 1e-10
        else:
            free = g > lower + 1e-10
        if not np.any(free):
            break
        step = diff / np.sum(free)
        g[free] += step
        g = np.clip(g, lower, upper)

    return g