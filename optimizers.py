import numpy as np
from model import project_onto_simplex_with_bounds

# NUMERICAL GRADIENT
def numerical_gradient(f, x, h=1e-4):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2.0 * h)
    return grad

def projected_gradient_descent(
    model,
    x0=None,
    step_size=1.0,
    max_iters=200,
    tol=1e-3,
    alpha=1e-4,
    beta=0.5,
    verbose=True,
):
    if x0 is None:
        x = np.full(4, model.G_total / 4.0)
    else:
        x = np.array(x0, dtype=float)

    history = []

    for k in range(max_iters):
        f_val = model.total_delay(x)
        history.append(f_val)

        grad = numerical_gradient(model.total_delay, x)
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq < 1e-16:
            if verbose:
                print(f"[PGD] Gradient ~ 0 at iter {k}")
            break

        # Backtracking line search
        step = step_size
        while True:
            x_trial = x - step * grad
            x_trial = project_onto_simplex_with_bounds(
                x_trial,
                total=model.G_total,
                lower=model.g_min,
                upper=model.g_max,
            )
            f_trial = model.total_delay(x_trial)

            # Armijo condition: sufficient decrease
            if f_trial <= f_val - alpha * step * grad_norm_sq:
                break

            step *= beta
            if step < 1e-6:
                # step became too small
                break

        if verbose and (k % 50 == 0 or k == max_iters - 1):
            print(f"[PGD] iter={k}, f={f_val:.5f}, step={step:.3e}")

        # update
        if np.linalg.norm(x_trial - x) < tol:
            x = x_trial
            if verbose:
                print(f"[PGD] Converged at iter {k}")
            break

        x = x_trial

    return x, np.array(history), k

# Optimization using SLSQP with equality and inequality constraints
def optimize_slsqp(model, x0=None):

    import scipy.optimize as opt

    if x0 is None:
        x0 = np.full(4, model.G_total / 4.0)

    bounds = [(model.g_min[i], model.g_max[i]) for i in range(4)]
    
    cons = [
        {"type": "eq", "fun": lambda g: np.sum(g) - model.G_total},
    ]

    history = []

    def callback(g):
        delay=model.total_delay(g)
        history.append(delay)
    result = opt.minimize(
        fun=model.total_delay,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        callback=callback,
        options={"maxiter": 500, "ftol": 1e-8, "disp": False},
    )

    return result,np.array(history)