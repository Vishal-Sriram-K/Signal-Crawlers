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

# Projected Gradient descent
def projected_gradient_descent( model, x0=None,
                                step_size=1, max_iters=200,
                                tol=1e-3, verbose=False,):
    
    if x0 is None:
        x = np.full(4, model.G_total / 4.0)
    else:
        x = np.array(x0, dtype=float)

    history = []
    for k in range(max_iters):
        f_val = model.total_delay(x)
        history.append(f_val)
        grad = numerical_gradient(model.total_delay, x)
        x_new = x - step_size * grad
        x_new = project_onto_simplex_with_bounds(
            x_new,
            total=model.G_total,
            lower=model.g_min,
            upper=model.g_max,
        )
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            if verbose:
                print(f"[PGD] Converged at iter {k}")
            break

        x = x_new
        if verbose and (k % 50 == 0 or k == max_iters - 1):
            print(f"[PGD] iter={k}, f={f_val:.5f}")

    return x, np.array(history)
