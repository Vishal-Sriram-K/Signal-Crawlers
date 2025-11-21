import numpy as np
import matplotlib.pyplot as plt

from model import IntersectionModel
from optimizers import optimize_slsqp, projected_gradient_descent
from simulation import simulate_intersection


def main():
    # Problem parameters
    C = 120.0   # cycle time (s)
    L = 16.0    # total lost time (s)
    G_total = C - L

    # arrival rates (veh/s)
    lambdas = np.array([0.5, 0.2, 0.3, 0.4])

    # saturation flows (veh/s) when green
    saturation_flows = np.array([1.2, 1.0, 1.1, 1.3])

    g_min = 5.0     # Constraint For Pedistrain walking
    g_max = None
    model = IntersectionModel(lambdas=lambdas, saturation_flows=saturation_flows, C=C, L=L, g_min=g_min, g_max=g_max)

    print("=== Intersection Parameters ===")
    print(f"Cycle time C = {C} s, lost time L = {L} s, G_total = {G_total} s")
    print(f"Arrival rates (veh/s): {lambdas}")
    print(f"Saturation flows (veh/s): {saturation_flows}")
    print()

    #Baseline equal split
    g_equal = np.full(4, G_total / 4.0)
    baseline_delay = model.total_delay(g_equal)

    print("Baseline (equal split) green times:", g_equal)
    print(f"Baseline total delay (model) = {baseline_delay:.4f}")
    print()

    #Optimization via SLSQP
    print("=== SLSQP Optimization ===")
    slsqp_result = optimize_slsqp(model, x0=g_equal)
    g_opt_slsqp = slsqp_result.x
    delay_opt_slsqp = model.total_delay(g_opt_slsqp)

    print("SLSQP success:", slsqp_result.success)
    print("SLSQP message:", slsqp_result.message)
    print("Optimized green times (SLSQP):", g_opt_slsqp)
    print(f"Optimized delay (SLSQP) = {delay_opt_slsqp:.4f}")
    print()

    #Optimization via PGD
    print("=== Projected Gradient Descent (PGD) ===")
    g_opt_pgd, history_pgd = projected_gradient_descent( model, x0=g_equal, step_size=0.5, max_iters=400, tol=1e-6)
    delay_opt_pgd = model.total_delay(g_opt_pgd)

    print("Optimized green times (PGD):", g_opt_pgd)
    print(f"Optimized delay (PGD) = {delay_opt_pgd:.4f}")
    print()

    # Simulation comparision
    print("=== Simulation Validation ===")
    sim_time = 3600  # 1 hour

    avg_delay_baseline, n_base = simulate_intersection( lambdas, saturation_flows, g_equal, C, L, sim_time=sim_time, seed=1)
    avg_delay_slsqp, n_slsqp = simulate_intersection( lambdas, saturation_flows, g_opt_slsqp, C, L, sim_time=sim_time, seed=1)

    print(f"Simulation (baseline): avg delay = {avg_delay_baseline:.2f} s/veh, N = {n_base}")
    print(f"Simulation (SLSQP)  : avg delay = {avg_delay_slsqp:.2f} s/veh, N = {n_slsqp}")
    print()

if __name__=="__main__":
    main()