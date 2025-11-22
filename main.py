import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import IntersectionModel
from optimizers import optimize_slsqp, projected_gradient_descent


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
    print("Iterations :", slsqp_result.nit)
    print("SLSQP message:", slsqp_result.message)
    print("Optimized green times (SLSQP):", g_opt_slsqp)
    print(f"Optimized delay (SLSQP) = {delay_opt_slsqp:.4f}")
    print()

    #Optimization via PGD
    print("=== Projected Gradient Descent (PGD) ===")
    g_opt_pgd, history_pgd, iter = projected_gradient_descent( model, x0=g_equal, step_size=1.0, max_iters=400, tol=1e-4, verbose=True )
    delay_opt_pgd = model.total_delay(g_opt_pgd)

    print("Optimized green times (PGD):", g_opt_pgd)
    print(f"Optimized delay (PGD) = {delay_opt_pgd:.4f}")
    print()

    # =====================================
    # VISUALIZATION 1: Compare Delays
    # =====================================
    methods = ["Baseline( 0 Iterations )", f"SLSQP( {slsqp_result.nit} Iterations )", f"PGD( {iter} Iterations )"]
    delays = [baseline_delay, delay_opt_slsqp, delay_opt_pgd]
    plt.figure(figsize=(8,5))
    sns.barplot(x=methods, y=delays)
    plt.title("Total Delay Comparison")
    plt.ylabel("Delay (veh-sec)")
    plt.savefig("viz_delay_comparison.png")
    plt.close()

    # =====================================
    # VISUALIZATION 2: PGD Convergence Curve
    # =====================================
    plt.figure(figsize=(8,5))
    plt.plot(history_pgd, linewidth=2)
    plt.title("PGD Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Delay")
    plt.grid()
    plt.savefig("viz_pgd_convergence.png")
    plt.close()

    # =====================================
    # VISUALIZATION 3: Green Times Comparison
    # =====================================
    plt.figure(figsize=(8,5))
    bar_w = 0.25
    x = np.arange(4)

    plt.bar(x - bar_w, g_equal, width=bar_w, label="Baseline")
    plt.bar(x, g_opt_slsqp, width=bar_w, label="SLSQP")
    plt.bar(x + bar_w, g_opt_pgd, width=bar_w, label="PGD")

    plt.xticks(x, [f"Phase {i+1}" for i in range(4)])
    plt.ylabel("Green Time (s)")
    plt.title("Green Times Comparison")
    plt.legend()
    plt.savefig("viz_green_times.png")
    plt.close()

    # =====================================
    # VISUALIZATION 4: Simulated Queue Over Time (Phase 1)
    # =====================================
    sim_time_series = 600  # first 10 minutes
    queues = []
    rng = np.random.default_rng(1)
    q = 0
    for t in range(sim_time_series):
        arr = rng.poisson(lambdas[0])
        q += arr
        if (t % int(C)) < g_opt_slsqp[0]:  # active if green
            served = min(q, int(saturation_flows[0]))
            q -= served
        queues.append(q)

    plt.figure(figsize=(8,5))
    plt.plot(queues)
    plt.title("Queue Length Over Time (Phase 1, SLSQP)")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue Length")
    plt.grid()
    plt.savefig("viz_queue_phase1.png")
    plt.close()


if __name__=="__main__":
    main()