import numpy as np

def simulate_intersection(lambdas,saturation_flows,g,C,L,sim_time=3600,seed=0,):
    
    rng = np.random.default_rng(seed)
    lambdas = np.asarray(lambdas, dtype=float)
    s = np.asarray(saturation_flows, dtype=float)
    g = np.asarray(g, dtype=float)

    queues = np.zeros(4, dtype=int)
    total_delay = 0.0
    total_vehicles = 0

    phase_intervals = []
    t = 0
    for i in range(4):
        phase_intervals.append((t, t + int(round(g[i])), i))
        t += int(round(g[i]))

    for current_time in range(sim_time):
        arrivals = rng.poisson(lambdas)
        queues += arrivals
        total_vehicles += int(np.sum(arrivals))

        time_in_cycle = current_time % int(round(C))
        active_phase = -1
        for start, end, phase_id in phase_intervals:
            if start <= time_in_cycle < end:
                active_phase = phase_id
                break

        if active_phase >= 0:
            i = active_phase
            capacity_this_sec = int(round(s[i]))  
            served = min(queues[i], capacity_this_sec)
            queues[i] -= served

        total_delay += float(np.sum(queues))

    if total_vehicles == 0:
        return 0.0, 0

    avg_delay = total_delay / total_vehicles
    return float(avg_delay), int(total_vehicles)
