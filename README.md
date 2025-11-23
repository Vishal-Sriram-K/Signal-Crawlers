# Traffic Signal Optimization

This project performs traffic signal green-time optimization using mathematical optimization and discrete traffic simulation.

It includes:

- SLSQP Optimization
- Projected Gradient Descent
- Delay and queue visualizations

## Team Members
- Vishal Sriram BT2024036
- Anish Reddy BT2024228
- Sri charan Reddy BT2024143

## Problem Statement and Motivation

Traffic signals with fixed timings often do not match real traffic demand. This leads to long queues, wasted green time, and high delays.  
The goal is to compute optimal green times that minimize total delay.

## Mathematical Formulation

### Objective
TotalDelay = Σ λᵢ × dᵢ(gᵢ)

### Delay Formula (Webster Approximation)
d = (0.5 × C × (1 − g/C)^2) / (1 − X)  
X = λ / (s × (g/C))

### Constraints
Σ gᵢ = C − L  
g_min ≤ gᵢ ≤ g_max  

| Symbol                            | Code Variable | Meaning                            | Units   |
| --------------------------------- | ------------- | ---------------------------------- | ------- |
| **C**                             | `C`           | Cycle time                         | seconds |
| **g<sub>i</sub>**                 | `g_i`         | Green time for movement *i*        | seconds |
| **g/C**                           | `g_ratio`     | Green ratio                        | —       |
| **λ<sub>i</sub>**                 | `lambda_i`    | Arrival flow rate                  | veh/s   |
| **s<sub>i</sub>**                 | `s_i`         | Saturation flow                    | veh/s   |
| **X**                             | `X`           | Degree of saturation               | —       |
| **d**                             | `d`           | Average delay per vehicle          | seconds |
| **λ<sub>i</sub> · d<sub>i</sub>** | —             | Delay contribution of approach *i* | veh     |


## Methodology and Solver Details

- SLSQP Optimization (nonlinear constrained optimization)
- Projected Gradient Descent (gradient with projection)
- Simulation (Poisson arrivals + saturation flow)

## Results, Analysis, and Discussion

SLSQP achieves the most fast delay reduction.  
PGD converges slow and gives-optimal timings.  

### Optimization Methods
- SLSQP
- PGD

### Visualizations
- Delay comparison
- PGD convergence curve
- SlSQP convergence curve
- Green time comparison
- Queue length evolution

## Project Structure

- main.py
- model.py
- optimizers.py
- requirements.txt
- README.md

## Installation

git clone https://github.com/Vishal-Sriram-K/Signal-Crawlers  
cd Signal-Crawlers  
pip install -r requirements.txt

- Free right-turn handling (uncontrolled turn with zero delay impact)
- Free left-turn handling (for left-hand traffic systems, uncontrolled movement)
- Adaptive cycle lengths and adaptive arrival rates
- Using better non linear optimization algorithms
