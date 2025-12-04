# Job Shop Scheduling Problem (JSSP) Framework with VAE-Enhanced PSO

A comprehensive Python framework for solving the Job Shop Scheduling Problem using Particle Swarm Optimization (PSO) with Variational Autoencoder (VAE)-based population initialization strategies.

## Overview

This repository implements advanced metaheuristic algorithms for the Job Shop Scheduling Problem, featuring:

- **Multiple Population Initialization Methods**: 20+ different strategies including heuristic-based, random-based, and hybrid approaches
- **VAE-Enhanced Optimization**: Deep learning-based population generation using Variational Autoencoders
- **PSO with Local Search**: Particle Swarm Optimization enhanced with intensive local search mechanisms
- **Comprehensive Evaluation Framework**: Tools for benchmarking and statistical analysis

## Features

### Population Initialization Methods

#### Heuristic-Based Methods
- **SPT (Shortest Processing Time)**: Prioritizes operations with shorter processing times
- **LPT (Longest Processing Time)**: Prioritizes operations with longer processing times
- **FIFO (First In First Out)**: Sequential operation ordering
- **Critical Ratio**: Based on remaining work and operation count
- **Most/Least Work Remaining**: Prioritizes based on total remaining processing time

#### Random-Based Methods
- **Uniform Random**: Pure random initialization
- **Random Normal**: Normal distribution-based initialization
- **Latin Hypercube Sampling (LHS)**: Space-filling design for better coverage
- **Sobol Sequence**: Quasi-random low-discrepancy sequences

#### Hybrid Methods
- **Quality-Diversity Balance**: Combines heuristic and random approaches
- **Elite Seeding**: Seeds population with high-quality heuristic solutions
- **Pure Heuristic Mix**: Mixture of multiple heuristic strategies
- **Staged Population**: Multi-stage initialization with different strategies

#### VAE-Enhanced Methods
- **Compact VAE**: Lightweight Variational Autoencoder for solution generation
- **Multi-Dimensional VAE System**: Handles different problem dimensions
- **Elite Solution Learning**: Trains on high-quality solutions from previous runs

### Optimization Algorithm

- **PSO (Particle Swarm Optimization)**
  - Adaptive inertia weight (linearly decreasing)
  - Cognitive and social components
  - Velocity clamping for stability
  - Diversity-based restart mechanism
  - Intensive local search on best solutions

### Problem Encoding

- **Random Key Encoding**: Continuous vector representation for JSSP
- **Priority-Based Decoding**: Converts continuous vectors to feasible schedules
- **Critical Path Integration**: Considers critical path information during decoding

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
torch
```

## Installation

```bash
git clone https://github.com/yourusername/jssp-vae-pso.git
cd jssp-vae-pso
pip install -r requirements.txt
```

## Dataset

This repository includes the Taillard benchmark instances (`taillard_dataset.txt`) for convenience. These instances are:

- **Public URL**: http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html
- **Citation**: Taillard, É. (1993). Benchmarks for basic scheduling problems. 
  *European Journal of Operational Research*, 64(2), 278-285. 
  DOI: [10.1016/0377-2217(93)90182-M](https://doi.org/10.1016/0377-2217(93)90182-M)

The dataset should be in the following format:

```
instance_name
num_jobs num_machines
machine_id1 processing_time1 machine_id2 processing_time2 ...
...
```

## Usage

### Basic Example

```python
from jssp_framework import JSSPSolver, update_problem_instance

# Load a problem instance
update_problem_instance("ta01")

# Initialize solver
solver = JSSPSolver(pop_size=50, max_iterations=400, num_runs=30)

# Run experiments with specific initialization methods
instances = ["ta01", "ta11", "ta21"]
methods = ["spt_heuristic", "lhs", "quality_diversity_balance"]

results_df = solver.run_experiments(instances, methods)
```

### Using VAE-Enhanced Initialization

```python
from vae_jssp_runner import FullVAEJSSPRunner

# Initialize VAE-enhanced runner
runner = FullVAEJSSPRunner(pop_size=50, max_iterations=400, num_runs=30)

# The VAE system will automatically load pre-trained models
# and generate populations based on learned elite solutions
```

### Custom Population Initialization

```python
from jssp_framework import PopulationInitializer

# Create custom initializer
initializer = PopulationInitializer(pop_size=100, dim=225)

# Use different methods
pop1 = initializer.spt_heuristic()
pop2 = initializer.latin_hypercube()
pop3 = initializer.quality_diversity_balance(heuristic_ratio=0.7)
```

## File Structure

```
.
├── jssp_framework.py              # Main JSSP solver framework
├── vae_population_framework.py    # VAE implementation for population generation
├── vae_jssp_runner.py            # Runner for VAE-enhanced PSO experiments
├── taillard_dataset.txt          # Taillard benchmark instances
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```



## Benchmark Instances

The framework has been tested on Taillard benchmark instances including:
- Small instances: ta01 (15×15), ta11 (20×15)
- Medium instances: ta21 (20×20), ta31 (30×15)
- Large instances: ta41 (30×20), ta51 (50×15)

## Acknowledgments

- Taillard benchmark instances for JSSP
- PyTorch for deep learning components
- NumPy and SciPy for numerical computations
