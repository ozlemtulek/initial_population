#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
import time
import os
from typing import Dict, List, Tuple
import torch
from jssp_framework import *

# VAE modellerini import et
from vae_population_framework import CompactVAE, MultiDimensionalVAESystem



class FullVAEJSSPRunner:
    
    def __init__(self, pop_size=50, max_iterations=400, num_runs=30):
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.num_runs = num_runs
        
        # VAE system
        self.vae_system = MultiDimensionalVAESystem()
        self.vae_system.load_models()
        
        print(f" Full PSO-VAE Runner")
    
    def calculate_population_diversity_stats(self, population: np.ndarray, fitness_values: np.ndarray) -> Dict[str, float]:
        
        if len(population) < 2:
            return {
                'initial_diversity': 0.0,
                'best_worst_diversity': 0.0,
                'mean_diversity': 0.0
            }
        
        # Initial population diversity (mean distance from population center)
        mean_vector = np.mean(population, axis=0)
        distances = np.linalg.norm(population - mean_vector, axis=1)
        initial_diversity = np.mean(distances)
        
        # Best vs worst solution diversity
        best_idx = np.argmin(fitness_values)
        worst_idx = np.argmax(fitness_values)
        best_worst_diversity = np.linalg.norm(population[best_idx] - population[worst_idx])
        
        # Mean pairwise diversity (sample for efficiency)
        if len(population) > 20:
            # Sample 20 random pairs for large populations
            indices = np.random.choice(len(population), size=min(20, len(population)), replace=False)
            sample_pop = population[indices]
        else:
            sample_pop = population
        
        pairwise_distances = []
        for i in range(len(sample_pop)):
            for j in range(i+1, len(sample_pop)):
                dist = np.linalg.norm(sample_pop[i] - sample_pop[j])
                pairwise_distances.append(dist)
        
        mean_diversity = np.mean(pairwise_distances) if pairwise_distances else 0.0
        
        return {
            'initial_diversity': initial_diversity,
            'best_worst_diversity': best_worst_diversity,
            'mean_diversity': mean_diversity
        }
    
    def calculate_diversity(self, population: np.ndarray) -> float:
       
        if len(population) < 2:
            return 0.0
        
        mean_vector = np.mean(population, axis=0)
        distances = np.linalg.norm(population - mean_vector, axis=1)
        return np.mean(distances)
    
    def _local_search(self, solution: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        
        # Import current global variables
        import jssp_framework
        machine_sequence = jssp_framework.machine_sequence
        processing_times = jssp_framework.processing_times
        
        best_solution = solution.copy()
        best_fitness = evaluate_jssp_solution(
            cached_decode(tuple(solution))[0], machine_sequence, processing_times
        )
        
        # Try small perturbations
        for _ in range(5):
            new_solution = solution.copy()
            # Randomly select 10% of dimensions to perturb
            indices = np.random.choice(len(solution), size=max(1, len(solution)//10), replace=False)
            for idx in indices:
                noise = np.random.normal(0, 0.05)
                new_solution[idx] = np.clip(solution[idx] + noise, lb[idx], ub[idx])
            
            new_fitness = evaluate_jssp_solution(
                cached_decode(tuple(new_solution))[0], machine_sequence, processing_times
            )
            
            if new_fitness < best_fitness:
                best_solution = new_solution.copy()
                best_fitness = new_fitness
        
        return best_solution
    
    def _intensive_local_search(self, solution: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        
        import jssp_framework
        machine_sequence = jssp_framework.machine_sequence
        processing_times = jssp_framework.processing_times
        
        best_solution = solution.copy()
        best_fitness = evaluate_jssp_solution(
            cached_decode(tuple(solution))[0], machine_sequence, processing_times
        )
        
        # Multiple local search strategies
        strategies = [0.01, 0.02, 0.05, 0.1]  # Different noise levels
        
        for noise_level in strategies:
            for _ in range(3):
                new_solution = solution.copy()
                # Perturb all dimensions with small noise
                noise = np.random.normal(0, noise_level, len(solution))
                new_solution += noise
                new_solution = np.clip(new_solution, lb, ub)
                
                new_fitness = evaluate_jssp_solution(
                    cached_decode(tuple(new_solution))[0], machine_sequence, processing_times
                )
                
                if new_fitness < best_fitness:
                    best_solution = new_solution.copy()
                    best_fitness = new_fitness
                    solution = best_solution.copy()  # Update base solution
        
        return best_solution
    
    def generate_vae_population(self, instance: str) -> np.ndarray:
        """VAE ile popülasyon üret"""
        update_problem_instance(instance)
        
        import jssp_framework
        dim = jssp_framework.num_jobs * jssp_framework.num_machines
        
        if dim not in self.vae_system.vae_models:
            raise ValueError(f"VAE model for {dim}D not found!")
        
        return self.vae_system.generate_population_for_dimension(dim, self.pop_size)
    
    def generate_traditional_population(self, instance: str, method: str = 'fifo') -> np.ndarray:

        update_problem_instance(instance)
        

        import jssp_framework
        dim = jssp_framework.num_jobs * jssp_framework.num_machines
        
        lb, ub = np.zeros(dim), np.ones(dim)
        

        from jssp_framework import PopulationInitializer
        initializer = PopulationInitializer(dim, self.pop_size, lb, ub)
        
        if method == 'fifo':
            return initializer.fifo_heuristic()
        elif method == 'most_work':
            return initializer.most_work_remaining()
        else:
            return initializer.fifo_heuristic()
    
    def generate_hybrid_population(self, instance: str, vae_ratio: float = 0.6) -> np.ndarray:
        """Hibrit popülasyon üret"""
        vae_size = int(self.pop_size * vae_ratio)
        traditional_size = self.pop_size - vae_size
        
        try:
            vae_pop = self.generate_vae_population(instance)[:vae_size]
            trad_pop = self.generate_traditional_population(instance, 'fifo')[:traditional_size]
            return np.vstack([vae_pop, trad_pop])
        except:
            return self.generate_traditional_population(instance, 'fifo')
    
    def run_single_experiment(self, instance: str, method: str, run_id: int) -> Dict:
        """Tek deney"""
        start_time = time.time()
        
        print(f"Running {instance} - {method} - Run {run_id}")
        
        update_problem_instance(instance)
        
        import jssp_framework
        global num_jobs, num_machines, machine_sequence, processing_times
        num_jobs = jssp_framework.num_jobs
        num_machines = jssp_framework.num_machines
        machine_sequence = jssp_framework.machine_sequence
        processing_times = jssp_framework.processing_times
        
        dim = num_jobs * num_machines
        lb = np.zeros(dim)
        ub = np.ones(dim)
        
        # Popülasyon üret
        if method == 'VAE_Pure':
            population = self.generate_vae_population(instance)
        elif method == 'VAE_Hybrid_60':
            population = self.generate_hybrid_population(instance, 0.6)
        elif method == 'VAE_Hybrid_40':
            population = self.generate_hybrid_population(instance, 0.4)
        elif method == 'FIFO_Traditional':
            population = self.generate_traditional_population(instance, 'fifo')
        elif method == 'MostWork_Traditional':
            population = self.generate_traditional_population(instance, 'most_work')
        else:
            population = self.generate_traditional_population(instance, 'fifo')
        

        fitness_values = np.array([evaluate_jssp_solution(
            cached_decode(tuple(ind))[0], machine_sequence, processing_times
        ) for ind in population])
        

        diversity_stats = self.calculate_population_diversity_stats(population, fitness_values)
        
      
        best_fitness_history = []
        
        # PSO parameters
        w = 0.9  # inertia weight
        c1 = 2.0  # cognitive parameter
        c2 = 2.0  # social parameter
        w_min = 0.4
        w_max = 0.9
        
        # Initialize velocities
        velocity = np.random.uniform(-0.1, 0.1, (self.pop_size, dim))
        
        # Personal best positions and fitness
        pbest_pos = population.copy()
        pbest_fit = fitness_values.copy()
        
        # Global best
        gbest_idx = np.argmin(fitness_values)
        gbest_pos = population[gbest_idx].copy()
        gbest_fit = fitness_values[gbest_idx]
        
        convergence_iteration = None
        target_fitness = gbest_fit * 0.95 if gbest_fit < 1e5 else gbest_fit - 100
        
        for iteration in range(self.max_iterations):
            # Update inertia weight 
            w = w_max - (w_max - w_min) * iteration / self.max_iterations
            
            # Update particles 
            for i in range(self.pop_size):
                # Update velocity
                r1, r2 = np.random.random(dim), np.random.random(dim)
                velocity[i] = (w * velocity[i] + 
                              c1 * r1 * (pbest_pos[i] - population[i]) + 
                              c2 * r2 * (gbest_pos - population[i]))
                
                # Limit velocity
                velocity[i] = np.clip(velocity[i], -0.5, 0.5)
                
                # Update position
                population[i] = population[i] + velocity[i]
                population[i] = np.clip(population[i], lb, ub)
                
                # Evaluate new position
                new_fitness = evaluate_jssp_solution(
                    cached_decode(tuple(population[i]))[0], machine_sequence, processing_times
                )
                
                fitness_values[i] = new_fitness
                
                # Update personal best
                if new_fitness < pbest_fit[i]:
                    pbest_pos[i] = population[i].copy()
                    pbest_fit[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < gbest_fit:
                        gbest_pos = population[i].copy()
                        gbest_fit = new_fitness
                        
                        if convergence_iteration is None and gbest_fit <= target_fitness:
                            convergence_iteration = iteration
            
            # Local search on best solution every 10 iterations 
            if iteration % 10 == 0 and gbest_fit < 1e5:
                improved_solution = self._local_search(gbest_pos, lb, ub)
                improved_fitness = evaluate_jssp_solution(
                    cached_decode(tuple(improved_solution))[0], machine_sequence, processing_times
                )
                
                if improved_fitness < gbest_fit:
                    gbest_pos = improved_solution.copy()
                    gbest_fit = improved_fitness
            
            # Record best fitness
            best_fitness_history.append(gbest_fit)
            
            # Diversity check and restart mechanism 
            current_diversity = self.calculate_diversity(population)
            if current_diversity < 0.01 and iteration > 50:
                # Reinitialize worst 30% of population
                worst_indices = np.argsort(fitness_values)[-int(0.3 * self.pop_size):]
                for idx in worst_indices:
                    if idx != np.argmin(fitness_values):  # Keep the best
                        population[idx] = np.random.uniform(lb, ub, dim)
                        velocity[idx] = np.random.uniform(-0.1, 0.1, dim)
        
        # Final local search on the best solution 
        if gbest_fit < 1e5:
            final_solution = self._intensive_local_search(gbest_pos, lb, ub)
            final_fitness = evaluate_jssp_solution(
                cached_decode(tuple(final_solution))[0], machine_sequence, processing_times
            )
            if final_fitness < gbest_fit:
                gbest_pos = final_solution
                gbest_fit = final_fitness
        
        # Calculate final diversity statistics 
        final_diversity_stats = self.calculate_population_diversity_stats(population, fitness_values)
        
        execution_time = time.time() - start_time
        

        result = {
            'instance': instance,
            'method': method,
            'run_id': run_id,
            'best_fitness': gbest_fit,
            'initial_diversity': diversity_stats['initial_diversity'],
            'initial_best_worst_diversity': diversity_stats['best_worst_diversity'],
            'initial_mean_diversity': diversity_stats['mean_diversity'],
            'final_diversity': final_diversity_stats['initial_diversity'],
            'final_best_worst_diversity': final_diversity_stats['best_worst_diversity'],
            'final_mean_diversity': final_diversity_stats['mean_diversity'],
            'convergence_iteration': convergence_iteration if convergence_iteration else self.max_iterations,
            'iterations_completed': iteration + 1,
            'execution_time': execution_time
        }
        
        print(f" ({method}-{run_id}): {gbest_fit:.2f}")
        
        return result
    
    def run_full_experiments(self, instances: List[str]) -> pd.DataFrame:
        """Tam deneyler çalıştır"""
        
        methods = [
            'VAE_Pure',
            'VAE_Hybrid_60', 
            'VAE_Hybrid_40',
            'FIFO_Traditional',
            'MostWork_Traditional'
        ]
        
        print(f" {len(instances)} instance × {len(methods)} metot × {self.num_runs} çalışma")
        print(f" Her deney: {self.max_iterations} PSO iterasyonu")
        
        results = []
        total_experiments = len(instances) * len(methods) * self.num_runs
        completed = 0
        
        for instance in instances:
            print(f"\n{'='*50}")
            print(f" INSTANCE: {instance}")
            
            for method in methods:
                print(f"\n METHOD: {method}")
                
                method_results = []
                
                for run in range(self.num_runs):
                    try:
                        result = self.run_single_experiment(instance, method, run)
                        
                        if result:
                            results.append(result)
                            method_results.append(result)
                        
                        completed += 1
                        progress = (completed / total_experiments) * 100
                        print(f"    Progress: {completed}/{total_experiments} ({progress:.1f}%)")
                        
                    except Exception as e:
                        print(f"    Run {run} failed: {e}")
                        completed += 1
                        continue
                
                # Metot özeti
                if method_results:
                    avg_fitness = np.mean([r['best_fitness'] for r in method_results])
                    std_fitness = np.std([r['best_fitness'] for r in method_results])
                    best_fitness = min([r['best_fitness'] for r in method_results])
                    avg_convergence = np.mean([r['convergence_iteration'] for r in method_results])
                    
                    print(f"  {method}: Ortalama={avg_fitness:.2f}±{std_fitness:.2f}, En iyi={best_fitness:.2f}, Convergence={avg_convergence:.1f}")
                else:
                    print(f"  {method}: Hiç sonuç yok")
        
        print(f"TAM DENEYLER TAMAMLANDI!")
        print(f"Toplam {len(results)} sonuç")
        
        return pd.DataFrame(results)
    
    def analyze_full_results(self, results_df: pd.DataFrame) -> Dict:
        
        print(f"\n TAM SONUÇ ANALİZİ")
        print("=" * 60)
        
        analysis = {}
        
        # Genel performans metotlara göre (best_fitness ile)
        method_stats = results_df.groupby('method').agg({
            'best_fitness': ['mean', 'std', 'min', 'max'],
            'initial_diversity': ['mean', 'std'],
            'final_diversity': ['mean', 'std'],
            'convergence_iteration': ['mean', 'std'],
            'execution_time': ['mean', 'std']
        }).round(4)
        
        analysis['method_statistics'] = method_stats
        
        print(f"\n METOT PERFORMANSLARI (ortalama best fitness):")
        method_ranking = results_df.groupby('method')['best_fitness'].mean().sort_values()
        for method, avg_fitness in method_ranking.items():
            std_fitness = results_df.groupby('method')['best_fitness'].std()[method]
            valid_count = len(results_df[results_df['method'] == method])
            print(f"  {method:20s}: {avg_fitness:.2f} ± {std_fitness:.2f} ({valid_count} çalışma)")
        
        analysis['method_ranking'] = method_ranking
        
        # Instance bazında performans
        print(f"\n INSTANCE BAZINDA PERFORMANS:")
        for instance in results_df['instance'].unique():
            print(f"\n   {instance}:")
            instance_data = results_df[results_df['instance'] == instance]
            instance_ranking = instance_data.groupby('method')['best_fitness'].mean().sort_values()
            
            for i, (method, avg_fitness) in enumerate(instance_ranking.items(), 1):

                print(f"  {method:18s}: {avg_fitness:.2f}")
        
        # VAE vs Traditional karşılaştırması
        vae_methods = results_df[results_df['method'].str.contains('VAE')]
        trad_methods = results_df[~results_df['method'].str.contains('VAE')]
        
        if len(vae_methods) > 0 and len(trad_methods) > 0:
            print(f"\n VAE-DİĞER YÖNTEMLER KARŞILAŞTIRMA:")
            
            vae_avg_fitness = vae_methods['best_fitness'].mean()
            trad_avg_fitness = trad_methods['best_fitness'].mean()
            fitness_improvement = ((trad_avg_fitness - vae_avg_fitness) / trad_avg_fitness) * 100
            
            print(f"   VAE ortalama fitness: {vae_avg_fitness:.2f}")
            print(f"   Geleneksel ortalama fitness: {trad_avg_fitness:.2f}")
            print(f"   VAE fitness iyileştirmesi: {fitness_improvement:+.1f}%")
            
            # Convergence karşılaştırması
            vae_conv = vae_methods['convergence_iteration'].mean()
            trad_conv = trad_methods['convergence_iteration'].mean()
            
            print(f"   VAE ortalama convergence: {vae_conv:.1f} iterasyon")
            print(f"   Geleneksel ortalama convergence: {trad_conv:.1f} iterasyon")
            
            analysis['vae_vs_traditional'] = {
                'vae_avg_fitness': vae_avg_fitness,
                'traditional_avg_fitness': trad_avg_fitness,
                'fitness_improvement_percent': fitness_improvement,
                'vae_avg_convergence': vae_conv,
                'traditional_avg_convergence': trad_conv
            }
        
        return analysis
    
    def save_full_results(self, results_df: pd.DataFrame, analysis: Dict):
        """Tam sonuçları kaydet"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # CSV kaydet
        csv_filename = f"full_pso_vae_results_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        
        # JSON analiz kaydet
        json_filename = f"full_pso_vae_analysis_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"TAM SONUÇLAR KAYDEDİLDİ:")
        print(f"Ham veri: {csv_filename}")
        print(f"Analiz: {json_filename}")

def main():
    """TAM PSO-VAE DENEYİ"""
    
    # Test instances
    test_instances = ['ta01', 'ta11', 'ta21', 'ta31', 'ta41', 'ta51']
    
    # Runner oluştur
    runner = FullVAEJSSPRunner(
        pop_size=50,
        max_iterations=400,
        num_runs=30
    )
    
    results_df = runner.run_full_experiments(test_instances)
    
    if len(results_df) > 0:
        # Analiz yap
        analysis = runner.analyze_full_results(results_df)
        
        # Kaydet
        runner.save_full_results(results_df, analysis)
        
        print(" TAM PSO-VAE DENEYİ TAMAMLANDI!")
    else:
        print("SONUÇ ELDE EDİLEMEDİ!")    

if __name__ == "__main__":
    main()