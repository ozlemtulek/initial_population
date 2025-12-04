import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
from functools import lru_cache
import heapq
import time
import json
from scipy.stats import qmc
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Global variables for JSSP problem
num_jobs = 0
num_machines = 0
machine_sequence = []
processing_times = []

@lru_cache(maxsize=50000)
def cached_decode(vector_tuple):
    vector = np.array(vector_tuple, dtype=np.float32)  
    return decode_random_key_with_critical_path_with_cp(vector, num_jobs, num_machines, machine_sequence, processing_times)

def read_jssp_dataset(filename: str, instance_name: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    with open(filename, "r") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == instance_name:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError(f"{instance_name} not found in dataset.")

    # Get dimensions
    num_jobs, num_machines = map(int, lines[start_idx].split())
    
    machine_sequence = np.zeros((num_jobs, num_machines), dtype=int)
    processing_times = np.zeros((num_jobs, num_machines), dtype=int)

    for j in range(num_jobs):
        values = list(map(int, lines[start_idx + 1 + j].split()))
        machine_sequence[j, :] = values[0::2]  
        processing_times[j, :] = values[1::2]  

    return num_jobs, num_machines, machine_sequence, processing_times

def update_problem_instance(instance_name: str):
    global num_jobs, num_machines, machine_sequence, processing_times
    num_jobs, num_machines, machine_sequence, processing_times = read_jssp_dataset(
        filename="taillard_dataset.txt", 
        instance_name=instance_name
    )
    machine_sequence -= 1  # 0-based indexing
    cached_decode.cache_clear()  # Clear cache
    print(f"Problem instance updated: {instance_name} ({num_jobs}x{num_machines})")

# JSSP için random key decode işlemi
def decode_random_key_with_critical_path_with_cp(vector, num_jobs, num_machines, machine_sequence, processing_times):
    
    expected_len = num_jobs * num_machines
    if len(vector) != expected_len:
        print(f" Vektör uzunluğu hatası: {len(vector)} != {expected_len}")
        return [], []
    
    if not np.all((vector >= 0) & (vector <= 1)):
        print(f" Vektör değerleri [0,1] dışında: min={np.min(vector):.3f}, max={np.max(vector):.3f}")
        # Normalize et
        vector = np.clip(vector, 0, 1)


    job_operations = []
    for job in range(num_jobs):
        ops = []
        for op_idx in range(num_machines):
            idx = job + num_jobs * op_idx
            ops.append((vector[idx], op_idx))
        ops.sort()
        job_operations.append(ops)

    ready_heap = []
    job_current_op = [0] * num_jobs
    for job in range(num_jobs):
        if job_operations[job] and len(job_operations[job]) > 0:
            priority, _ = job_operations[job][0]
            heapq.heappush(ready_heap, (priority, job))

    try:
        min_machine_id = int(np.min(machine_sequence))
        max_machine_id = int(np.max(machine_sequence))
        machine_offset = -min_machine_id if min_machine_id < 0 else 0
        machine_ready = [0.0] * (max_machine_id + machine_offset + 1)
    except Exception as e:
        print(f" Makine ID array hatası: {e}")
        return [], []
    
    job_ready = [0.0] * num_jobs
    schedule = []
    start_times = np.zeros((num_jobs, num_machines), dtype=np.float64)
    end_times = np.zeros((num_jobs, num_machines), dtype=np.float64)

    iteration_count = 0
    max_iterations = num_jobs * num_machines * 3  # Biraz daha cömert limit

    while ready_heap and iteration_count < max_iterations:
        iteration_count += 1
        
        try:
            priority, job = heapq.heappop(ready_heap)
            op_seq = job_current_op[job]
            
            if op_seq >= num_machines:
                continue

            _, op_idx = job_operations[job][op_seq]
            
            # Güvenli indeks kontrolü
            if not (0 <= job < num_jobs and 0 <= op_idx < num_machines):
                print(f" Geçersiz indeks: Job {job}, Op {op_idx}")
                continue
                
            machine = int(machine_sequence[job, op_idx])
            duration = float(processing_times[job, op_idx])
            
            # Duration kontrolü ve düzeltmesi
            if not np.isfinite(duration) or duration <= 0:
                duration = 1.0
                if iteration_count <= 5:  # İlk birkaç hata için uyarı
                    print(f" Geçersiz süre düzeltildi: Job {job}, Op {op_idx}, Duration {duration}")

            machine_idx = machine + machine_offset
            if not (0 <= machine_idx < len(machine_ready)):
                print(f" Makine indeksi aralık dışı: {machine_idx} (aralık: 0-{len(machine_ready)-1})")
                continue

            start = max(job_ready[job], machine_ready[machine_idx])
            end = start + duration
            
            # Zaman tutarlılık kontrolü
            if end <= start or not np.isfinite(start) or not np.isfinite(end):
                print(f" Zaman tutarsızlığı: Job {job}, Op {op_idx}, Start {start}, End {end}")
                continue

            # Schedule'a ekle
            schedule.append((job, op_idx, machine, start, end))
            start_times[job, op_idx] = start
            end_times[job, op_idx] = end
            job_ready[job] = end
            machine_ready[machine_idx] = end

            # Sonraki operasyonu ekle
            job_current_op[job] += 1
            if job_current_op[job] < num_machines and job_current_op[job] < len(job_operations[job]):
                next_priority, _ = job_operations[job][job_current_op[job]]
                heapq.heappush(ready_heap, (next_priority, job))

        except Exception as e:
            print(f" İterasyon {iteration_count} hatası: {e}")
            continue

    if iteration_count >= max_iterations:
        print(f" Maksimum iterasyon ({max_iterations}) aşıldı. Mevcut schedule: {len(schedule)} operasyon")

    scheduled_ops = set((j, o) for j, o, *_ in schedule)
    missing_ops = []
    
    for job in range(num_jobs):
        for op in range(num_machines):
            if (job, op) not in scheduled_ops:
                missing_ops.append((job, op))
    
    if missing_ops:
        print(f" {len(missing_ops)} eksik operasyon tespit edildi, tamamlanıyor...")
        
        for job, op in missing_ops:
            try:
                machine = int(machine_sequence[job, op])
                duration = float(processing_times[job, op])
                
                if duration <= 0:
                    duration = 1.0
                
                machine_idx = machine + machine_offset
                
                # Güvenli start time hesaplama
                job_time = job_ready[job] if job < len(job_ready) else 0
                machine_time = machine_ready[machine_idx] if 0 <= machine_idx < len(machine_ready) else 0
                start = max(job_time, machine_time)
                end = start + duration

                schedule.append((job, op, machine, start, end))
                
                # Update ready times güvenli şekilde
                if job < len(job_ready):
                    job_ready[job] = end
                if 0 <= machine_idx < len(machine_ready):
                    machine_ready[machine_idx] = end
                    
            except Exception as e:
                print(f" Eksik operasyon {job},{op} eklenemedi: {e}")


    final_count = len(schedule)
    expected_count = num_jobs * num_machines
    
    if final_count != expected_count:
        print(f" Son kontrol: {final_count}/{expected_count} operasyon")
    
    critical_path_ops = []
    if schedule:  # Boş schedule kontrolü
        try:
            critical_path_ops, _ = advanced_critical_path_analysis(schedule)
        except Exception as e:
            if len(schedule) > 0:  # Sadece operasyon varsa uyarı ver
                print(f" Kritik yol analizi başarısız: {e}")
            critical_path_ops = []

    return schedule, critical_path_ops

# Kritik yol analiz    
def advanced_critical_path_analysis(schedule):
    if not schedule:
        return [], {}
    
    schedule_dict = {}
    for job, op_idx, machine, start, end in schedule:
        schedule_dict[(job, op_idx)] = (machine, start, end)
    
    makespan = max(end for _, _, _, _, end in schedule)
    
    critical_path = []
    analysis_info = {
        'makespan': makespan,
        'critical_machines': set(),
        'bottleneck_jobs': set(),
        'machine_utilization': {}
    }

    end_operations = []
    for job in range(num_jobs):
        last_op = num_machines - 1
        if (job, last_op) in schedule_dict:
            _, start, end = schedule_dict[(job, last_op)]
            if abs(end - makespan) < 1e-6:
                end_operations.append((job, last_op))
    
    if not end_operations:
        return critical_path, analysis_info
    
    longest_path = []
    for end_op in end_operations:
        path = _find_critical_path_from_operation(end_op, schedule_dict)
        if len(path) > len(longest_path):
            longest_path = path
    
    critical_path = longest_path
    
    for job, op_idx in critical_path:
        if (job, op_idx) in schedule_dict:
            machine, _, _ = schedule_dict[(job, op_idx)]
            analysis_info['critical_machines'].add(machine)
            analysis_info['bottleneck_jobs'].add(job)
    
    machine_times = defaultdict(float)
    for _, _, machine, start, end in schedule:
        machine_times[machine] += (end - start)
    
    for machine, total_time in machine_times.items():
        analysis_info['machine_utilization'][machine] = total_time / makespan
    
    return critical_path, analysis_info

def _find_critical_path_from_operation(start_op, schedule_dict):
    path = []
    current_op = start_op
    
    while current_op:
        path.append(current_op)
        job, op_idx = current_op
        
        if current_op not in schedule_dict:
            break
        
        machine, start, end = schedule_dict[current_op]
        next_op = None
        
        # 1. Job precedence kontrolü
        if op_idx > 0:
            prev_job_op = (job, op_idx - 1)
            if prev_job_op in schedule_dict:
                _, prev_start, prev_end = schedule_dict[prev_job_op]
                if abs(prev_end - start) < 1e-6:
                    next_op = prev_job_op
        
        # 2. Machine precedence kontrolü
        if next_op is None:
            for other_job in range(num_jobs):
                for other_op in range(num_machines):
                    other = (other_job, other_op)
                    if other != current_op and other in schedule_dict:
                        other_machine, other_start, other_end = schedule_dict[other]
                        if other_machine == machine and abs(other_end - start) < 1e-6:
                            next_op = other
                            break
                if next_op:
                    break
        
        current_op = next_op
    
    path.reverse()
    return path

def evaluate_jssp_solution(schedule, machine_sequence, processing_times):
    if not schedule:
        return 1e6
    
    num_jobs, num_machines = machine_sequence.shape
    expected_operations = num_jobs * num_machines
    
    # Total operation count check
    if len(schedule) != expected_operations:
        penalty = abs(expected_operations - len(schedule)) * 1000
        return 1e6 + penalty
    
    seen_operations = set()
    job_op_counts = [0] * num_jobs
    
    for job, op_idx, machine_id, start_time, end_time in schedule:
        op_key = (job, op_idx)
        
        # Duplicate operation check
        if op_key in seen_operations:
            return 1e6 + 10000
        seen_operations.add(op_key)
        
        # Time validity check
        if start_time < 0 or end_time <= start_time:
            return 1e6 + 5000
        
        # Index validity check
        if not (0 <= job < num_jobs and 0 <= op_idx < num_machines):
            return 1e6 + 5000
        
        # Machine assignment check
        expected_machine = machine_sequence[job, op_idx]
        if machine_id != expected_machine:
            return 1e6 + 8000
        
        # Processing time check
        expected_duration = processing_times[job, op_idx]
        actual_duration = end_time - start_time
        if abs(actual_duration - expected_duration) > 1e-6:
            return 1e6 + 7000
        
        job_op_counts[job] += 1
    
    # Operation count per job check
    for job, count in enumerate(job_op_counts):
        if count != num_machines:
            penalty = abs(count - num_machines) * 1000
            return 1e6 + penalty
    
    # Precedence constraint check
    job_operations = defaultdict(list)
    for job, op_idx, machine_id, start_time, end_time in schedule:
        job_operations[job].append((op_idx, start_time, end_time))
    
    for job in range(num_jobs):
        ops = sorted(job_operations[job], key=lambda x: x[0])
        expected_indices = list(range(num_machines))
        actual_indices = [op[0] for op in ops]
        
        if actual_indices != expected_indices:
            return 1e6 + 9000
        
        for i in range(len(ops) - 1):
            _, _, end_i = ops[i]
            _, start_j, _ = ops[i + 1]
            if start_j < end_i - 1e-6:
                return 1e6 + 10000
    
    # Machine capacity constraint check
    machine_schedules = defaultdict(list)
    for job, op_idx, machine_id, start_time, end_time in schedule:
        machine_schedules[machine_id].append((start_time, end_time, job, op_idx))
    
    for machine_id, operations in machine_schedules.items():
        operations.sort(key=lambda x: x[0])
        
        for i in range(len(operations) - 1):
            _, end_i, _, _ = operations[i]
            start_j, _, _, _ = operations[i + 1]
            if end_i > start_j + 1e-6:
                return 1e6 + 10000
    
    # Valid solution - return makespan
    makespan = max(end for _, _, _, _, end in schedule)
    return makespan

class PopulationInitializer:
    
    def __init__(self, dim: int, pop_size: int, lb: np.ndarray, ub: np.ndarray):
        self.dim = dim
        self.pop_size = pop_size
        self.lb = lb
        self.ub = ub
        
    def ml_strategy_selection(self) -> np.ndarray:
        """ML-based strategy selection"""
        from ml_framework import MLPopulationInitializer
        if not hasattr(self, '_ml_initializer'):
            self._ml_initializer = MLPopulationInitializer(self)
        return self._ml_initializer.ml_strategy_selection()
    
    def ml_ensemble_strategy(self) -> np.ndarray:
        """ML-based ensemble strategy"""
        from ml_framework import MLPopulationInitializer
        if not hasattr(self, '_ml_initializer'):
            self._ml_initializer = MLPopulationInitializer(self)
        return self._ml_initializer.ml_ensemble_strategy()
    
    def random_uniform(self) -> np.ndarray:
        """Pure random uniform initialization"""
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
    
    def random_normal(self, mean: float = 0.5, std: float = 0.167) -> np.ndarray:
        """Random normal initialization"""
        pop = np.random.normal(mean, std, (self.pop_size, self.dim))
        return np.clip(pop, self.lb, self.ub)
    
    def latin_hypercube(self) -> np.ndarray:
        """Latin Hypercube Sampling initialization"""
        sampler = qmc.LatinHypercube(d=self.dim)
        samples = sampler.random(n=self.pop_size)
        return self.lb + samples * (self.ub - self.lb)
    
    def sobol_sequence(self) -> np.ndarray:
        """Sobol sequence initialization"""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        samples = sampler.random(n=self.pop_size)
        return self.lb + samples * (self.ub - self.lb)
    
    def spt_heuristic(self) -> np.ndarray:
        """Shortest Processing Time heuristic initialization"""
        pop = np.zeros((self.pop_size, self.dim))
        
        for p in range(self.pop_size):
            vector = np.zeros(self.dim)
            noise_level = np.random.uniform(0.01, 0.05)
            
            for job in range(num_jobs):
                op_priorities = []
                
                for op in range(num_machines):
                    idx = job + num_jobs * op
                    proc_time = processing_times[job, op]
                    op_priorities.append((idx, proc_time))
                
                # Sort by processing time (ascending for SPT)
                op_priorities.sort(key=lambda x: x[1] + np.random.uniform(-noise_level, noise_level))
                
                for rank, (idx, _) in enumerate(op_priorities):
                    base_priority = rank / num_machines
                    noise = np.random.normal(0, noise_level)
                    vector[idx] = np.clip(base_priority + noise, 0, 1)
            
            pop[p] = vector
        
        return pop
    
    def lpt_heuristic(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.dim))
        
        for p in range(self.pop_size):
            vector = np.zeros(self.dim)
            noise_level = np.random.uniform(0.01, 0.05)
            
            for job in range(num_jobs):
                op_priorities = []
                
                for op in range(num_machines):
                    idx = job + num_jobs * op
                    proc_time = processing_times[job, op]
                    op_priorities.append((idx, -proc_time))  # Negative for descending
                
                op_priorities.sort(key=lambda x: x[1] + np.random.uniform(-noise_level, noise_level))
                
                for rank, (idx, _) in enumerate(op_priorities):
                    base_priority = rank / num_machines
                    noise = np.random.normal(0, noise_level)
                    vector[idx] = np.clip(base_priority + noise, 0, 1)
            
            pop[p] = vector
        
        return pop
    
    def fifo_heuristic(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.dim))
        
        for p in range(self.pop_size):
            vector = np.zeros(self.dim)
            noise_level = np.random.uniform(0.01, 0.05)
            
            for job in range(num_jobs):
                for op in range(num_machines):
                    idx = job + num_jobs * op
                    base_priority = op / num_machines
                    noise = np.random.normal(0, noise_level)
                    vector[idx] = np.clip(base_priority + noise, 0, 1)
            
            pop[p] = vector
        
        return pop
    
    def critical_ratio_heuristic(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.dim))
        
        for p in range(self.pop_size):
            vector = np.zeros(self.dim)
            noise_level = np.random.uniform(0.01, 0.05)
            
            for job in range(num_jobs):
                op_priorities = []
                
                for op in range(num_machines):
                    idx = job + num_jobs * op
                    total_remaining_time = sum(processing_times[job, op:])
                    remaining_ops = num_machines - op
                    critical_ratio = total_remaining_time / remaining_ops if remaining_ops > 0 else 1e6
                    op_priorities.append((idx, critical_ratio))
                
                op_priorities.sort(key=lambda x: x[1] + np.random.uniform(-noise_level, noise_level))
                
                for rank, (idx, _) in enumerate(op_priorities):
                    base_priority = rank / num_machines
                    noise = np.random.normal(0, noise_level)
                    vector[idx] = np.clip(base_priority + noise, 0, 1)
            
            pop[p] = vector
        
        return pop
    
    def most_work_remaining(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.dim))
        
        for p in range(self.pop_size):
            vector = np.zeros(self.dim)
            noise_level = np.random.uniform(0.01, 0.05)
            
            for job in range(num_jobs):
                op_priorities = []
                
                for op in range(num_machines):
                    idx = job + num_jobs * op
                    remaining_work = sum(processing_times[job, op:])
                    op_priorities.append((idx, -remaining_work))  # Negative for descending
                
                op_priorities.sort(key=lambda x: x[1] + np.random.uniform(-noise_level, noise_level))
                
                for rank, (idx, _) in enumerate(op_priorities):
                    base_priority = rank / num_machines
                    noise = np.random.normal(0, noise_level)
                    vector[idx] = np.clip(base_priority + noise, 0, 1)
            
            pop[p] = vector
        
        return pop
    
    def least_work_remaining(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.dim))
        
        for p in range(self.pop_size):
            vector = np.zeros(self.dim)
            noise_level = np.random.uniform(0.01, 0.05)
            
            for job in range(num_jobs):
                op_priorities = []
                
                for op in range(num_machines):
                    idx = job + num_jobs * op
                    remaining_work = sum(processing_times[job, op:])
                    op_priorities.append((idx, remaining_work))  # Ascending
                
                op_priorities.sort(key=lambda x: x[1] + np.random.uniform(-noise_level, noise_level))
                
                for rank, (idx, _) in enumerate(op_priorities):
                    base_priority = rank / num_machines
                    noise = np.random.normal(0, noise_level)
                    vector[idx] = np.clip(base_priority + noise, 0, 1)
            
            pop[p] = vector
        
        return pop
    
    def pure_heuristic_mix(self) -> np.ndarray:
        quarter = self.pop_size // 4
        remainder = self.pop_size % 4
        
        # Adjust population temporarily for each heuristic
        original_pop_size = self.pop_size
        
        # SPT
        self.pop_size = quarter + (1 if remainder > 0 else 0)
        pop1 = self.spt_heuristic()
        
        # LPT
        self.pop_size = quarter + (1 if remainder > 1 else 0)
        pop2 = self.lpt_heuristic()
        
        # FIFO
        self.pop_size = quarter + (1 if remainder > 2 else 0)
        pop3 = self.fifo_heuristic()
        
        # Critical Ratio
        self.pop_size = quarter
        pop4 = self.critical_ratio_heuristic()
        
        # Restore original population size
        self.pop_size = original_pop_size
        
        return np.vstack([pop1, pop2, pop3, pop4])
    
    def quality_diversity_balance(self, heuristic_ratio: float = 0.6) -> np.ndarray:
        heuristic_count = int(self.pop_size * heuristic_ratio)
        random_count = self.pop_size - heuristic_count
        
        # Heuristic part (mix of best performing heuristics)
        original_pop_size = self.pop_size
        self.pop_size = heuristic_count
        heuristic_pop = self.pure_heuristic_mix()
        
        # Random part
        self.pop_size = random_count
        random_pop = self.latin_hypercube()
        
        # Restore original population size
        self.pop_size = original_pop_size
        
        return np.vstack([heuristic_pop, random_pop])
    
    def elite_seeding(self) -> np.ndarray:
        elite_count = max(1, int(self.pop_size * 0.2))
        diverse_count = self.pop_size - elite_count
        
        # Find best performing heuristic (SPT is often good for JSSP)
        original_pop_size = self.pop_size
        self.pop_size = elite_count
        elite_pop = self.spt_heuristic()
        
        # Diverse methods
        methods_count = diverse_count // 3
        remainder = diverse_count % 3
        
        self.pop_size = methods_count + (1 if remainder > 0 else 0)
        pop1 = self.latin_hypercube()
        
        self.pop_size = methods_count + (1 if remainder > 1 else 0)
        pop2 = self.random_normal()
        
        self.pop_size = methods_count
        pop3 = self.lpt_heuristic()
        
        # Restore original population size
        self.pop_size = original_pop_size
        
        return np.vstack([elite_pop, pop1, pop2, pop3])
    
    def staged_population(self) -> np.ndarray:
        counts = [
            int(self.pop_size * 0.3),  # SPT
            int(self.pop_size * 0.3),  # LPT
            int(self.pop_size * 0.2),  # Random
            int(self.pop_size * 0.2)   # LHS
        ]
        
        # Adjust for rounding errors
        total = sum(counts)
        if total < self.pop_size:
            counts[0] += self.pop_size - total
        
        original_pop_size = self.pop_size
        populations = []
        
        methods = [self.spt_heuristic, self.lpt_heuristic, self.random_uniform, self.latin_hypercube]
        
        for count, method in zip(counts, methods):
            if count > 0:
                self.pop_size = count
                populations.append(method())
        
        # Restore original population size
        self.pop_size = original_pop_size
        
        return np.vstack(populations)
    
    def multi_level_hybrid(self) -> np.ndarray:
        heuristic_count = int(self.pop_size * 0.4)
        lhs_count = int(self.pop_size * 0.3)
        normal_count = self.pop_size - heuristic_count - lhs_count
        
        original_pop_size = self.pop_size
        
        # Heuristic mix
        self.pop_size = heuristic_count
        heuristic_pop = self.pure_heuristic_mix()
        
        # LHS
        self.pop_size = lhs_count
        lhs_pop = self.latin_hypercube()
        
        # Random Normal
        self.pop_size = normal_count
        normal_pop = self.random_normal()
        
        # Restore original population size
        self.pop_size = original_pop_size
        
        return np.vstack([heuristic_pop, lhs_pop, normal_pop])
    
    def problem_size_adaptive(self) -> np.ndarray:
       
        problem_size = num_jobs * num_machines
        
        # Based on your instance dimensions:
        # ta01: 15x15=225, ta11: 20x15=300 (Small)
        # ta21: 20x20=400, ta31: 30x15=450 (Medium)  
        # ta41: 30x20=600, ta51: 50x15=750 (Large)
        
        if problem_size <= 300:  # Small problems (ta01, ta11)
            return self.quality_diversity_balance(heuristic_ratio=0.8)
        elif problem_size <= 450:  # Medium problems (ta21, ta31)
            return self.staged_population()
        else:  # Large problems (ta41, ta51)
            return self.quality_diversity_balance(heuristic_ratio=0.4)
    
    def machine_job_ratio_based(self) -> np.ndarray:
        
        ratio = num_machines / num_jobs
        
        # Based on your instances:
        # ta01: 15/15=1.0, ta21: 20/20=1.0 (Balanced)
        # ta11: 15/20=0.75, ta41: 20/30=0.67 (Slightly more jobs)
        # ta31: 15/30=0.5, ta51: 15/50=0.3 (Much more jobs than machines)
        
        if ratio >= 1.0:  # Equal or more machines than jobs (ta01, ta21)
            return self.multi_level_hybrid()
        elif ratio >= 0.65:  # Slightly more jobs than machines (ta11, ta41)
            return self.staged_population()
        else:  # Much more jobs than machines (ta31, ta51)
            return self.quality_diversity_balance(heuristic_ratio=0.7)

def calculate_diversity(population: np.ndarray) -> float:
   
    if len(population) < 2:
        return 0.0
    
    mean_vector = np.mean(population, axis=0)
    distances = np.linalg.norm(population - mean_vector, axis=1)
    return np.mean(distances)

def calculate_fitness_diversity(fitness_values: np.ndarray) -> float:
    
    if len(fitness_values) < 2:
        return 0.0
    
    std_fitness = np.std(fitness_values)
    mean_fitness = np.mean(fitness_values)
    
    if mean_fitness < 1e-8:
        return 0.0
    
    return std_fitness / mean_fitness

class ExperimentRunner:
    
    
    def __init__(self, pop_size: int = 50, max_iterations: int = 400, num_runs: int = 30):
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.num_runs = num_runs
        self.results = {}
        
        # Define all population methods to test
        self.population_methods = {
            'random_uniform': 'random_uniform',
            'random_normal': 'random_normal',
            'latin_hypercube': 'latin_hypercube',
            'sobol_sequence': 'sobol_sequence',
            'spt_heuristic': 'spt_heuristic',
            'lpt_heuristic': 'lpt_heuristic',
            'fifo_heuristic': 'fifo_heuristic',
            'critical_ratio_heuristic': 'critical_ratio_heuristic',
            'most_work_remaining': 'most_work_remaining',
            'least_work_remaining': 'least_work_remaining',
            'pure_heuristic_mix': 'pure_heuristic_mix',
            'quality_diversity_balance': 'quality_diversity_balance',
            'elite_seeding': 'elite_seeding',
            'staged_population': 'staged_population',
            'multi_level_hybrid': 'multi_level_hybrid',
            'problem_size_adaptive': 'problem_size_adaptive',
            'machine_job_ratio_based': 'machine_job_ratio_based'
        }
        
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
    
    def run_single_experiment(self, instance_name: str, method_name: str, run_id: int) -> Dict[str, Any]:
      
        print(f"Running {instance_name} - {method_name} - Run {run_id}")
        
        # Update problem instance
        update_problem_instance(instance_name)
        
        # Initialize population
        dim = num_jobs * num_machines
        lb = np.zeros(dim)
        ub = np.ones(dim)
        
        initializer = PopulationInitializer(dim, self.pop_size, lb, ub)
        init_method = getattr(initializer, method_name)
        
        # Get initial population
        population = init_method()
        
        # Evaluate initial population
        fitness_values = np.array([evaluate_jssp_solution(
            cached_decode(tuple(ind))[0], machine_sequence, processing_times
        ) for ind in population])
        
        # Calculate initial diversity statistics
        diversity_stats = self.calculate_population_diversity_stats(population, fitness_values)
        
        # PSO optimization
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
            # Update inertia weight (linearly decreasing)
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
            current_diversity = calculate_diversity(population)
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
        
        return {
            'instance': instance_name,
            'method': method_name,
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
            'execution_time': 0.0  # Will be set by caller
        }
    
    def _local_search(self, solution: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
       
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
    
    def run_experiments(self, instances: List[str], methods: List[str] = None) -> pd.DataFrame:
        
        if methods is None:
            methods = list(self.population_methods.keys())
        
        all_results = []
        total_experiments = len(instances) * len(methods) * self.num_runs
        experiment_count = 0
        
        print(f"Starting experiments: {len(instances)} instances × {len(methods)} methods × {self.num_runs} runs = {total_experiments} total")
        
        for instance in instances:
            for method in methods:
                instance_method_results = []
                
                for run in range(self.num_runs):
                    experiment_count += 1
                    start_time = time.time()
                    
                    try:
                        result = self.run_single_experiment(instance, method, run)
                        result['execution_time'] = time.time() - start_time
                        instance_method_results.append(result)
                        all_results.append(result)
                        
                        print(f" ({experiment_count}/{total_experiments}) {instance}-{method}-{run}: {result['best_fitness']:.2f}")
                        
                    except Exception as e:
                        print(f" ({experiment_count}/{total_experiments}) {instance}-{method}-{run}: Error - {e}")
                        continue
                
                # Save intermediate results
                if instance_method_results:
                    self._save_intermediate_results(instance, method, instance_method_results)

        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        return df
    

    def _save_intermediate_results(self, instance: str, method: str, results: List[Dict]):
       
        filename = f"intermediate_{instance}_{method}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    
    def _get_problem_size(self, instance_name: str) -> str:
        size_map = {         
            # Small problems
            'ta01': 'small',  # 15x15=225
            'ta11': 'small',  # 20x15=300
            # Medium problems  
            'ta21': 'medium', # 20x20=400
            'ta31': 'medium', # 30x15=450
            # Large problems
            'ta41': 'large',  # 30x20=600
            'ta51': 'large',  # 50x15=750
        }
        return size_map.get(instance_name, 'unknown')
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        analysis = {}
        
        # Overall statistics by method
        method_stats = df.groupby('method').agg({
            'best_fitness': ['mean', 'std', 'min', 'max'],
            'initial_diversity': ['mean', 'std'],
            'final_diversity': ['mean', 'std'],
            'initial_best_worst_diversity': ['mean', 'std'],
            'final_best_worst_diversity': ['mean', 'std'],
            'convergence_iteration': ['mean', 'std'],
            'execution_time': ['mean', 'std']
        }).round(4)
        
        analysis['method_statistics'] = method_stats
        
        # Best performing methods
        method_ranking = df.groupby('method')['best_fitness'].mean().sort_values()
        analysis['method_ranking'] = method_ranking
        
        # Problem size analysis
        df['problem_size'] = df['instance'].apply(self._get_problem_size)
        size_analysis = df.groupby(['problem_size', 'method'])['best_fitness'].mean().unstack()
        analysis['size_analysis'] = size_analysis
        
        # Diversity analysis
        diversity_analysis = df.groupby('method').agg({
            'initial_diversity': 'mean',
            'final_diversity': 'mean',
            'initial_best_worst_diversity': 'mean',
            'final_best_worst_diversity': 'mean'
        }).round(4)
        analysis['diversity_analysis'] = diversity_analysis
        
        # Convergence analysis
        convergence_analysis = df.groupby('method').agg({
            'convergence_iteration': ['mean', 'std', 'min', 'max'],
            'iterations_completed': ['mean', 'std']
        }).round(2)
        analysis['convergence_analysis'] = convergence_analysis
        
        return analysis
    
    
    def save_results(self, df: pd.DataFrame, analysis: Dict[str, Any], filename: str = 'jssp_population_results'):
        
        # Save raw data
        df.to_csv(f'{filename}_raw.csv', index=False)
        
        # Save analysis
        with open(f'{filename}_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save summary statistics
        summary = analysis['method_statistics']
        summary.to_csv(f'{filename}_summary.csv')
        
       
        print(f"- Raw data: {filename}_raw.csv")
        print(f"- Analysis: {filename}_analysis.json") 
        print(f"- Summary: {filename}_summary.csv")
        
    def save_instance_method_results(self, instance_name: str, method_name: str, run_results: List[Dict]):
        
        filename = f"{instance_name}_{method_name}.json"
        
        # Extract only the essential data for each run
        summary_data = {
            'instance': instance_name,
            'method': method_name,
            'problem_size': f"{num_jobs}x{num_machines}",
            'total_runs': len(run_results),
            'run_results': []
        }
        
        for result in run_results:
            run_summary = {
                'run_id': result['run_id'],
                'best_fitness': result['best_fitness'],
                'initial_diversity': result['initial_diversity'],
                'initial_best_worst_diversity': result['initial_best_worst_diversity'],
                'initial_mean_diversity': result['initial_mean_diversity'],
                'final_diversity': result['final_diversity'],
                'final_best_worst_diversity': result['final_best_worst_diversity'],
                'final_mean_diversity': result['final_mean_diversity'],
                'convergence_iteration': result['convergence_iteration'],
                'iterations_completed': result['iterations_completed'],
                'execution_time': result['execution_time']
            }
            summary_data['run_results'].append(run_summary)
        
        # Calculate statistics across runs
        fitness_values = [r['best_fitness'] for r in run_results]
        summary_data['statistics'] = {
            'best_fitness': min(fitness_values),
            'worst_fitness': max(fitness_values),
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'median_fitness': np.median(fitness_values),
            'success_rate': sum(1 for f in fitness_values if f < 1e6) / len(fitness_values),
            'mean_convergence_iteration': np.mean([r['convergence_iteration'] for r in run_results]),
            'mean_execution_time': np.mean([r['execution_time'] for r in run_results])
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f" Saved {len(run_results)} runs to {filename}")

