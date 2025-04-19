import sys
import time
from Bso import BSO, Solution
from Dfs import DFSSolver
from gridSearch import grid_search, plot_convergence
from problem import JSSPProblem
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_gantt(solution):
    problem = solution.problem
    job_end = [0] * problem.get_num_jobs()
    machine_end = {}
    tasks = []

    for op in solution.permutation:
        job_id, step = op
        machine, duration = problem.jobs[job_id][step]

        start = max(job_end[job_id], machine_end.get(machine, 0))
        end = start + duration

        job_end[job_id] = end
        machine_end[machine] = end

        tasks.append((machine, start, end, f"J{job_id}O{step}"))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab20", len(problem.jobs))

    for machine, start, end, label in tasks:
        ax.barh(y=f"M{machine}", width=end-start, left=start,
                height=0.4, align="center", color=colors(int(label[1])), edgecolor='black')
        ax.text(start + (end-start)/2, f"M{machine}", label,
                ha='center', va='center', color='white', fontsize=8)

    ax.set_xlabel("Temps")
    ax.set_title("Diagramme de Gantt - Job Shop Scheduling")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def lire_instance_taillard(chemin_fichier):
    jobs_data = []
    with open(chemin_fichier, 'r') as f:
        lines = f.readlines()
    
    # Chercher les lignes
    i = 0
    # while i < len(lines):
    line = lines[i].strip()
    
    if line.startswith("Nb of jobs"):         
        i += 1
        njobs, nmachines, *_ = map(int, lines[i].split())
        i+=1
          
    line=lines[i]
    if "Times" in line:
        i += 1
        processing_times = []
        
        for _ in range(njobs):
            processing_times.append(list(map(int, lines[i].split())))
            i += 1
            
    line=lines[i]        
    if "Machines" in line:
            i += 1
            machines = []
            for _ in range(njobs):
                machines.append(list(map(lambda x: int(x)-1, lines[i].split())))  # machine indices start from 0
                i += 1

            print(machines)
                # Fusionner machines et durées
            for job_id in range(njobs):
                job = []
                for op_id in range(nmachines):
                    m_id = machines[job_id][op_id]
                    time = processing_times[job_id][op_id]
                    job.append((m_id, time))
                jobs_data.append(job)
    
      

    return jobs_data



import time
from typing import List, Tuple
import matplotlib.pyplot as plt

def main():
    print("Job Shop Scheduling Problem Solver")
    print("Choose an algorithm:")
    print("1. Bee Swarm Optimization (BSO)")
    print("2. Depth-First Search (DFS - exact method)")
    
    algorithm_choice = input("Enter your choice (1 or 2): ")
    
    print("\nAvailable instances:")
    print("1. tai20_15.txt (20 jobs, 15 machines)")
    print("2. tai30_15.txt (30 jobs, 15 machines)")
    print("3. tai50_15.txt (50 jobs, 15 machines)")
    print("4. tai100_20.txt (100 jobs, 20 machines)")
    print("5. 2 jobs, 3 machines")
    
    instance_choice = input("Enter instance number (1-5): ")
    
    # Map instance choices to file names
    instance_files = {
        '1': "dataSet/tai20_15.txt",
        '2': "dataSet/tai30_15.txt",
        '3': "dataSet/tai50_15.txt",
        '4': "dataSet/tai100_20.txt",
        '5': "dfs",
    }
    
    # Best parameters for each instance (from your table)
    instance_params = {
        '1': {'flip': 8, 'max_iter': 150, 'k': 9, 'max_steps': 25, 'max_chances': 3},
        '2': {'flip': 10, 'max_iter': 150, 'k': 9, 'max_steps': 35, 'max_chances': 3},
        '3': {'flip': 10, 'max_iter': 150, 'k': 9, 'max_steps': 35, 'max_chances': 5},
        '4': {'flip': 11, 'max_iter': 150, 'k': 11, 'max_steps': 35, 'max_chances': 5},
        '5': {'flip': 8, 'max_iter': 150, 'k': 9, 'max_steps': 25, 'max_chances': 3}
    }
    
    # Load the problem instance
    chemin = instance_files[instance_choice]
    if chemin == "dfs":
        jobs = [
            [(0, 2), (1, 2), (3,4)],  # Job 0: Machine 0 for 2 time units, then Machine 1 for 2 time units
            [(1, 2), (0, 5), (3,1)]   # Job 1: Machine 1 for 2 time units, then Machine 0 for 2 time units
        ]
    else:
        jobs = lire_instance_taillard(chemin)
    
    problem = JSSPProblem(jobs)
    
    if algorithm_choice == '1':
        # Run BSO with best parameters for selected instance
        params = instance_params[instance_choice]
        print(f"\nRunning BSO with parameters: {params}")
        
        bso = BSO(
            problem,
            flip=params['flip'],
            max_iter=params['max_iter'],
            k=params['k'],
            max_steps=params['max_steps'],
            max_chances=params['max_chances']
        )
        
        best_overall_schedule = None
        best_overall_makespan = float('inf')
        best_overall_history = None
        best_time = float('inf')
        best_solution = None
        
        num_runs = 5  # Number of independent runs
        
        for i in range(num_runs):
            print(f"\nRun {i + 1}/{num_runs}...")
            start_time = time.time()
            schedule, makespan, history = bso.run()
            run_time = time.time() - start_time
            
            print(f"Run {i + 1} - Makespan: {makespan}, Time: {run_time:.2f} seconds")
            
            if makespan < best_overall_makespan:
                best_overall_makespan = makespan
                best_overall_schedule = schedule
                best_overall_history = history
                best_time = run_time
                best_solution = bso.best_solution
        
        print(f"\nBest solution found - Makespan: {best_overall_makespan}, Time: {best_time:.2f} seconds")
        
        # Plot convergence history
        if best_overall_history:
            iterations = [x[0] for x in best_overall_history]
            makespans = [x[1] for x in best_overall_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, makespans, 'b-')
            plt.title('Convergence History')
            plt.xlabel('Iteration')
            plt.ylabel('Makespan')
            plt.grid(True)
            plt.show()
        
        # Plot Gantt chart
        plot_gantt(best_solution)
        
    elif algorithm_choice == '2':
        # Run DFS
        if int(instance_choice) != 5:
            print("\nWarning: DFS is only practical for very small instances.")
            print("Larger instances may take prohibitively long time or run out of memory.")
            proceed = input("Do you want to continue anyway? (y/n): ")
            if proceed.lower() != 'y':
                return
            
        
        print("\nRunning DFS (this may take a while for larger instances)...")
        dfs = DFSSolver(problem)
        start_time = time.time()
        makespan, schedule = dfs.solve()
        run_time = time.time() - start_time
        
        print(f"\nOptimal solution found - Makespan: {makespan}, Time: {run_time:.2f} seconds")
        
        # Plot Gantt chart
        if schedule:
            # Need to convert DFS schedule to Solution object for plotting
            permutation = []
            for op in schedule:
                permutation.append((op[0], op[1]))
            
            solution = Solution(problem, permutation)
            solution.schedule = schedule
            solution.makespan = makespan
            plot_gantt(solution)
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()



















# not for now



# def read_makespan_history(filename):
#     iterations = []
#     makespans = []
#     with open(filename, 'r') as file:
#         for line in file:
#             if line.startswith("Iteration"):
#                 parts = line.strip().split(":")
#                 iteration = int(parts[0].split()[1])
#                 makespan = int(parts[1])
#                 iterations.append(iteration)
#                 makespans.append(makespan)
#     return iterations, makespans

# def plot_makespan_convergence(filename, threshold=1376):
#     iterations, makespans = read_makespan_history(filename)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(iterations, makespans, label="Makespan per Iteration", color='blue')
#     plt.axhline(y=threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")

#     plt.xlabel('Iteration')
#     plt.ylabel('Makespan')
#     plt.title('Makespan vs. Iteration')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()