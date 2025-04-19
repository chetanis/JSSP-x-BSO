import sys
import time
from Bso import BSO
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



import matplotlib.pyplot as plt

def read_makespan_history(filename):
    iterations = []
    makespans = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Iteration"):
                parts = line.strip().split(":")
                iteration = int(parts[0].split()[1])
                makespan = int(parts[1])
                iterations.append(iteration)
                makespans.append(makespan)
    return iterations, makespans

def plot_makespan_convergence(filename, threshold=1376):
    iterations, makespans = read_makespan_history(filename)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, makespans, label="Makespan per Iteration", color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")

    plt.xlabel('Iteration')
    plt.ylabel('Makespan')
    plt.title('Makespan vs. Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Data
sizes = ['20x15', '30x15', '50x15', '100x20']
jobs_machines = [300, 450, 750, 2000]  # jobs * machines
times_bso = [20, 34, 98, 458]

times_hours = [t / 60 for t in times_bso]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(jobs_machines, times_hours, marker='o', color='navy', linewidth=2)
plt.xticks(jobs_machines, sizes)
plt.xlabel('Taille de la matrice (Jobs x Machines)')
plt.ylabel('Temps (minutes)')
plt.title('Temps d\'exécution de l\'algorithme BSO selon la taille du problème JSSP')
plt.grid(True)
plt.tight_layout()
plt.show()



# Example usage:
# plot_makespan_convergence("results_tai100_20_modified2.txt",threshold=5464)


# print(sys.version)


# if __name__ == "__main__":
#     # Définir le problème (exemple simple)
#     chemin = "dataSet/tai20_15.txt"  # Remplace avec le vrai nom du fichier

#     jobs = lire_instance_taillard(chemin)
   
#     problem = JSSPProblem(jobs)

#     # solver = DFSSolver(problem)
#     # Grid search parameters
#     # param_grid = {
#     #     'flip': [8 , 9,10],
#     #     'max_iter': [150],
#     #     'k': [9,10,11],
#     #     'local_iter': [25, 30, 35],
#     #     'taboo_size': [25, 30, 35]
#     # }
    
#     # # Run grid search
#     # print("Starting grid search...")
#     # best_params, best_makespan, results = grid_search(problem, param_grid, num_runs=2)


    
    
#     print("\nRunning BSO with best parameters...")
#     # Run BSO with best parameters
#     bso = BSO(
#         problem,
#         flip=8,
#         max_iter=150,
#         k=9,
#         local_iter=25,
#         taboo_size=35
#     )
#     best_overall_schedule = None
#     best_overall_makespan = float('inf')
#     best_overall_history = None
#     besttime = float('inf')

#     for i in range(5):
#         print(f"Run {i + 1}...")
#         start_time = time.time()
#         schedule,makespan , history= bso.run()
#         run_time = time.time() - start_time

#         print(f"Run {i + 1} - Makespan: {makespan}, Time: {run_time:.2f} seconds")

#         if makespan < best_overall_makespan:
#             best_overall_makespan = makespan
#             best_overall_schedule = schedule
#             best_overall_history = history
#             besttime = run_time

#     best_schedule = best_overall_schedule
#     best_makespan = best_overall_makespan
#     history = best_overall_history

#     plot_gantt(bso.best_solution)
    
    
    # Create convergence plot
    # instance_name =  in.split('/')[-1].split('.')[0]
    # instance_name =  "tai6_6"
    # plot_convergence(history, f"BSO Convergence - {instance_name}")
    
    # print(f"Best makespan: {best_makespan}")
    
    # # Save results to files for future reference
    # with open(f"results_dfs_{instance_name}.txt", "w") as f:
    #     f.write(f"Instance: {instance_name}\n")
    #     # f.write(f"Best parameters: {best_params}\n")
    #     f.write(f"Best makespan: {best_makespan}\n")
    #     f.write(f"Optimal run time: {besttime:.2f} seconds\n")
    #     f.write("\nMakespan history:\n")
    #     for iteration, makespan in history:
    #         f.write(f"Iteration {iteration}: {makespan}\n")

