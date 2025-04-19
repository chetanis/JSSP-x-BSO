import matplotlib.pyplot as plt
import itertools
import time
from tqdm import tqdm

from Bso import BSO  # For progress bars

def grid_search(problem, param_grid, num_runs=3):
    """
    Perform grid search to find best parameters for BSO.
    
    Args:
        problem: The JSSP problem instance
        param_grid: Dictionary with parameter names and possible values
        num_runs: Number of runs for each parameter combination
    
    Returns:
        best_params: Dictionary with best parameter combination
        best_makespan: Best makespan found
        results: Full results of all parameter combinations
    """
    results = []
    best_makespan = float('inf')
    best_params = None
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Running grid search with {len(param_combinations)} parameter combinations...")
    
    for params in tqdm(param_combinations):
        param_dict = {name: value for name, value in zip(param_names, params)}
        
        avg_makespan = 0
        avg_time = 0
        
        for run in range(num_runs):
            bso = BSO(
                problem,
                flip=param_dict.get('flip', 4),
                max_iter=param_dict.get('max_iter', 100),
                k=param_dict.get('k', 20),
                local_iter=param_dict.get('local_iter', 20),
                taboo_size=param_dict.get('taboo_size', 50)
            )
            
            start_time = time.time()
            best_schedule, run_makespan, history = bso.run()
            run_time = time.time() - start_time
            
            avg_makespan += run_makespan
            avg_time += run_time
        
        avg_makespan /= num_runs
        avg_time /= num_runs
        
        results.append({
            'params': param_dict,
            'avg_makespan': avg_makespan,
            'avg_time': avg_time
        })
        
        if avg_makespan < best_makespan:
            best_makespan = avg_makespan
            best_params = param_dict
    
    # Sort results by makespan
    results.sort(key=lambda x: x['avg_makespan'])
    
    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Parameters: {result['params']}")
        print(f"   Avg Makespan: {result['avg_makespan']:.2f}, Avg Time: {result['avg_time']:.2f}s")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best average makespan: {best_makespan:.2f}")
    
    return best_params, best_makespan, results

def plot_convergence(history, title="BSO Convergence"):
    """
    Plot the convergence of the makespan over iterations.
    
    Args:
        history: List of (iteration, makespan) tuples
        title: Title for the plot
    """
    iterations, makespans = zip(*history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, makespans)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Makespan")
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()