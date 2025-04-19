import random

class Solution:
    def __init__(self, problem, permutation=None):
        self.problem = problem
        self.permutation = permutation if permutation else self.generate_random()
        self.makespan = None
        self.schedule = None  # Pour stocker le planning détaillé

    def generate_random(self):
        job_ops = []
        for job_id in range(self.problem.get_num_jobs()):
            ops = [(job_id, step) for step in range(self.problem.get_num_operations(job_id))]
            job_ops.append(ops)
    
        permutation = []
        pointers = [0] * len(job_ops)
        total_ops = sum(len(ops) for ops in job_ops)
        
        while len(permutation) < total_ops:
            available_jobs = [i for i, ops in enumerate(job_ops) if pointers[i] < len(ops)]
            chosen_job = random.choice(available_jobs)
            permutation.append((chosen_job, pointers[chosen_job]))
            pointers[chosen_job] += 1
            
        return permutation

    def evaluate(self):
        job_end = [0] * self.problem.get_num_jobs()
        machine_end = {}
        schedule = []

        for job_id, step in self.permutation:
            machine, duration = self.problem.jobs[job_id][step]
            start = max(job_end[job_id], machine_end.get(machine, 0))
            end = start + duration
            job_end[job_id] = end
            machine_end[machine] = end
            schedule.append((job_id, step, machine, start, end))

        self.makespan = max(job_end)
        self.schedule = schedule
        return self.makespan

    def is_valid_permutation(self, permutation):
        """Vérifie si la permutation respecte les contraintes de précédence."""
        last_seen = {}
        for index, (job_id, step) in enumerate(permutation):
            if step > 0:
                # Vérifie si l'opération précédente a déjà été vue
                if (job_id, step - 1) not in last_seen:
                    return False
                if last_seen[(job_id, step - 1)] > index:
                    return False
            last_seen[(job_id, step)] = index
        return True

    def repair_permutation(self, permutation):
        """Répare une permutation invalide en respectant les contraintes de précédence."""
        job_count = self.problem.get_num_jobs()
        op_counts = [self.problem.get_num_operations(i) for i in range(job_count)]
        
        # Comptage des opérations par job
        job_op_count = {}
        for job_id, step in permutation:
            if job_id not in job_op_count:
                job_op_count[job_id] = {}
            job_op_count[job_id][step] = job_op_count[job_id].get(step, 0) + 1
        
        # Identification des opérations manquantes ou en double
        missing_ops = []
        duplicate_ops = []
        
        for job_id in range(job_count):
            for step in range(op_counts[job_id]):
                if job_id not in job_op_count or step not in job_op_count[job_id]:
                    missing_ops.append((job_id, step))
                elif job_op_count[job_id][step] > 1:
                    for _ in range(job_op_count[job_id][step] - 1):
                        duplicate_ops.append((job_id, step))
        
        # Remplacer les opérations en double par les manquantes
        fixed_perm = permutation.copy()
        for i, (job_id, step) in enumerate(fixed_perm):
            if (job_id, step) in duplicate_ops:
                if missing_ops:
                    fixed_perm[i] = missing_ops.pop(0)
                    duplicate_ops.remove((job_id, step))
        
        # Trier pour respecter les contraintes de précédence
        job_step_indices = {}
        for i, (job_id, step) in enumerate(fixed_perm):
            if job_id not in job_step_indices:
                job_step_indices[job_id] = {}
            job_step_indices[job_id][step] = i
        
        # Vérifier et corriger l'ordre des étapes
        for job_id in job_step_indices:
            steps = sorted(job_step_indices[job_id].keys())
            for i in range(1, len(steps)):
                if job_step_indices[job_id][steps[i-1]] > job_step_indices[job_id][steps[i]]:
                    # Échanger les positions
                    pos1 = job_step_indices[job_id][steps[i-1]]
                    pos2 = job_step_indices[job_id][steps[i]]
                    fixed_perm[pos1], fixed_perm[pos2] = fixed_perm[pos2], fixed_perm[pos1]
                    # Mettre à jour les indices
                    job_step_indices[job_id][steps[i-1]] = pos2
                    job_step_indices[job_id][steps[i]] = pos1
        
        return fixed_perm

    def swap(self, idx1, idx2):
        """Échange deux opérations si possible."""
        new_perm = self.permutation.copy()
        new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]
        
        # Vérification de la validité
        if not self.is_valid_permutation(new_perm):
            new_perm = self.repair_permutation(new_perm)
            
        return Solution(self.problem, permutation=new_perm)
    
    def invert_segment(self, start_idx, length):
        """Inverse un segment de la permutation."""
        if start_idx + length >= len(self.permutation):
            length = len(self.permutation) - start_idx - 1
            
        if length <= 0:
            return Solution(self.problem, self.permutation.copy())
            
        new_perm = self.permutation.copy()
        segment = new_perm[start_idx:start_idx+length+1]
        segment.reverse()
        new_perm[start_idx:start_idx+length+1] = segment
        
        # Vérification de la validité
        if not self.is_valid_permutation(new_perm):
            new_perm = self.repair_permutation(new_perm)
            
        return Solution(self.problem, permutation=new_perm)
    
    def get_critical_path(self):
        """Identifie le chemin critique dans l'ordonnancement."""
        if not self.schedule:
            self.evaluate()
            
        # Trier le planning par temps de fin
        sorted_schedule = sorted(self.schedule, key=lambda x: x[4], reverse=True)
        
        # Le makespan est le temps de fin maximal
        makespan = sorted_schedule[0][4]
        
        # Identifier les opérations sur le chemin critique
        critical_ops = []
        current_time = makespan
        
        while current_time > 0:
            critical_op = None
            for op in sorted_schedule:
                job_id, step, machine, start, end = op
                if end == current_time:
                    critical_op = op
                    break
                    
            if critical_op:
                critical_ops.append(critical_op)
                current_time = critical_op[3]  # Start time devient la nouvelle cible
            else:
                break
                
        return critical_ops

class BSO:
    def __init__(self, problem, flip=3, max_iter=100, k=20, max_steps=10, taboo_size=50,max_chances=3):
        self.problem = problem
        self.flip = flip  # Détermine le nombre de variables à inverser (n/flip)
        self.max_iter = max_iter  # Nombre maximum d'itérations
        self.k = k  # Taille de la population/zone de recherche
        self.max_steps = max_steps  # Nombre d'itérations pour la recherche locale
        self.best_solution = None
        self.sref = None
        self.nb_chances = max_chances
        self.max_chances = max_chances
        self.dense_list = []  # Liste des meilleures solutions locales
        self.taboo_list = []  # Liste tabou
        self.taboo_size = taboo_size  # Taille maximale de la liste tabou
         # Seuil pour la diversification
        
    def hash_solution(self, solution):
        """Crée un hash unique pour une solution."""
        return tuple(solution.permutation)
    def hamming_distance(self, solution1, solution2):
        """Calcule la distance de Hamming entre deux solutions."""
        distance = 0
        for i in range(len(solution1.permutation)):
            if solution1.permutation[i] != solution2.permutation[i]:
                distance += 1
        return distance
      
    def generate_search_area(self, sref):
        """Génère la zone de recherche en inversant n/flip variables depuis Sref."""
        search_area = [sref]  # Inclure sref dans la zone de recherche
        
        # Calculer le nombre de variables à inverser
        n = len(sref.permutation)
        num_to_invert = max(1, int(n / self.flip))
        
        # Générer k-1 solutions dans la zone de recherche
        for _ in range(self.k - 1):
            # Sélectionner aléatoirement une position de départ
            start_pos = random.randint(0, n - num_to_invert)
            
            # Créer une nouvelle solution en inversant le segment
            new_sol = sref.invert_segment(start_pos, num_to_invert - 1)
            new_sol.evaluate()
            
            search_area.append(new_sol)
        
        return search_area

    def apply_local_search(self, solution, intensity=1.0):
        """Applique une recherche locale guidée par le chemin critique."""
        improved_solution = solution
        best_makespan = solution.makespan
        
        for _ in range(int(self.max_steps * intensity)):
            # Identifier le chemin critique
            critical_ops = improved_solution.get_critical_path()
            
            if not critical_ops or len(critical_ops) < 2:
                # Pas assez d'opérations critiques, utiliser une approche standard
                idx1 = random.randint(0, len(improved_solution.permutation) - 1)
                idx2 = random.randint(0, len(improved_solution.permutation) - 1)
                new_solution = improved_solution.swap(idx1, idx2)
            else:
                # Trouver les indices dans la permutation des opérations critiques
                critical_indices = []
                for critical_op in critical_ops:
                    job_id, step = critical_op[0], critical_op[1]
                    for i, (j, s) in enumerate(improved_solution.permutation):
                        if j == job_id and s == step:
                            critical_indices.append(i)
                            break
                
                if len(critical_indices) >= 2:
                    # Échanger deux opérations critiques adjacentes
                    idx = random.randint(0, len(critical_indices) - 2)
                    new_solution = improved_solution.swap(critical_indices[idx], critical_indices[idx + 1])
                else:
                    # Fallback à un échange aléatoire
                    idx1 = random.randint(0, len(improved_solution.permutation) - 1)
                    idx2 = random.randint(0, len(improved_solution.permutation) - 1)
                    new_solution = improved_solution.swap(idx1, idx2)
            
            new_solution.evaluate()
            
            # Accepter la nouvelle solution si elle est meilleure
            if new_solution.makespan < best_makespan:
                improved_solution = new_solution
                best_makespan = new_solution.makespan
        
        return improved_solution

    def update_dense_list(self, solution):
        """Met à jour la liste dense avec une nouvelle solution."""
        # Vérifier si la solution est déjà dans la liste
        solution_hash = self.hash_solution(solution)
        
        for existing_sol in self.dense_list:
            if self.hash_solution(existing_sol) == solution_hash:
                return  # Solution déjà présente
        
        self.dense_list.append(solution)
        # Trier par makespan pour garder les meilleures solutions
        self.dense_list.sort(key=lambda s: s.makespan)
        if len(self.dense_list) > self.k:
            self.dense_list = self.dense_list[:self.k]
 # # Diversification si pas d'amélioration
        # if self.no_improvement_count > self.max_no_improvement:
        #     # Reset counter
        #     self.no_improvement_count = 0
        #     # Choisir aléatoirement parmi les meilleures solutions
        #     idx = random.randint(0, min(5, len(self.dense_list) - 1))
        #     return self.dense_list[idx]
   
    def select_sref(self):
        """Sélectionne la prochaine solution de référence (sref)."""
        min_distance = float('inf')
        closest_sol = None
        if not self.dense_list:
            return Solution(self.problem)

        # Sélection élitiste : meilleure solution
        sbest = self.dense_list[0]
        delta_f = sbest.makespan - self.sref.makespan
        
        if delta_f < 0:
            # Amélioration trouvée
            self.nb_chances = self.max_chances
            sbest_hash = self.hash_solution(sbest)
            if sbest_hash in self.taboo_list:
                # Si la meilleure est taboue, prendre la meilleure non taboue
                for sol in self.dense_list:
                    sol_hash = self.hash_solution(sol)
                    if sol_hash not in self.taboo_list:
                        return sol
                # Si toutes sont taboues, on retourne quand même sbest
            return sbest
        else:
            
            # Pas d'amélioration
            self.nb_chances -= 1
            if self.nb_chances > 0:
                return self.sref
            else:
                self.dense_list.remove(sbest)
                # Exploitation locale : choisir la solution la plus proche de sbest
                for sol in self.dense_list:
                    dist = self.hamming_distance(sbest, sol)
                    if dist < min_distance:
                        closest_sol = sol
                        min_distance = dist
                self.nb_chances = self.max_chances         
                return closest_sol if closest_sol else self.sref
    def update_taboo_list(self, solution):
        """Ajoute une solution à la liste tabou et maintient sa taille."""
        solution_hash = self.hash_solution(solution)
        if solution_hash not in self.taboo_list:
            self.taboo_list.append(solution_hash)
            
        # Limiter la taille de la liste tabou
        if len(self.taboo_list) > self.taboo_size:
            self.taboo_list.pop(0)  # Supprimer le plus ancien
    
    def run(self):
        """Exécute l'algorithme BSO."""
        # Initialisation
        self.sref = Solution(self.problem)
        self.sref.evaluate()
        self.best_solution = self.sref
        self.dense_list = [self.sref]
        history = []
        # Paramètres adaptatifs
        intensification_ratio = 1.0
        for iteration in range(self.max_iter):
            
            # Générer la zone de recherche
            search_area = self.generate_search_area(self.sref)
            
            # Appliquer la recherche locale à chaque solution
            for sol in search_area:
                sol = self.apply_local_search(sol, intensification_ratio)
                sol.evaluate()
                self.update_dense_list(sol)
            
            # Mettre à jour la meilleure solution
            if self.dense_list and self.dense_list[0].makespan < self.best_solution.makespan:
                self.best_solution = self.dense_list[0]
               
                # Augmenter l'intensification quand on trouve une meilleure solution
                intensification_ratio = min(2.0, intensification_ratio * 1.1)
            else:
                
                # Diminuer l'intensification pour favoriser la diversification
                intensification_ratio = max(0.5, intensification_ratio * 0.95)
            
            # Sélectionner la nouvelle solution de référence
            self.sref = self.select_sref()
            self.update_taboo_list(self.sref)
            
            # Reset de la liste dense tous les 10 itérations pour favoriser la diversité
            if iteration % 10 == 0 and iteration > 0:
                # Garder seulement les 3 meilleures solutions
                self.dense_list = self.dense_list[:3] if len(self.dense_list) > 3 else self.dense_list

            history.append((iteration, self.best_solution.makespan))
        
        # Résultat final
        return self.best_solution.schedule, self.best_solution.makespan , history

    def get_detailed_schedule(self, solution):
        """Retourne le planning détaillé d'une solution."""
        job_end = [0] * self.problem.get_num_jobs()
        machine_end = {}
        schedule = []

        for job_id, step in solution.permutation:
            machine, duration = self.problem.jobs[job_id][step]
            start = max(job_end[job_id], machine_end.get(machine, 0))
            end = start + duration
            job_end[job_id] = end
            machine_end[machine] = end

            schedule.append((job_id, step, machine, start, end))

        return schedule, max(job_end)  # retourne le planning et le makespan