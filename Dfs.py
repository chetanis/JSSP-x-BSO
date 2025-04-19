class DFSSolver:
    def __init__(self, problem):
        self.problem = problem
        self.num_jobs = self.problem.get_num_jobs()
        self.num_machines = max(machine for job in self.problem.jobs for machine, _ in job) + 1
        self.best_makespan = float('inf')
        self.best_schedule = []

    def is_complete(self, job_steps):
        return all(step >= self.problem.get_num_operations(job_id) for job_id, step in enumerate(job_steps))
    
    


    def dfs(self, schedule, job_steps, job_end, machine_end):
        # print(f"iteration: {len(schedule)}")
        if self.is_complete(job_steps):
            # print(f"Schedule: {schedule}")
            makespan = max(job_end)
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = schedule.copy()
            return
        if(max(job_end)>self.best_makespan):
            return
        for job_id in range(self.num_jobs):
            step = job_steps[job_id]
            if step >= self.problem.get_num_operations(job_id):
                continue

            machine, duration = self.problem.jobs[job_id][step]
            start_time = max(job_end[job_id], machine_end[machine])
            end_time = start_time + duration

            job_steps_copy = job_steps.copy()
            job_steps_copy[job_id] += 1
            job_end_copy = job_end.copy()
            job_end_copy[job_id] = end_time
            machine_end_copy = machine_end.copy()
            machine_end_copy[machine] = end_time

            schedule.append((job_id, step, machine, start_time, end_time))
            self.dfs(schedule, job_steps_copy, job_end_copy, machine_end_copy)
            schedule.pop()

    def solve(self):
        self.dfs(
            schedule=[],
            job_steps=[0] * self.num_jobs,
            job_end=[0] * self.num_jobs,
            machine_end=[0] * self.num_machines
        )
        return self.best_makespan, self.best_schedule