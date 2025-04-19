
class JSSPProblem:
    def __init__(self, jobs):
        self.jobs = jobs

    def get_num_jobs(self):
        return len(self.jobs)

    def get_num_operations(self, job_id):
        return len(self.jobs[job_id])