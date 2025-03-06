# 定义 FJSP 问题
class FJSP:
    def __init__(self, jobs, machines):
        self.jobs = jobs  # 作业列表，每个作业是一个工序列表，每个工序是一个元组（可选机器列表，加工时间列表）
        self.machines = machines  # 机器列表

    def makespan(self, schedule):
        # 计算给定调度方案的 makespan
        machine_times = {machine: 0 for machine in self.machines}  # 每台机器的完成时间
        for job in schedule:
            for operation in job:
                machine, time = operation
                machine_times[machine] += time
        return max(machine_times.values())  # makespan 是最后一台机器完成的时间


# 解析dataset的数据结构
def parse_fjsp_data(data):
    jobs = []
    for job_data in data['jobs']:
        operations = []
        for operation_data in job_data:
            machines = [op['machine'] for op in operation_data]
            times = [op['process_time'] for op in operation_data]
            operations.append((machines, times))
        jobs.append(operations)
    machines = list(range(1, data['machine_num'] + 1))
    return jobs, machines