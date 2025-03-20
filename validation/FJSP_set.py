

# 定义 FJSP 问题
class FJSP:
    def __init__(self):
        self.machines = []  # 机器列表
        self.jobs = []  # 作业列表，每个作业是一个工序列表，每个工序是一个元组（可选机器列表，加工时间列表）

    def makespan(self, schedule):
        # 计算给定调度方案的 makespan
        machine_times = {machine: 0 for machine in self.machines}  # 每台机器的完成时间
        for job in schedule:
            for operation in job:
                machine, time = operation
                machine_times[machine] += time
        return max(machine_times.values())  # makespan 是最后一台机器完成的时间

    def parse_fjsp_data(self, data):
        """
        解析输入的数据结构，将其转换为类的属性。
        :param data: 输入的数据结构，包含机器数量和作业信息。
        """
        # 解析机器数量
        self.machines = list(range(1, data['machine_num'] + 1))

        # 解析作业信息
        self.jobs = []
        for job in data.get("jobs", []):
            job_operations = []
            for operation in job:
                # 每个工序是一个元组（可选机器列表，加工时间列表）
                machines = [op["machine"] for op in operation]
                process_times = [op["process_time"] for op in operation]
                job_operations.append((machines, process_times))
            self.jobs.append(job_operations)


def calculate_makespan(sequence):
    """
    计算给定序列的 makespan。
    :param sequence: 作业序列，每个作业是一个工序列表，每个工序是一个元组 (machine, process_time)。
    :return: makespan 值。
    """
    # 初始化每台机器的可用时间
    machine_available_time = {}
    # 初始化每个作业的完成时间
    job_completion_time = [0] * len(sequence)

    # 遍历每个作业
    for job_idx, job in enumerate(sequence):
        # 遍历作业的每个工序
        for operation in job:
            machine, process_time = operation
            # 获取机器的可用时间
            machine_time = machine_available_time.get(machine, 0)
            # 作业的当前完成时间
            job_time = job_completion_time[job_idx]
            # 工序的开始时间是机器可用时间和作业完成时间的最大值
            start_time = max(machine_time, job_time)
            # 更新机器的可用时间
            machine_available_time[machine] = start_time + process_time
            # 更新作业的完成时间
            job_completion_time[job_idx] = start_time + process_time

    # makespan 是所有作业完成时间的最大值
    makespan = max(job_completion_time)
    return makespan



#
# # 解析dataset的数据结构
# def parse_fjsp_data(data):
#     jobs = []
#     for job_data in data['jobs']:
#         operations = []
#         for operation_data in job_data:
#             machines = [op['machine'] for op in operation_data]
#             times = [op['process_time'] for op in operation_data]
#             operations.append((machines, times))
#         jobs.append(operations)
#     machines = list(range(1, data['machine_num'] + 1))
#     return jobs, machines