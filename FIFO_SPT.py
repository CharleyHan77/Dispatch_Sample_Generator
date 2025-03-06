
# 解析数据集
def parse_fjsp_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 第一行：工件数量、设备数量、平均每道工序可用设备数量
    first_line = lines[0].strip().split('\t')
    num_jobs = int(first_line[0])  # 工件数量
    num_machines = int(first_line[1])  # 设备数量
    avg_machines_per_op = int(first_line[2])  # 平均每道工序可用设备数量

    # 初始化数据结构
    jobs = []
    for line in lines[1:]:
        line = line.strip().split('\t')
        index = 0
        num_operations = int(line[index])  # 工序数量
        index += 1
        operations = []
        for _ in range(num_operations):
            num_machines_op = int(line[index])  # 当前工序可用设备数量
            index += 1
            machines = []
            for _ in range(num_machines_op):
                machine = int(line[index])  # 设备编号
                index += 1
                processing_time = int(line[index])  # 加工时间
                index += 1
                machines.append((machine, processing_time))
            operations.append(machines)
        jobs.append(operations)

    return num_jobs, num_machines, avg_machines_per_op, jobs



# FIFO调度规则（工序排序）
def fifo_schedule(jobs):
    # 将所有工序按作业顺序排列
    operations = []
    for job_id, job in enumerate(jobs):
        for op_id, op in enumerate(job):
            operations.append((job_id, op_id, op))  # (作业ID, 工序ID, 可选机器列表)
    return operations


# SPT调度规则（机器选择）
def spt_schedule(operations, num_machines):
    # 初始化机器时间表
    machine_times = [0] * num_machines
    # 初始化作业的当前工序索引
    job_op_indices = [0] * len(operations)

    # 按FIFO顺序处理每个工序
    for job_id, op_id, op in operations:
        # 选择加工时间最短的可用机器
        min_time = float('inf')
        selected_machine = -1
        for machine, processing_time in op:
            if machine_times[machine] < min_time:
                min_time = machine_times[machine]
                selected_machine = machine

        # 更新机器的完成时间
        start_time = machine_times[selected_machine]
        end_time = start_time + processing_time
        machine_times[selected_machine] = end_time

    # 最大完工时间
    makespan = max(machine_times)
    return makespan


# 主函数
def main():
    # 数据集路径
    file_path = "dataset/Brandimarte/Text/Mk01.fjs"

    num_jobs, num_machines, avg_machines_per_op, jobs = parse_fjsp_data(file_path)

    # 输出解析结果
    print(f"工件数量: {num_jobs}")
    print(f"设备数量: {num_machines}")
    print(f"平均每道工序可用设备数量: {avg_machines_per_op}")
    for job_id, job in enumerate(jobs, start=1):
        print(f"工件 {job_id}:")
        for op_id, op in enumerate(job, start=1):
            print(f"  工序 {op_id}: 可用设备 {op}")

    # 先使用FIFO调度规则生成工序顺序
    operations = fifo_schedule(jobs)

    # 使用SPT调度规则分配机器并计算最大完工时间
    makespan = spt_schedule(operations, num_machines)

    # 输出结果
    print(f"最大完工时间（Makespan）: {makespan}")


if __name__ == "__main__":
    main()