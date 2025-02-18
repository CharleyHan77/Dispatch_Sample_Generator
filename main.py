import copy
import os
import timeit

from app.heuristics import Heuristics
from app.scheduler import Scheduler
from app.customparser import parse
import pandas as pd

def start(path):
    # for i in range(1, 19):
    #     if i < 10:
    #         i = "0" +str(i)
    #     else:
    #         i = str(i)
    #path = f"data/Hurink_Data/Text/edata/{file_path}"
    print(path)
    jobs_list, machines_list, number_max_operations = parse(path)
    number_total_machines = len(machines_list)
    number_total_jobs = len(jobs_list)

    print("Scheduler launched with the following parameters:")
    print('\t', number_total_jobs, "jobs")
    print('\t', number_total_machines, "machine(s)")
    print('\t', "Machine(s) can process", str(number_max_operations), "operation(s) at the same time")
    print("\n")

    total_time = 0
    mapkespan = 0
    for i in range(10):
        time, mapkespan = once_schedule(path, number_total_machines, number_total_jobs, number_max_operations, jobs_list, machines_list)
        total_time += time

    print("运行10次总用时" + str(total_time) + "s")
    print(str(round(total_time/10, 5)))
    print(30 * "-", "MENU", 30 * "-")
    return mapkespan, str(round(total_time/10, 5))


def once_schedule(path, number_total_machines, number_total_jobs, number_max_operations, jobs_list, machines_list):
    # 机器选择SPT
    heuristic = Heuristics.select_first_operation
    temp_jobs_list = copy.deepcopy(jobs_list)
    temp_machines_list = copy.deepcopy(machines_list)

    start = timeit.default_timer()
    s = Scheduler(temp_machines_list, number_max_operations, temp_jobs_list)
    makespan = s.run(heuristic)
    stop = timeit.default_timer()
    return stop - start, makespan

# python main.py
if __name__ == "__main__":
    directory = "data/Hurink_Data/Text/vdata"
    file_path_list, makespan_list, time_list = list(), list(), list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 将文件名作为字符串传入脚本运行
            #file_path = "data/Hurink_Data/Text/rdata\orb10.fjs"
            mapkespan, time = start(file_path)
            file_display = "/Text/vdata" + f"/{file}"
            file_path_list.append(file_display)
            makespan_list.append(mapkespan)
            time_list.append(time)
    # 直接导出excel
    # file_path makespan time
    print(file_path_list)
    print(makespan_list)
    print(time_list)
    data = {
        "file_path": file_path_list,
        "makespan": makespan_list,
        "time": time_list
    }
    print(data)
    df = pd.DataFrame(data)
    columns_to_export = ['file_path', 'makespan', "time"]
    df_exported = df[columns_to_export]
    output_file = 'Hurink_Data_vdata_MS_SPT_0218.xlsx'
    df.to_excel(output_file, index=False)
