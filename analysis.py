import re
import matplotlib.pyplot as plt

OUTPUT_DIR = "./"
DATA_DIR = "./"
DATA_FILE_TYPE = ".txt"
IMAGE_TYPE = ".png"

def read_txt(file_path: str):
    run_time = None
    process_time = {}
    calculate_top_n_time = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Program runs"):
                run_time = float(re.findall(r"[0-9.]+", line)[0])
            elif "processing data" in line:
                parts = re.findall(r"[0-9.]+", line)
                if len(parts) >= 2:
                    proc_rank = int(parts[0])
                    process_time[proc_rank] = float(parts[1])
            elif "calculating top-n" in line:
                parts = re.findall(r"[0-9.]+", line)
                if len(parts) >= 2:
                    proc_rank = int(parts[0])
                    calculate_top_n_time[proc_rank] = float(parts[1])
    return run_time, process_time, calculate_top_n_time

def draw_bar_chart(file_name: str):
    labels = []
    times = []
    
    # 1 node, 1 core
    file1 = DATA_DIR + "1node1core" + DATA_FILE_TYPE
    rt, _, _ = read_txt(file1)
    labels.append("1 Node 1 Core")
    times.append(rt)
    
    # 1 node, 8 cores
    file2 = DATA_DIR + "1node8core" + DATA_FILE_TYPE
    rt, _, _ = read_txt(file2)
    labels.append("1 Node 8 Cores")
    times.append(rt)
    
    # 2 nodes, 8 cores
    file3 = DATA_DIR + "2node8core" + DATA_FILE_TYPE
    rt, _, _ = read_txt(file3)
    labels.append("2 Nodes 8 Cores")
    times.append(rt)
    
    plt.figure(figsize=(10,6))
    plt.bar(labels, times, color="skyblue")
    plt.xlabel("Resources")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Mastodon Data Analytics Execution Time")
    for i, v in enumerate(times):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(OUTPUT_DIR + file_name + IMAGE_TYPE)
    plt.show()

if __name__ == "__main__":
    draw_bar_chart("MastodonPerformanceComparison")
