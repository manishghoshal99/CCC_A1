import argparse
import heapq
import operator
import time
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from util import (
    read_data_line_by_line, preprocess_data, dump_time, read_n_lines,
    processing_data, merge_list, dump_happiest_hours, dump_saddest_hours,
    dump_happiest_users, dump_saddest_users, dump_num_processor
)

def main(mastodon_data_path):
    program_start = time.time()
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Display number of processors (only on root)
    if comm_rank == 0:
        dump_num_processor(comm_size)
    
    # Dictionaries for accumulating sentiment data
    hour_sentiment_dict = defaultdict(int)
    user_sentiment_dict = {}
    
    # --- Parallel File Reading and Processing ---
    n_lines = None
    if comm_rank == 0:
        n_lines = read_n_lines(mastodon_data_path)
    n_lines = comm.bcast(n_lines, root=0)
    
    lines_per_core = n_lines // comm_size
    line_to_start = lines_per_core * comm_rank
    # Last processor picks up any remainder
    line_to_end = line_to_start + lines_per_core if comm_rank < comm_size - 1 else n_lines
    
    process_start = time.time()
    for line_number, line in enumerate(read_data_line_by_line(mastodon_data_path)):
        if line_number < line_to_start:
            continue
        if line_number >= line_to_end:
            break
        preprocessed_line = preprocess_data(line)
        if preprocessed_line:
            processing_data(preprocessed_line, hour_sentiment_dict, user_sentiment_dict)
    process_time = time.time() - process_start
    dump_time(comm_rank, "data processing", process_time)
    
    # --- Parallel Top-N Calculation ---
    top_n = 5
    calculate_top_n_start = time.time()
    
    if comm_size > 1:
        # Reduce hour sentiment dictionaries across processors
        reduced_hour_sentiment = comm.reduce(hour_sentiment_dict, root=0, op=operator.add)
        # Gather user sentiment dictionaries to the root
        all_user_data = comm.gather(user_sentiment_dict, root=0)
        reduced_user_sentiment = {}
        if comm_rank == 0:
            # Merge the user sentiment dictionaries
            for proc_data in all_user_data:
                for user_id, (username, score) in proc_data.items():
                    if user_id in reduced_user_sentiment:
                        _, existing_score = reduced_user_sentiment[user_id]
                        reduced_user_sentiment[user_id] = (username, existing_score + score)
                    else:
                        reduced_user_sentiment[user_id] = (username, score)
            # Prepare lists for top-n calculation
            hour_items = list(reduced_hour_sentiment.items())
            user_items = list(reduced_user_sentiment.items())
            split_hour_np_array = np.array_split(hour_items, comm_size)
            split_user_np_array = np.array_split(user_items, comm_size)
        else:
            split_hour_np_array = None
            split_user_np_array = None
        
        # Scatter the lists among processors
        local_hour_data = comm.scatter(split_hour_np_array, root=0)
        local_user_data = comm.scatter(split_user_np_array, root=0)
        
        # Each processor finds its local top-n results
        local_happiest_hours = heapq.nlargest(top_n, local_hour_data, key=lambda x: x[1])
        local_saddest_hours = heapq.nsmallest(top_n, local_hour_data, key=lambda x: x[1])
        local_happiest_users = heapq.nlargest(top_n, local_user_data, key=lambda x: x[1][1])
        local_saddest_users = heapq.nsmallest(top_n, local_user_data, key=lambda x: x[1][1])
        
        # Reduce (merge) local top-n lists into global results at root
        reduced_happiest_hours = comm.reduce(local_happiest_hours, root=0, op=merge_list)
        reduced_saddest_hours = comm.reduce(local_saddest_hours, root=0, op=lambda x, y: merge_list(y, x))
        reduced_happiest_users = comm.reduce(local_happiest_users, root=0, op=merge_list)
        reduced_saddest_users = comm.reduce(local_saddest_users, root=0, op=lambda x, y: merge_list(y, x))
    else:
        # Single processor top-n calculation
        reduced_hour_sentiment = hour_sentiment_dict
        reduced_user_sentiment = user_sentiment_dict
        reduced_happiest_hours = heapq.nlargest(top_n, reduced_hour_sentiment.items(), key=lambda x: x[1])
        reduced_saddest_hours = heapq.nsmallest(top_n, reduced_hour_sentiment.items(), key=lambda x: x[1])
        reduced_happiest_users = heapq.nlargest(top_n, reduced_user_sentiment.items(), key=lambda x: x[1][1])
        reduced_saddest_users = heapq.nsmallest(top_n, reduced_user_sentiment.items(), key=lambda x: x[1][1])
    
    calculate_top_n_time = time.time() - calculate_top_n_start
    dump_time(comm_rank, "calculating top-n", calculate_top_n_time)
    
    # --- Output Results on Root ---
    if comm_rank == 0:
        dump_happiest_hours(reduced_happiest_hours)
        dump_saddest_hours(reduced_saddest_hours)
        dump_happiest_users(reduced_happiest_users)
        dump_saddest_users(reduced_saddest_users)
        total_time = time.time() - program_start
        print(f"Program runs in {total_time:.2f} seconds")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mastodon Data Analytics using MPI")
    parser.add_argument("-data", type=str, required=True, help="Path to Mastodon data file (ndjson)")
    args = parser.parse_args()
    main(args.data)
