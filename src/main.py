import argparse
import heapq
import operator
import time
import os
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from util import (
    read_data_chunk, preprocess_data, dump_time, count_lines,
    processing_data, merge_list, dump_happiest_hours, dump_saddest_hours,
    dump_happiest_users, dump_saddest_users, dump_num_processor
)

def main(mastodon_data_path, output_dir=None):
    """
    Main function to analyze Mastodon data in parallel.
    
    Args:
        mastodon_data_path (str): Path to the Mastodon NDJSON file
        output_dir (str, optional): Directory to save output files
    """
    program_start = time.time()
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Create output directory if specified
    if output_dir and comm_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Display number of processors (only on root)
    if comm_rank == 0:
        dump_num_processor(comm_size)
    
    # Dictionaries for accumulating sentiment data
    hour_sentiment_dict = defaultdict(int)
    user_sentiment_dict = {}
    
    # --- Parallel File Reading and Processing ---
    # Count total lines only on root process to avoid duplicate I/O
    n_lines = None
    if comm_rank == 0:
        n_lines = count_lines(mastodon_data_path)
    # Broadcast the line count to all processes
    n_lines = comm.bcast(n_lines, root=0)
    
    # Calculate chunk size for each process
    chunk_size = n_lines // comm_size
    start_line = comm_rank * chunk_size
    # Last processor takes any remainder lines
    end_line = start_line + chunk_size if comm_rank < comm_size - 1 else n_lines
    
    # Process data in chunks for better memory efficiency
    process_start = time.time()
    lines_processed = 0
    max_chunk = 10000  # Process in smaller chunks to manage memory
    
    for chunk_start in range(start_line, end_line, max_chunk):
        chunk_end = min(chunk_start + max_chunk, end_line)
        
        for line in read_data_chunk(mastodon_data_path, chunk_start, chunk_end):
            preprocessed_line = preprocess_data(line)
            if preprocessed_line:
                processing_data(preprocessed_line, hour_sentiment_dict, user_sentiment_dict)
                lines_processed += 1
        
        # Optional: Progress reporting for long-running jobs
        if comm_rank == 0 and end_line - start_line > 100000:
            progress = (chunk_end - start_line) / (end_line - start_line) * 100
            print(f"Progress: {progress:.1f}% ({chunk_end}/{end_line} lines)")
    
    process_time = time.time() - process_start
    dump_time(comm_rank, "data processing", process_time)
    
    # --- Parallel Top-N Calculation ---
    top_n = 5
    calculate_top_n_start = time.time()
    
    if comm_size > 1:
        # Reduce hour sentiment dictionaries across processors using MPI.Op.Create
        all_hour_data = comm.gather(hour_sentiment_dict, root=0)
        all_user_data = comm.gather(user_sentiment_dict, root=0)
        
        if comm_rank == 0:
            # Merge hour sentiment data
            reduced_hour_sentiment = defaultdict(int)
            for proc_data in all_hour_data:
                for hour, score in proc_data.items():
                    reduced_hour_sentiment[hour] += score
            
            # Merge user sentiment data
            reduced_user_sentiment = {}
            for proc_data in all_user_data:
                for user_id, (username, score) in proc_data.items():
                    if user_id in reduced_user_sentiment:
                        _, existing_score = reduced_user_sentiment[user_id]
                        reduced_user_sentiment[user_id] = (username, existing_score + score)
                    else:
                        reduced_user_sentiment[user_id] = (username, score)
        else:
            reduced_hour_sentiment = None
            reduced_user_sentiment = None
        
        # Since finding top-n is quick for this dataset size, just do it on root
        if comm_rank == 0:
            reduced_happiest_hours = heapq.nlargest(top_n, reduced_hour_sentiment.items(), key=lambda x: x[1])
            reduced_saddest_hours = heapq.nsmallest(top_n, reduced_hour_sentiment.items(), key=lambda x: x[1])
            reduced_happiest_users = heapq.nlargest(top_n, reduced_user_sentiment.items(), key=lambda x: x[1][1])
            reduced_saddest_users = heapq.nsmallest(top_n, reduced_user_sentiment.items(), key=lambda x: x[1][1])
        else:
            reduced_happiest_hours = None
            reduced_saddest_hours = None
            reduced_happiest_users = None
            reduced_saddest_users = None
    else:
        # Single processor calculation
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
        dump_happiest_hours(reduced_happiest_hours, output_dir=output_dir)
        dump_saddest_hours(reduced_saddest_hours, output_dir=output_dir)
        dump_happiest_users(reduced_happiest_users, output_dir=output_dir)
        dump_saddest_users(reduced_saddest_users, output_dir=output_dir)
        total_time = time.time() - program_start
        print(f"Program runs in {total_time:.2f} seconds")
        
        # Save runtime to output file if directory specified
        if output_dir:
            with open(os.path.join(output_dir, "runtime.txt"), "w") as f:
                f.write(f"Program runs in {total_time:.2f} seconds\n")
                f.write(f"Data processing time: {process_time:.2f} seconds\n")
                f.write(f"Top-N calculation time: {calculate_top_n_time:.2f} seconds\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mastodon Data Analytics using MPI")
    parser.add_argument("-data", type=str, required=True, help="Path to Mastodon data file (ndjson)")
    parser.add_argument("-output", type=str, help="Directory to save output files")
    args = parser.parse_args()
    main(args.data, args.output)