import datetime
import os
import io
import json
import mmap
from MastodonData import MastodonData

# A long separator for clearer printing output
SEPARATOR = "=" * 50

def preprocess_data(data: str):
    """
    Remove extra whitespace and validate the data line.
    Returns None if the line is invalid.
    """
    if data and data.strip():
        return data.strip()
    return None

def processing_data(preprocessed_line: str, hour_sentiment_dict: dict, user_sentiment_dict: dict):
    """
    Process a preprocessed JSON line into sentiment by hour and per user.
    Updates the dictionaries in place.
    
    Args:
        preprocessed_line: A string containing a valid JSON object
        hour_sentiment_dict: Dictionary to store hour -> sentiment score
        user_sentiment_dict: Dictionary to store user_id -> (username, score)
    """
    try:
        mastodon_data = MastodonData(preprocessed_line)
        
        # Skip entries without required data
        if not mastodon_data.created_at or mastodon_data.sentiment is None:
            return
            
        # Process sentiment per hour
        try:
            created_datetime = datetime.datetime.fromisoformat(
                mastodon_data.created_at.replace('Z', '+00:00')
            )
            # Hour key format: YYYY-MM-DD HH (e.g., 2023-03-15 14)
            hour_key = created_datetime.strftime("%Y-%m-%d %H")
            hour_sentiment_dict[hour_key] += mastodon_data.sentiment
        except Exception:
            # Skip entries with invalid dates
            pass
        
        # Process sentiment per user if user_id exists
        if mastodon_data.user_id:
            if mastodon_data.user_id in user_sentiment_dict:
                username, score = user_sentiment_dict[mastodon_data.user_id]
                user_sentiment_dict[mastodon_data.user_id] = (mastodon_data.username, score + mastodon_data.sentiment)
            else:
                user_sentiment_dict[mastodon_data.user_id] = (mastodon_data.username, mastodon_data.sentiment)
    except Exception as e:
        # Skip entries that can't be processed
        pass

def count_lines(file_path: str):
    """
    Count the number of lines in the file efficiently.
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: Number of lines in the file
    """
    with open(file_path, 'rb') as f:
        # Memory map the file for efficient line counting
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            line_count = 0
            while mm.readline():
                line_count += 1
            return line_count
        except ValueError:
            # Fallback for files that can't be memory-mapped
            return sum(1 for _ in f)

def read_data_chunk(file_path: str, start_line: int, end_line: int):
    """
    Read a chunk of lines from a file, skipping to start_line and
    stopping at end_line.
    
    Args:
        file_path: Path to the file
        start_line: Line number to start reading (0-indexed)
        end_line: Line number to stop reading (exclusive)
        
    Yields:
        str: Each line in the specified range
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        # Skip lines before start_line
        for _ in range(start_line):
            next(file, None)
        
        # Read lines from start_line to end_line
        for i, line in enumerate(file):
            if i >= end_line - start_line:
                break
            yield line

def format_hour_range(hour_str: str):
    """
    Format an hour string into a human-readable range.
    
    Args:
        hour_str: Hour string in format "YYYY-MM-DD HH"
    
    Returns:
        str: Formatted hour range string
    """
    try:
        dt = datetime.datetime.strptime(hour_str, "%Y-%m-%d %H")
        end_hour = dt.hour + 1
        return f"{dt.strftime('%Y-%m-%d %H:00')} to {dt.strftime('%Y-%m-%d')} {end_hour:02d}:00"
    except Exception:
        return hour_str

def dump_happiest_hours(happy_hours: list, output_dir=None):
    """
    Print the top happiest hours in a human-friendly format.
    
    Args:
        happy_hours: List of (hour, score) tuples
        output_dir: Directory to save output file (optional)
    """
    print(SEPARATOR)
    print("Top Happiest Hours")
    print(SEPARATOR)
    
    output = []
    for i, (hour, score) in enumerate(happy_hours, start=1):
        formatted_hour = format_hour_range(hour)
        line = f"{i}. {formatted_hour} with sentiment +{score}"
        print(line)
        output.append(line)
    print()
    
    # Save to file if output_dir is specified
    if output_dir:
        with open(os.path.join(output_dir, "happiest_hours.txt"), "w") as f:
            f.write("Top Happiest Hours\n")
            f.write(SEPARATOR + "\n")
            for line in output:
                f.write(line + "\n")

def dump_saddest_hours(sad_hours: list, output_dir=None):
    """
    Print the top saddest hours.
    
    Args:
        sad_hours: List of (hour, score) tuples
        output_dir: Directory to save output file (optional)
    """
    print(SEPARATOR)
    print("Top Saddest Hours")
    print(SEPARATOR)
    
    output = []
    for i, (hour, score) in enumerate(sad_hours, start=1):
        formatted_hour = format_hour_range(hour)
        line = f"{i}. {formatted_hour} with sentiment {score}"
        print(line)
        output.append(line)
    print()
    
    # Save to file if output_dir is specified
    if output_dir:
        with open(os.path.join(output_dir, "saddest_hours.txt"), "w") as f:
            f.write("Top Saddest Hours\n")
            f.write(SEPARATOR + "\n")
            for line in output:
                f.write(line + "\n")

def dump_happiest_users(happy_users: list, output_dir=None):
    """
    Print the top happiest users.
    
    Args:
        happy_users: List of (user_id, (username, score)) tuples
        output_dir: Directory to save output file (optional)
    """
    print(SEPARATOR)
    print("Top Happiest Users")
    print(SEPARATOR)
    
    output = []
    for i, (user_id, (username, score)) in enumerate(happy_users, start=1):
        line = f"{i}. {username} (ID: {user_id}) with total sentiment +{score}"
        print(line)
        output.append(line)
    print()
    
    # Save to file if output_dir is specified
    if output_dir:
        with open(os.path.join(output_dir, "happiest_users.txt"), "w") as f:
            f.write("Top Happiest Users\n")
            f.write(SEPARATOR + "\n")
            for line in output:
                f.write(line + "\n")

def dump_saddest_users(sad_users: list, output_dir=None):
    """
    Print the top saddest users.
    
    Args:
        sad_users: List of (user_id, (username, score)) tuples
        output_dir: Directory to save output file (optional)
    """
    print(SEPARATOR)
    print("Top Saddest Users")
    print(SEPARATOR)
    
    output = []
    for i, (user_id, (username, score)) in enumerate(sad_users, start=1):
        line = f"{i}. {username} (ID: {user_id}) with total sentiment {score}"
        print(line)
        output.append(line)
    print()
    
    # Save to file if output_dir is specified
    if output_dir:
        with open(os.path.join(output_dir, "saddest_users.txt"), "w") as f:
            f.write("Top Saddest Users\n")
            f.write(SEPARATOR + "\n")
            for line in output:
                f.write(line + "\n")

def dump_time(comm_rank, title, time_period):
    """
    Print the time taken by a processor for a specific task.
    
    Args:
        comm_rank: MPI rank of the processor
        title: Description of the task
        time_period: Time taken in seconds
    """
    print(SEPARATOR)
    print(f"Processor #{comm_rank} completed {title} in {time_period:.2f} seconds")
    print(SEPARATOR)

def merge_list(x: list, y: list, n=5):
    """
    Merge two sorted lists and return the top n items.
    
    Args:
        x: First sorted list
        y: Second sorted list
        n: Number of items to return
        
    Returns:
        list: Top n items from the merged lists
    """
    # Use heapq.merge for efficient merging (assumes both lists are sorted in the same order)
    merged = list(heapq.merge(x, y, key=lambda item: -item[1]))[:n]
    return merged

def dump_num_processor(comm_size):
    """
    Print the number of processors used.
    
    Args:
        comm_size: Number of MPI processes
    """
    print(SEPARATOR * 2)
    print(f"Running with {comm_size} processors")
    print(SEPARATOR * 2)
    print()