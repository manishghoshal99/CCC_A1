import datetime
from MastodonData import MastodonData

# A long separator for clearer printing output
SEPARATOR = "=" * 50

def preprocess_data(data: str):
    """
    Remove extra whitespace and validate the data line.
    """
    if data and data.strip():
        return data.strip()
    print("Invalid line:", data)
    return None

def processing_data(preprocessed_line: str, hour_sentiment_dict: dict, user_sentiment_dict: dict):
    """
    Process a preprocessed JSON line into sentiment by hour and per user.
    """
    try:
        mastodon_data = MastodonData(preprocessed_line)
        
        # Process sentiment per hour if valid created_at and sentiment present.
        if mastodon_data.created_at and mastodon_data.sentiment is not None:
            try:
                created_datetime = datetime.datetime.fromisoformat(
                    mastodon_data.created_at.replace('Z', '+00:00')
                )
                # Hour key format: YYYY-MM-DD HH (e.g., 2023-03-15 14)
                hour_key = created_datetime.strftime("%Y-%m-%d %H")
            except Exception as e:
                print("Error parsing date:", mastodon_data.created_at)
                return
            
            hour_sentiment_dict[hour_key] = hour_sentiment_dict.get(hour_key, 0) + mastodon_data.sentiment
        
        # Process sentiment per user if user_id exists.
        if mastodon_data.user_id and mastodon_data.sentiment is not None:
            if mastodon_data.user_id in user_sentiment_dict:
                username, score = user_sentiment_dict[mastodon_data.user_id]
                user_sentiment_dict[mastodon_data.user_id] = (mastodon_data.username, score + mastodon_data.sentiment)
            else:
                user_sentiment_dict[mastodon_data.user_id] = (mastodon_data.username, mastodon_data.sentiment)
    except Exception as e:
        print(f"Error processing data: {e}")

def read_data_line_by_line(file_path: str):
    """
    Lazy generator to read a file line by line.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line

def read_n_lines(file_path: str):
    """
    Count the number of lines in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)

def dump_happiest_hours(happy_hours: list):
    """
    Print the top happiest hours in a human-friendly format.
    """
    print(SEPARATOR)
    print("Top Happiest Hours")
    print(SEPARATOR)
    for i, (hour, score) in enumerate(happy_hours, start=1):
        try:
            dt = datetime.datetime.strptime(hour, "%Y-%m-%d %H")
            end_hour = dt.hour + 1
            print(f"{i}. {dt.strftime('%Y-%m-%d %H:00')} to {dt.strftime('%Y-%m-%d')} {end_hour:02d}:00 with sentiment +{score}")
        except Exception:
            print(f"{i}. {hour} with sentiment +{score}")
    print()

def dump_saddest_hours(sad_hours: list):
    """
    Print the top saddest hours.
    """
    print(SEPARATOR)
    print("Top Saddest Hours")
    print(SEPARATOR)
    for i, (hour, score) in enumerate(sad_hours, start=1):
        try:
            dt = datetime.datetime.strptime(hour, "%Y-%m-%d %H")
            end_hour = dt.hour + 1
            print(f"{i}. {dt.strftime('%Y-%m-%d %H:00')} to {dt.strftime('%Y-%m-%d')} {end_hour:02d}:00 with sentiment {score}")
        except Exception:
            print(f"{i}. {hour} with sentiment {score}")
    print()

def dump_happiest_users(happy_users: list):
    """
    Print the top happiest users.
    """
    print(SEPARATOR)
    print("Top Happiest Users")
    print(SEPARATOR)
    for i, (user_id, (username, score)) in enumerate(happy_users, start=1):
        print(f"{i}. {username} (ID: {user_id}) with total sentiment +{score}")
    print()

def dump_saddest_users(sad_users: list):
    """
    Print the top saddest users.
    """
    print(SEPARATOR)
    print("Top Saddest Users")
    print(SEPARATOR)
    for i, (user_id, (username, score)) in enumerate(sad_users, start=1):
        print(f"{i}. {username} (ID: {user_id}) with total sentiment {score}")
    print()

def dump_time(comm_rank, title, time_period):
    """
    Print the time taken by a processor for a specific task.
    """
    print(SEPARATOR)
    print(f"Processor #{comm_rank} completed {title} in {time_period:.2f} seconds")
    print(SEPARATOR)

def merge_list(x: list, y: list, n=5):
    """
    Merge two sorted lists and return the top n items.
    """
    merged = []
    while len(merged) < n and (x or y):
        if x and (not y or x[0][1] >= y[0][1]):
            merged.append(x.pop(0))
        elif y:
            merged.append(y.pop(0))
    return merged

def dump_num_processor(comm_size):
    """
    Print the number of processors used.
    """
    print(SEPARATOR * 2)
    print(f"Running with {comm_size} processors")
    print(SEPARATOR * 2)
    print()
