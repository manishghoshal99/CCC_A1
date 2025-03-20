import datetime
import heapq
import numpy as np
import json
from collections import defaultdict, Counter
from itertools import islice
from mpi4py import MPI
from MastodonData import MastodonData

class MastodonAnalyzer:
    """
    Advanced analysis class for Mastodon data.
    Provides methods for sentiment analysis, temporal patterns, and user behavior metrics.
    Optimized for parallel processing with MPI.
    """
    
    def __init__(self, comm=None):
        """
        Initialize the analyzer with optional MPI communicator.
        
        Args:
            comm: MPI communicator (default: None for sequential processing)
        """
        self.comm = comm
        self.comm_rank = 0
        self.comm_size = 1
        
        if self.comm:
            self.comm_rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()
            
        # Initialize data structures for analysis
        self.hour_sentiment = defaultdict(float)
        self.day_sentiment = defaultdict(float)
        self.user_sentiment = {}
        self.language_counts = Counter()
        self.hourly_post_counts = defaultdict(int)
        self.interaction_counts = defaultdict(int)
        self.sentiment_values = []
        
    def process_line(self, line):
        """
        Process a single line of Mastodon data.
        Updates all internal data structures for analysis.
        
        Args:
            line: String containing JSON data
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Parse and validate data
            if not line or not line.strip():
                return False
                
            # Create MastodonData object
            mastodon_data = MastodonData(line)
            
            # Skip entries without required fields
            if not mastodon_data.created_at or mastodon_data.sentiment is None:
                return False
                
            # Process datetime
            try:
                created_datetime = datetime.datetime.fromisoformat(
                    mastodon_data.created_at.replace('Z', '+00:00')
                )
                
                # Add to hour sentiment
                hour_key = created_datetime.strftime("%Y-%m-%d %H")
                self.hour_sentiment[hour_key] += mastodon_data.sentiment
                
                # Add to day sentiment
                day_key = created_datetime.strftime("%Y-%m-%d")
                self.day_sentiment[day_key] += mastodon_data.sentiment
                
                # Count posts per hour
                self.hourly_post_counts[hour_key] += 1
                
            except Exception:
                # Skip entries with invalid dates
                pass
                
            # Process user sentiment
            if mastodon_data.user_id:
                if mastodon_data.user_id in self.user_sentiment:
                    username, score, count = self.user_sentiment[mastodon_data.user_id]
                    self.user_sentiment[mastodon_data.user_id] = (
                        mastodon_data.username, 
                        score + mastodon_data.sentiment,
                        count + 1
                    )
                else:
                    self.user_sentiment[mastodon_data.user_id] = (
                        mastodon_data.username, 
                        mastodon_data.sentiment,
                        1
                    )
            
            # Additional data extraction from raw JSON
            try:
                # Parse original JSON for additional fields
                json_data = json.loads(line)
                
                # Extract language information
                language = json_data.get("language")
                if language:
                    self.language_counts[language] += 1
                    
                # Extract interaction data
                if json_data.get("in_reply_to_id"):
                    self.interaction_counts["replies"] += 1
                if json_data.get("reblog"):
                    self.interaction_counts["reblogs"] += 1
                if json_data.get("favourites_count"):
                    self.interaction_counts["favorites"] += json_data.get("favourites_count", 0)
                    
            except Exception:
                # Continue even if additional data extraction fails
                pass
                
            # Track sentiment values for distribution analysis
            self.sentiment_values.append(mastodon_data.sentiment)
            
            return True
            
        except Exception as e:
            # Skip problematic entries
            return False
            
    def merge_results(self):
        """
        Merge analysis results from all MPI processes.
        Must be called after all data has been processed.
        
        Returns:
            dict: Merged analysis results
        """
        if not self.comm or self.comm_size == 1:
            # Sequential processing - no merging needed
            return self._get_analysis_results()
            
        # Gather data from all processes
        all_hour_sentiment = self.comm.gather(self.hour_sentiment, root=0)
        all_day_sentiment = self.comm.gather(self.day_sentiment, root=0)
        all_user_sentiment = self.comm.gather(self.user_sentiment, root=0)
        all_language_counts = self.comm.gather(self.language_counts, root=0)
        all_hourly_post_counts = self.comm.gather(self.hourly_post_counts, root=0)
        all_interaction_counts = self.comm.gather(self.interaction_counts, root=0)
        all_sentiment_values = self.comm.gather(self.sentiment_values, root=0)
        
        # Process on root only
        if self.comm_rank == 0:
            # Merge hour sentiment
            merged_hour_sentiment = defaultdict(float)
            for proc_data in all_hour_sentiment:
                for hour, score in proc_data.items():
                    merged_hour_sentiment[hour] += score
            
            # Merge day sentiment
            merged_day_sentiment = defaultdict(float)
            for proc_data in all_day_sentiment:
                for day, score in proc_data.items():
                    merged_day_sentiment[day] += score
            
            # Merge user sentiment
            merged_user_sentiment = {}
            for proc_data in all_user_sentiment:
                for user_id, (username, score, count) in proc_data.items():
                    if user_id in merged_user_sentiment:
                        _, existing_score, existing_count = merged_user_sentiment[user_id]
                        merged_user_sentiment[user_id] = (
                            username, 
                            existing_score + score,
                            existing_count + count
                        )
                    else:
                        merged_user_sentiment[user_id] = (username, score, count)
            
            # Merge language counts
            merged_language_counts = Counter()
            for proc_data in all_language_counts:
                merged_language_counts.update(proc_data)
            
            # Merge hourly post counts
            merged_hourly_post_counts = defaultdict(int)
            for proc_data in all_hourly_post_counts:
                for hour, count in proc_data.items():
                    merged_hourly_post_counts[hour] += count
            
            # Merge interaction counts
            merged_interaction_counts = defaultdict(int)
            for proc_data in all_interaction_counts:
                for interaction_type, count in proc_data.items():
                    merged_interaction_counts[interaction_type] += count
            
            # Merge sentiment values
            merged_sentiment_values = []
            for proc_data in all_sentiment_values:
                merged_sentiment_values.extend(proc_data)
            
            # Update local data with merged results
            self.hour_sentiment = merged_hour_sentiment
            self.day_sentiment = merged_day_sentiment
            self.user_sentiment = merged_user_sentiment
            self.language_counts = merged_language_counts
            self.hourly_post_counts = merged_hourly_post_counts
            self.interaction_counts = merged_interaction_counts
            self.sentiment_values = merged_sentiment_values
            
        # Return analysis results from root
        if self.comm_rank == 0:
            return self._get_analysis_results()
        else:
            return None
    
    def _get_analysis_results(self):
        """
        Extract and calculate final analysis results.
        
        Returns:
            dict: Analysis results
        """
        top_n = 5
        
        # Calculate top hours by sentiment
        happiest_hours = heapq.nlargest(top_n, self.hour_sentiment.items(), key=lambda x: x[1])
        saddest_hours = heapq.nsmallest(top_n, self.hour_sentiment.items(), key=lambda x: x[1])
        
        # Calculate top days by sentiment
        happiest_days = heapq.nlargest(top_n, self.day_sentiment.items(), key=lambda x: x[1])
        saddest_days = heapq.nsmallest(top_n, self.day_sentiment.items(), key=lambda x: x[1])
        
        # Calculate top users by sentiment
        happiest_users = heapq.nlargest(top_n, self.user_sentiment.items(), key=lambda x: x[1][1])
        saddest_users = heapq.nsmallest(top_n, self.user_sentiment.items(), key=lambda x: x[1][1])
        
        # Calculate top users by post count
        most_active_users = heapq.nlargest(top_n, self.user_sentiment.items(), key=lambda x: x[1][2])
        
        # Calculate top languages
        top_languages = self.language_counts.most_common(top_n)
        
        # Calculate busiest hours
        busiest_hours = heapq.nlargest(top_n, self.hourly_post_counts.items(), key=lambda x: x[1])
        
        # Calculate sentiment distribution statistics
        sentiment_stats = {
            "mean": np.mean(self.sentiment_values) if self.sentiment_values else 0,
            "median": np.median(self.sentiment_values) if self.sentiment_values else 0,
            "std": np.std(self.sentiment_values) if self.sentiment_values else 0,
            "min": np.min(self.sentiment_values) if self.sentiment_values else 0,
            "max": np.max(self.sentiment_values) if self.sentiment_values else 0,
            "total_posts": len(self.sentiment_values)
        }
        
        # Calculate interaction statistics
        interaction_stats = dict(self.interaction_counts)
        
        # Calculate average sentiment per user
        avg_sentiment_per_user = {
            user_id: (info[0], info[1] / info[2])
            for user_id, info in self.user_sentiment.items()
            if info[2] > 0
        }
        
        # Find users with most extreme average sentiment
        most_positive_users = heapq.nlargest(
            top_n, 
            avg_sentiment_per_user.items(), 
            key=lambda x: x[1][1]
        )
        most_negative_users = heapq.nsmallest(
            top_n, 
            avg_sentiment_per_user.items(), 
            key=lambda x: x[1][1]
        )
        
        # Compile all results
        results = {
            "happiest_hours": happiest_hours,
            "saddest_hours": saddest_hours,
            "happiest_days": happiest_days,
            "saddest_days": saddest_days,
            "happiest_users": happiest_users,
            "saddest_users": saddest_users,
            "most_active_users": most_active_users,
            "top_languages": top_languages,
            "busiest_hours": busiest_hours,
            "sentiment_stats": sentiment_stats,
            "interaction_stats": interaction_stats,
            "most_positive_users": most_positive_users,
            "most_negative_users": most_negative_users
        }
        
        return results
        
    def analyze_chunk(self, lines):
        """
        Process a chunk of data lines.
        
        Args:
            lines: Iterable of JSON data lines
            
        Returns:
            int: Number of successfully processed lines
        """
        processed = 0
        for line in lines:
            if self.process_line(line):
                processed += 1
        return processed
        
    def get_hourly_sentiment_avg(self):
        """
        Calculate average sentiment per hour.
        
        Returns:
            dict: Hour -> average sentiment
        """
        result = {}
        for hour, total in self.hour_sentiment.items():
            count = self.hourly_post_counts.get(hour, 0)
            if count > 0:
                result[hour] = total / count
        return result
        
    def format_results(self, results):
        """
        Format analysis results for output.
        
        Args:
            results: Dict of analysis results
            
        Returns:
            dict: Formatted results for output
        """
        formatted = {}
        
        # Format happiest hours
        formatted["happiest_hours"] = [
            {
                "hour": self._format_hour_range(hour),
                "sentiment": score
            }
            for hour, score in results.get("happiest_hours", [])
        ]
        
        # Format saddest hours
        formatted["saddest_hours"] = [
            {
                "hour": self._format_hour_range(hour),
                "sentiment": score
            }
            for hour, score in results.get("saddest_hours", [])
        ]
        
        # Format happiest days
        formatted["happiest_days"] = [
            {
                "day": day,
                "sentiment": score
            }
            for day, score in results.get("happiest_days", [])
        ]
        
        # Format saddest days
        formatted["saddest_days"] = [
            {
                "day": day,
                "sentiment": score
            }
            for day, score in results.get("saddest_days", [])
        ]
        
        # Format happiest users
        formatted["happiest_users"] = [
            {
                "id": user_id,
                "username": info[0],
                "sentiment": info[1],
                "posts": results["user_sentiment"].get(user_id, (None, None, 0))[2]
            }
            for user_id, info in results.get("happiest_users", [])
        ]
        
        # Format saddest users
        formatted["saddest_users"] = [
            {
                "id": user_id,
                "username": info[0],
                "sentiment": info[1],
                "posts": results["user_sentiment"].get(user_id, (None, None, 0))[2]
            }
            for user_id, info in results.get("saddest_users", [])
        ]
        
        # Format most active users
        formatted["most_active_users"] = [
            {
                "id": user_id,
                "username": info[0],
                "posts": info[2],
                "sentiment": info[1]
            }
            for user_id, info in results.get("most_active_users", [])
        ]
        
        # Format top languages
        formatted["top_languages"] = [
            {
                "language": lang,
                "posts": count
            }
            for lang, count in results.get("top_languages", [])
        ]
        
        # Format busiest hours
        formatted["busiest_hours"] = [
            {
                "hour": self._format_hour_range(hour),
                "posts": count
            }
            for hour, count in results.get("busiest_hours", [])
        ]
        
        # Include sentiment stats
        formatted["sentiment_stats"] = results.get("sentiment_stats", {})
        
        # Include interaction stats
        formatted["interaction_stats"] = results.get("interaction_stats", {})
        
        # Format most positive users
        formatted["most_positive_users"] = [
            {
                "id": user_id,
                "username": info[0],
                "avg_sentiment": info[1],
                "posts": results["user_sentiment"].get(user_id, (None, None, 0))[2]
            }
            for user_id, info in results.get("most_positive_users", [])
        ]
        
        # Format most negative users
        formatted["most_negative_users"] = [
            {
                "id": user_id,
                "username": info[0],
                "avg_sentiment": info[1],
                "posts": results["user_sentiment"].get(user_id, (None, None, 0))[2]
            }
            for user_id, info in results.get("most_negative_users", [])
        ]
        
        return formatted
        
    def _format_hour_range(self, hour_str):
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


def analyze_mastodon_data(data_path, chunk_size=10000, comm=None):
    """
    Analyze Mastodon data from a file using parallel processing.
    
    Args:
        data_path: Path to the Mastodon data file
        chunk_size: Number of lines to process in each chunk
        comm: MPI communicator (optional)
        
    Returns:
        dict: Analysis results
    """
    # Initialize analyzer
    analyzer = MastodonAnalyzer(comm)
    
    # Get MPI rank and size
    comm_rank = 0
    comm_size = 1
    if comm:
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
    
    # Count lines in file (only on root process)
    n_lines = None
    if comm_rank == 0:
        with open(data_path, 'r') as f:
            n_lines = sum(1 for _ in f)
    
    # Broadcast line count to all processes
    if comm:
        n_lines = comm.bcast(n_lines, root=0)
    
    # Calculate chunk assignments for each process
    lines_per_process = n_lines // comm_size
    start_line = comm_rank * lines_per_process
    end_line = start_line + lines_per_process if comm_rank < comm_size - 1 else n_lines
    
    # Process assigned chunks
    with open(data_path, 'r', encoding='utf-8') as f:
        # Skip to start line
        for _ in range(start_line):
            next(f, None)
        
        # Process chunks
        lines_processed = 0
        current_chunk = []
        
        for i, line in enumerate(f):
            if i >= end_line - start_line:
                break
                
            current_chunk.append(line)
            
            if len(current_chunk) >= chunk_size:
                lines_processed += analyzer.analyze_chunk(current_chunk)
                current_chunk = []
        
        # Process any remaining lines
        if current_chunk:
            lines_processed += analyzer.analyze_chunk(current_chunk)
    
    # Merge results from all processes
    results = analyzer.merge_results()
    
    # Format results (only on root)
    if comm_rank == 0 and results:
        return analyzer.format_results(results)
    else:
        return None


def parallel_analyze_mastodon_data(data_path, output_path=None, chunk_size=10000):
    """
    Analyze Mastodon data using MPI parallelization.
    
    Args:
        data_path: Path to Mastodon data file
        output_path: Path to save results (optional)
        chunk_size: Size of chunks to process at once
        
    Returns:
        dict: Analysis results (on root process only)
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # Start timing
    start_time = MPI.Wtime()
    
    # Run analysis
    results = analyze_mastodon_data(data_path, chunk_size, comm)
    
    # End timing
    end_time = MPI.Wtime()
    elapsed = end_time - start_time
    
    # Gather timing information
    all_times = comm.gather(elapsed, root=0)
    
    # Only root process handles output
    if comm_rank == 0:
        # Calculate parallel efficiency
        if comm_size > 1:
            max_time = max(all_times)
            min_time = min(all_times)
            avg_time = sum(all_times) / len(all_times)
            load_imbalance = max_time / avg_time
            
            print(f"Parallel analysis completed in {max_time:.2f} seconds")
            print(f"Average process time: {avg_time:.2f} seconds")
            print(f"Load imbalance factor: {load_imbalance:.2f}")
            
            if results:
                results["performance"] = {
                    "total_time": max_time,
                    "avg_process_time": avg_time,
                    "load_imbalance": load_imbalance,
                    "processor_count": comm_size
                }
        else:
            print(f"Sequential analysis completed in {elapsed:.2f} seconds")
            
            if results:
                results["performance"] = {
                    "total_time": elapsed,
                    "processor_count": 1
                }
        
        # Save results if output path specified
        if output_path and results:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
            
        return results
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Mastodon data")
    parser.add_argument("-data", type=str, required=True, help="Path to Mastodon data file")
    parser.add_argument("-output", type=str, help="Path to save results")
    parser.add_argument("-chunk", type=int, default=10000, help="Chunk size for processing")
    
    args = parser.parse_args()
    
    # Run analysis
    parallel_analyze_mastodon_data(args.data, args.output, args.chunk)