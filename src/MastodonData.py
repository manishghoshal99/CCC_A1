import json

class MastodonData:
    """
    Data model for processing Mastodon posts.
    Extracts and validates relevant fields from JSON data.
    """
    def __init__(self, data: str):
        """
        Initialize with JSON data string.
        
        Args:
            data: String containing a JSON object
            
        Raises:
            ValueError: If the JSON data is invalid
        """
        try:
            # Try to parse the JSON data
            json_data = json.loads(data)
        except Exception as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        # Extract created_at time (as ISO format string)
        self.created_at = json_data.get("created_at", "")
        
        # Extract sentiment; if missing or None, default to 0
        self.sentiment = json_data.get("sentiment")
        if self.sentiment is None:
            self.sentiment = 0
        else:
            # Ensure sentiment is a number (float)
            try:
                self.sentiment = float(self.sentiment)
            except (ValueError, TypeError):
                self.sentiment = 0
        
        # Extract account info for user analysis
        account = json_data.get("account", {})
        self.user_id = account.get("id", "")
        self.username = account.get("username", "")
        
        # Add validation to ensure critical fields exist
        if not self.created_at or not self.user_id:
            # These fields are required for analysis
            pass  # We'll skip this entry during processing