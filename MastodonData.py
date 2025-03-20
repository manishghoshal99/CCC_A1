import json

class MastodonData:
    """Data model for processing Mastodon posts."""
    def __init__(self, data: str):
        try:
            json_data = json.loads(data)
        except Exception as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        # Extract created_at time (as ISO format string)
        self.created_at = json_data.get("created_at", "")
        
        # Extract sentiment; if missing, default to 0
        self.sentiment = json_data.get("sentiment", 0)
        
        # Extract account info for user analysis
        account = json_data.get("account", {})
        self.user_id = account.get("id", "")
        self.username = account.get("username", "")
