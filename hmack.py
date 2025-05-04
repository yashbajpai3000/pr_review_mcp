import hmac
import hashlib
import json

def calculate_signature(secret, payload):
    """
    Calculates the X-Hub-Signature-256 header value.

    Args:
        secret: The GitHub webhook secret (string).
        payload: The raw JSON payload string (string).

    Returns:
        The X-Hub-Signature-256 header value (string).
    """
    # Ensure the payload is a bytes-like object
    if not isinstance(payload, bytes):
        payload = payload.encode('utf-8')

    # Calculate the HMAC-SHA256 hash
    hmac_hash = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256)
    signature = "sha256=" + hmac_hash.hexdigest()
    return signature

# Example usage (replace with your actual secret and payload)
webhook_secret = "YOUR_GITHUB_WEBHOOK_SECRET"  #  Your secret from .env
payload = json.dumps({  #  The JSON payload from Postman's body, as a string
          "action": "opened",
          "number": 123,
          "pull_request": {
            "head": {
              "ref": "feature/new-feature",
              "sha": "abcdef1234567890abcdef1234567890abcdef12",
              "repo": {
                "name": "pr_review_mcp",
                "owner": {
                  "login": "yashbajpai3000"
                }
              }
            },
            "base": {
              "ref": "main"
            },
            "url": "https://api.github.com/repos/yashbajpai3000/pr_review_mcp/pulls/123"
          },
          "repository": {
            "name": "pr_review_mcp",
            "owner": {
              "login": "yashbajpai3000"
            }
          }
        })
signature = calculate_signature(webhook_secret, payload)
print(f"X-Hub-Signature-256: {signature}")  # Print the signature to use in Postman
