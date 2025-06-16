# app.py
import os
import time
import hmac
import hashlib
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv
import logging
import yaml
import json
from jsonschema import validate, ValidationError
from github import Github, GithubException
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
import traceback
import re
import requests

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_file: str) -> Dict:
    """Loads and validates the configuration from a YAML file."""
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Error loading configuration: {e}")

    schema = {
        "type": "object",
        "properties": {
            "review_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["review_criteria"],
        "additionalProperties": False,
    }
    try:
        validate(instance=config, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")
    return config


config = load_config("config.yaml")  # Load config.yaml


# GitHub Client
class GitHubClient:

    def __init__(self, token: str):
        """
        Initializes the GitHub client.

        Args:
            token: The GitHub personal access token or installation token.
        """
        self.github = Github(token)
        self.token = token # Store the token for later use
        
    def get_pr_diff(self, owner: str, repo_name: str, pr_number: int) -> str:
        """
        Fetches the complete diff for all files in a pull request.
        
        Args:
            owner: Repository owner
            repo_name: Repository name
            pr_number: Pull request number
            
        Returns:
            Combined diff text for all files in the PR
        """
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            pr = repo.get_pull(pr_number)
            
            # Get all files from the PR
            all_files = pr.get_files()
            
            # Combine diffs from all files
            diffs = []
            for file in all_files:
                if file.patch:  # Some files might not have a patch (e.g., binary files)
                    file_header = f"diff --git a/{file.filename} b/{file.filename}"
                    diffs.append(f"{file_header}\n{file.patch}")
            
            combined_diff = "\n\n".join(diffs)
            return combined_diff
        except GithubException as e:
            logger.error(f"Error fetching PR diff: {traceback.format_exc()}")
            raise

    def get_file_diff(self, owner: str, repo_name: str, pr_number: int, file_path: str) -> str:
        """
        Fetches the diff for a specific file in a pull request.
        
        Args:
            owner: Repository owner
            repo_name: Repository name
            pr_number: Pull request number
            file_path: Path to the file
            
        Returns:
            Diff text for the specified file
        """
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            pr = repo.get_pull(pr_number)
            
            # Get all files from the PR
            all_files = pr.get_files()
            
            # Find the specific file
            for file in all_files:
                if file.filename == file_path:
                    return file.patch
            
            return ""  # File not found in PR
        except GithubException as e:
            logger.error(f"Error fetching file diff: {traceback.format_exc()}")
            raise

    def get_pr_files(self, owner: str, repo_name: str, pr_number: int) -> List[Dict]:
        """
        Fetches detailed information about files changed in a pull request.
        
        Returns:
            A list of dicts with filename, status, and other metadata
        """
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            pr = repo.get_pull(pr_number)
            files = pr.get_files()
            
            file_data = []
            for file in files:
                data = {
                    "filename": file.filename,
                    "status": file.status,  # Added, modified, removed, etc.
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "has_patch": bool(file.patch)  # Whether the file has a diff patch
                }
                file_data.append(data)
                
            return file_data
        except GithubException as e:
            logger.error(f"Error fetching PR files: {e}")
            raise

    def get_repo_content(self, owner: str, repo_name: str, file_path: str, ref: str = None) -> str:
        """
        Fetches the content of a file from the repository.
        
        Args:
            owner: Repository owner
            repo_name: Repository name
            file_path: Path to the file
            ref: Reference (branch, commit SHA) to get the file from
            
        Returns:
            Content of the file as a string
        """
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            content_file = repo.get_contents(file_path)
            return content_file.decoded_content.decode("utf-8")
        except GithubException as e:
            logger.error(f"Error fetching file content: {e}")
            return ""  # Return empty string if file not found or other error

        """Fetches the content of a file from the repository."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            content_file = repo.get_contents(file_path)
            return content_file.decoded_content.decode("utf-8")
        except GithubException as e:
            logger.error(f"Error fetching file content: {e}")
            return ""  # Return empty string if file not found or other error

    def create_review_comment(
        self,
        owner: str,
        repo_name: str,
        pr_number: int,
        comments: List[Dict],
    ) -> None:
        """Posts review comments to a pull request."""
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            logger.info(f"repo data {repo}")
            pr = repo.get_pull(pr_number)
            commit = repo.get_commit(sha=pr.head.sha)
            logger.info(f"commitdata {comments}")
            for comment_data in comments:
                if "line" in comment_data:
                    pr.create_review_comment(
                        body=comment_data["body"],
                        path=comment_data["path"],
                        commit=commit,
                        line=comment_data["line"],
                    )
                elif "start_line" in comment_data and "end_line" in comment_data:
                    pr.create_review_comment(
                        body=comment_data["body"],
                        path=comment_data["path"],
                        commit=commit,
                        start_line=comment_data["start_line"],
                        end_line=comment_data["end_line"],
                    )
                else:
                    pr.create_review_comment(
                        body=comment_data["body"], path=comment_data["path"],commit=commit,line=2,
                    )  # file level comment
            logger.info(
                f"Successfully posted {len(comments)} review comments to {owner}/{repo_name}#{pr_number}"
            )
        except GithubException as e:
            logger.error(
                f"Error posting comments to {owner}/{repo_name}#{pr_number}: {e}"
            )
            raise

    def get_installation_id(self, org_name: str) -> Optional[int]:
        """
        Retrieves the installation ID for a GitHub App in an organization.

        Args:
            org_name: The name of the organization.

        Returns:
            The installation ID, or None if not found.
        """
        try:
            org = self.github.get_organization(org_name)
            apps = org.get_apps()
            #  Find the app by name.  Assumes the app name is the same as
            #  the environment variable GITHUB_APP_NAME
            app_name = os.getenv("GITHUB_APP_NAME")
            if not app_name:
                logger.error("GITHUB_APP_NAME is not set.")
                return None
            for app in apps:
                if app.name == app_name:
                    installation = org.get_installation(app_id=app.id)
                    return installation.id
            logger.warning(f"Installation not found for app {app_name} in organization {org_name}")
            return None
        except GithubException as e:
            logger.error(f"Error getting installation ID: {e}")
            return None

    def get_installation_token(self, installation_id: int) -> Optional[str]:
        """
        Retrieves an installation access token for a GitHub App installation.

        Args:
            installation_id: The installation ID.

        Returns:
            The installation access token, or None on error.
        """
        try:
            app_id = os.getenv("GITHUB_APP_ID")
            private_key = os.getenv("GITHUB_PRIVATE_KEY")
            if not app_id or not private_key:
                logger.error("GITHUB_APP_ID or GITHUB_PRIVATE_KEY is not set.")
                return None

            from jwt import jwt  # PyJWT library

            now = int(time.time())
            payload = {
                "iat": now,
                "exp": now + 600,  # Token expires in 10 minutes
                "iss": app_id,
            }
            encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")

            # Use the JWT to get an installation access token.
            url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
            headers = {
                "Authorization": f"Bearer {encoded_jwt}",
                "Accept": "application/vnd.github.v3+json",
            }
            response = requests.post(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            token_data = response.json()
            return token_data["token"]
        except Exception as e:
            logger.error(f"Error getting installation token: {e}")
            return None

    def get_user_permission(self, owner: str, repo_name: str, username: str) -> str:
        """
        Gets the permission level of a user in a repository.

        Args:
            owner: The owner of the repository.
            repo_name: The name of the repository.
            username: The username of the user.

        Returns:
            The permission level (e.g., "admin", "write", "read"), or an empty string
            if the user is not found or an error occurs.
        """
        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            user = self.github.get_user(username)
            permission = repo.get_collaborator_permission(user)
            return permission
        except GithubException as e:
            logger.error(f"Error getting user permission: {e}")
            return ""

    def close(self):
        """Closes the GitHub connection."""
        self.github.close()


def get_github_client() -> GitHubClient:
    """
    Returns a GitHub client instance, authenticated either with a personal access
    token or a GitHub App installation token, depending on the environment variables.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
    GITHUB_PRIVATE_KEY = os.getenv("GITHUB_PRIVATE_KEY")
    logger.info(f"git token is set {GITHUB_APP_ID and GITHUB_PRIVATE_KEY}")
    if GITHUB_APP_ID and GITHUB_PRIVATE_KEY:
        #  We are running as a GitHub App.  We need to get an installation token.
        #  The installation ID is per-organization.
        org_name = os.getenv("GITHUB_ORG_NAME") #  Need the org name.
        if not org_name:
            raise ValueError("GITHUB_ORG_NAME is not set when using GitHub App authentication.")
        github_client = GitHubClient("") #  Start with empty token.
        installation_id = github_client.get_installation_id(org_name)
        if installation_id:
            installation_token = github_client.get_installation_token(installation_id)
            if installation_token:
                return GitHubClient(installation_token)  # Return client with installation token
            else:
                raise ValueError(
                    "Failed to obtain installation token."
                )  #  Don't fall back to GITHUB_TOKEN
        else:
             raise ValueError(
                    f"Installation not found for organization {org_name}"
                ) #  Don't fall back.
    elif github_token:
        logger.info("git token is set")
        return GitHubClient(github_token)  # Return client with personal access token
    else:
        raise ValueError(
            "GITHUB_TOKEN is not set, and the application is not configured as a GitHub App."
        )



# Authentication
def authenticate_user(headers: Dict) -> Optional[str]:
    """Authenticates the user from the Authorization header and returns the username."""
    auth_header = headers.get("Authorization")
    if not auth_header:
        return None
    try:
        auth_type, auth_token = auth_header.split(" ")
        if auth_type != "token":
            return None
        github_client = get_github_client()
        user = github_client.github.get_user()
        github_client.close()  # Close after getting the user.
        return user.login
    except GithubException as e:
        logger.error(f"Error authenticating user: {e}")
        return None
    except ValueError:
        return None  # Handle invalid Authorization header format


# Webhook verification
def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verifies the signature of a GitHub webhook request."""
    if not signature:
        return False
    expected_signature = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected_signature)


# Diff Analyzer
class DiffAnalyzer:
    """Analyzes diffs to identify changed code blocks."""

    def parse_diff(self, diff_text: str) -> List[Dict]:
        """Parses a diff string and returns a list of hunks."""
        hunks = []
        lines = diff_text.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("@@"):
                try:
                    match = re.match(r"@@ -\d+,\d+ \+(\d+),(\d+) @@", lines[i])  # start line number, number of lines.
                    if match:
                        start_line = int(match.group(1))
                        num_lines = int(match.group(2))
                        code_lines = []
                        i += 1
                        while i < len(lines) and not lines[i].startswith("@@"):
                            if lines[i].startswith("+"):
                                code_lines.append(
                                    {"type": "add", "line": lines[i][1:], "line_number": start_line}
                                )
                                start_line += 1
                            elif lines[i].startswith(" "):
                                start_line += 1
                            i += 1
                        hunks.append({"start_line": start_line - num_lines, "code_lines": code_lines})
                    else:
                        i += 1
                except Exception as e:
                    logger.error(f"Error parsing diff hunk: {e}")
                    i += 1
            else:
                i += 1
        return hunks


# Context Provider
class ContextProvider:
    """Provides context for the code review, such as relevant code from the surrounding file."""

    def __init__(self, github_client: GitHubClient):
        self.github_client = github_client

    def get_context(self, owner: str, repo_name: str) -> Dict[str, str]:
        """
        Fetches the entire file content for the changed files in the PR.

        Returns:
            A dictionary where the key is the file path and the value is the file content.
        """
        # This version gets the file list from the diff.
        # get_pr_files was removed
        context = {}
        # pr = self.github_client.get_pull(pr_number) # pr_number not available here
        # files = pr.get_files()
        # for file in files: # Now get files from the diff.
        #     file_path = file.filename
        #     file_content = self.github_client.get_repo_content(owner, repo_name, file_path)
        #     if file_content:
        #         context[file_path] = file_content
        return context


# LLM Prompt Engineer
class LLMPromptEngineer:
    """Generates prompts for the LLM based on the diff and context."""

    def __init__(self, config: Dict):
        self.config = config

    def create_prompt(
        self, diff_text: str, file_contents: str, context: Dict[str, str], review_criteria: List[str]
    ) -> str:
        """Creates a prompt for the LLM."""
        prompt = f"""
        You are a code reviewer providing feedback on a pull request.  
        
        Here is the diff of the changes:
        
        {diff_text}
        
        Review Criteria:
        {review_criteria}
        
        path:
        {file_contents}
        Provide feedback.Include the file name and line number.
        """
        return prompt



# LLM Client Interface
class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """

    @abstractmethod
    def get_response(self, prompt: str, model_name: str) -> str:
        """
        Sends a prompt to the LLM and returns the response.

        Args:
            prompt: The prompt string.
            model_name: The name of the LLM model to use.

        Returns:
            The LLM's response as a string.
        """
        pass



# OpenAI Client
class OpenAIClient(LLMClient):
    """Client for interacting with the OpenAI API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def get_response(self, prompt: str, model_name: str) -> str:
        """Sends a prompt to the OpenAI API and returns the response."""
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with OpenAI: {e}")
            return ""  # Return empty string on error
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return ""



# Gemini Client
class GeminiClient(LLMClient):
    """Client for interacting with the Google Gemini API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"  # Corrected URL
        self.headers = {"Content-Type": "application/json"}

    def get_response(self, prompt: str, model_name: str) -> str:
        """Sends a prompt to the Gemini API and returns the response.

        Args:
            prompt: The prompt string.
            model_name: The name of the Gemini model to use (e.g., "gemini-pro").  This parameter is included for consistency
                        with the LLMClient interface, but Gemini's API doesn't strictly require it in the same way as OpenAI.

        Returns:
            The LLM's response as a string, or an empty string on error.
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }
        params = {"key": self.api_key}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, params=params)
            response.raise_for_status()
            data = response.json()
            # The response structure is different from OpenAI.  The actual text is nested.
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Gemini: {e}")
            return ""
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.error(f"Error parsing Gemini response: {e}")
            return ""


# Claude Client
class ClaudeClient(LLMClient):
    """Client for interacting with the Anthropic Claude API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.anthropic.com/v1/messages"  # Changed to /messages
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "Anthropic-Version": "2024-04-01",  # Specify the Anthropic API version
        }

    def get_response(self, prompt: str, model_name: str) -> str:
        """Sends a prompt to the Anthropic Claude API and returns the response."""
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],  # Use messages structure
            "max_tokens": 4000,  # Or an appropriate value for your use case
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            #  The response structure is different for the /messages endpoint.
            return data["content"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Claude: {e}")
            return ""
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing Claude response: {e}")
            return ""



# Review Comment Formatter
class ReviewCommentFormatter:
    """Formats LLM responses into GitHub review comments."""
    def format_comments(self, llm_response: str) -> List[Dict]:
        """Formats the LLM response into a list of GitHub review comments."""
        comments = []
        lines = llm_response.splitlines()
        current_file = None  # Store the last detected filename

        for line in lines:
            # Capture filenames inside backticks or Markdown-style headers
            file_match = re.search(r"\*\*\d+\.\s+`([^`]+)`\*\*", line)
            if file_match:
                current_file = file_match.group(1).strip()  # Extract file name
                continue  # Move to next line

            if current_file and line.strip():  # Associate comments with extracted file
                comments.append({"path": current_file, "body": line.strip()})

        return comments


# Main App Route
@app.route("/webhook", methods=["POST"])
def github_webhook():
    """
    Handles GitHub webhook events, specifically for pull requests.
    """
    payload = request.get_data()
    signature = request.headers.get("X-Hub-Signature-256")
    #  Retrieve the webhook secret from environment variable
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRE")

    if webhook_secret:
        if not verify_github_signature(payload, signature, webhook_secret):
            logger.warning("Invalid GitHub webhook signature.")
            return jsonify({"message": "Invalid signature"}), 401
    else:
        logger.warning(
            "GITHUB_WEBHOOK_SECRET is not set.  Skipping signature verification."
        )

    event = request.headers.get("X-GitHub-Event")
    if event == "ping":
        return jsonify({"message": "Ping received"}), 200

    if event == "pull_request":
        payload_json = request.get_json()
        action = payload_json["action"]
        # changed to a list
        valid_actions = ["opened", "synchronize", "reopened"]
        if action in valid_actions:
            owner = payload_json["repository"]["owner"]["login"]
            repo_name = payload_json["repository"]["name"]
            pr_number = payload_json["number"]
            logger.info(
                f"Received pull_request event for {owner}/{repo_name}#{pr_number} (action: {action})"
            )
            # Call review_pr function directly
            try:
                review_pr(owner, repo_name, pr_number)
                return jsonify({"message": "Pull request review initiated"}), 202  # 202 Accepted
            except Exception as e:
                logger.error(f"Error reviewing PR: {traceback.format_exc()}")
                return jsonify({"message": "Error reviewing PR"}), 500

        else:
            logger.info(f"Ignoring pull_request action: {action}")
            return jsonify({"message": f"Ignoring pull_request action: {action}"}), 200
    else:
        logger.info(f"Ignoring event: {event}")
        return jsonify({"message": f"Ignoring event: {event}"}), 200
    return jsonify({"message": "OK"}), 200



def review_pr(owner, repo_name, pr_number): 
    start_time = time.time()
    logger.info(f"Starting code review for {owner}/{repo_name}#{pr_number}")
    github_client = get_github_client()  # Get the GitHub client instance
    try:
        # 1. Fetch the diff
        diff_text = github_client.get_pr_diff(owner, repo_name, pr_number)
        logger.info(f"diff text {diff_text}")
        if not diff_text:
            logger.warning(
                f"No diff found for {owner}/{repo_name}#{pr_number}.  Skipping review."
            )
            return

        # 2. Analyze the diff
        diff_analyzer = DiffAnalyzer()
        hunks = diff_analyzer.parse_diff(diff_text)

        # 3. Fetch file contents
        files = github_client.get_pr_files(owner, repo_name, pr_number)
        file_contents = {}
        for file_data in files:
            file_content = github_client.get_repo_content(
                owner, repo_name, file_data["filename"]
            )
            if file_content:
                logger.error(f'filecontent {file_data}')
                file_contents[file_data["filename"]] = file_content

        # 4. Get context
        context_provider = ContextProvider(github_client)
        context = context_provider.get_context(owner, repo_name)

        # 5. Generate LLM prompt
        prompt_engineer = LLMPromptEngineer(config)
        prompt = prompt_engineer.create_prompt(
            diff_text, file_data["filename"], context, config["review_criteria"]
        )

        # 6. Get LLM response
        llm_client = None
        LLM_PROVIDER=os.getenv("LLM_PROVIDER")
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
        if LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set.")
            llm_client = OpenAIClient(OPENAI_API_KEY)
        elif LLM_PROVIDER == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")
            llm_client = GeminiClient(GEMINI_API_KEY)
        elif LLM_PROVIDER == "claude":
            if not CLAUDE_API_KEY:
                raise ValueError("CLAUDE_API_KEY is not set.")
            llm_client = ClaudeClient(CLAUDE_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
        #logger.error(f'prompts {prompt}')
        llm_response = llm_client.get_response(prompt, 'gemini-pro')  # Hardcoded gpt-4

        # 7. Format comments
        #logger.info(f'llmres {llm_response}')
        comment_formatter = ReviewCommentFormatter()
        comments = comment_formatter.format_comments(llm_response)

        # 8. Post comments (after authentication)
        #  Check if we are running as a GitHub App.  If so, we should have
        #  an installation ID in the environment.
        logger.info( f'app key {comments}')
        if False and GITHUB_APP_ID and GITHUB_PRIVATE_KEY:
            org_name = payload_json["repository"]["owner"]["login"]  # needed for app authentication
            installation_id = github_client.get_installation_id(org_name)
            if installation_id:
                installation_token = github_client.get_installation_token(installation_id)
                if installation_token:
                    # Re-initialize the GitHub client with the installation token.
                    github_client = GitHubClient(installation_token)
                    github_client.create_review_comment(owner, repo_name, pr_number, comments)
                else:
                    logger.error(f"Failed to obtain installation token for {owner}")
            else:
                logger.error(f"Installation not found for {org_name}")
        else:  # If not a GitHub App, use user authentication.
            # 8. Get the authenticated user.
            username = authenticate_user(request.headers)
            if username:
                # 9.  Check user permissions.
                permission = github_client.get_user_permission(owner, repo_name, username)
                if permission in ["admin", "write"]:
                    github_client.create_review_comment(owner, repo_name, pr_number, comments)
                else:
                    logger.warning(
                        f"User {username} does not have sufficient permissions to comment on {owner}/{repo_name}#{pr_number}.  Required permissions are admin or write, user has {permission}"
                    )
            else:
                logger.warning(
                    f"Authentication required to post comments on {owner}/{repo_name}#{pr_number}."
                )
                # Return the comments in the response for the user to post manually.
                #  This is ONLY done if the request is NOT from a webhook.  For webhooks,
                #  we do not have a user context, so we can't return the comments.
                if request.headers.get("X-GitHub-Event") != "pull_request":
                    return jsonify(
                        {
                            "message": "Authentication required to post comments.  Please authenticate and try again.",
                            "comments": comments,
                        }
                    ), 403
    except Exception as e:
        logger.error(f"Error reviewing PR {owner}/{repo_name}#{pr_number}: {e}")
        raise  # Re-raise the exception so Celery's retry mechanism can handle it

    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(
            f"Finished code review for {owner}/{repo_name}#{pr_number} in {duration:.2f} seconds."
        )
        if github_client:
            github_client.close()



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 6969)))
