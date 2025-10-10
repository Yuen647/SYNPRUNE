import requests
import json
import ast
from datetime import datetime
import time
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configuration Parameters
GITHUB_TOKEN = 'YOUR-GITHUB-TOKEN'

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Search for active Python repositories created after 2024
SEARCH_QUERY = "language:python created:>=2024-01-01"
OUTPUT_FILE = "negative_raw.jsonl"
STATE_FILE = "crawler_state.json"  # File to save crawling state
FUNCTIONS_PER_REPO = 10  # Maximum number of functions to collect per repository
MAX_RETRIES = 10  # Maximum number of retries for requests
RETRY_BACKOFF_FACTOR = 2  # Multiplier for retry waiting time
NETWORK_ERROR_SLEEP = 30  # Waiting time (seconds) after network errors

def create_session():
    """Create a requests session with retry mechanism"""
    session = requests.Session()
    
    # Set up retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504, 408],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    # Apply retry strategy to HTTP and HTTPS connections
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

def save_state(state):
    """Save current crawling state to file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_state():
    """Load crawling state from file"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'current_page': 1,
        'processed_repos': [],
        'processed_files': set(),
        'repo_function_counts': {}
    }

def parse_functions(source_code):
    """Parse Python code and return list of functions"""
    try:
        tree = ast.parse(source_code)
    except Exception as e:
        print(f"Failed to parse code: {str(e)}")
        return []
    
    functions = []
    source_lines = source_code.split('\n')
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not hasattr(node, 'end_lineno'):
                continue  # Skip functions where end line number cannot be obtained
            
            start_line = node.lineno - 1
            end_line = node.end_lineno
            function_code = '\n'.join(source_lines[start_line:end_line])
            
            functions.append({
                'name': node.name,
                'code': function_code
            })
    
    return functions

def fetch_github_api(url, params=None, session=None):
    """Send GitHub API request and handle rate limits and network exceptions"""
    if not session:
        session = create_session()
        
    retries = 0
    
    while retries < MAX_RETRIES:
        try:
            response = session.get(url, headers=HEADERS, params=params, timeout=60, verify=True)
            
            if response.status_code == 403 and 'rate limit' in response.text.lower():
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                sleep_time = max(reset_time - time.time(), 0) + 5
                print(f"Rate limited, waiting {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                continue
                
            return response
            
        except requests.exceptions.SSLError as e:
            print(f"SSL Error: {str(e)}, attempting retry {retries+1}/{MAX_RETRIES}")
            retries += 1
            time.sleep(NETWORK_ERROR_SLEEP)  # Wait longer after network errors
            
        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {str(e)}, attempting retry {retries+1}/{MAX_RETRIES}")
            retries += 1
            time.sleep(NETWORK_ERROR_SLEEP)  # Wait longer after network errors
    
    print(f"Reached maximum retry attempts, skipping URL: {url}")
    return None

def get_file_creation_date(repo_full_name, file_path, commit_sha, session=None):
    """Get file creation date (first commit date)"""
    # Get file commit history
    commits_url = f"https://api.github.com/repos/{repo_full_name}/commits?path={file_path}&per_page=100"
    commits_response = fetch_github_api(commits_url, session=session)
    
    if not commits_response or commits_response.status_code != 200:
        return None
    
    commits = commits_response.json()
    if not commits:
        return None
    
    # The last commit is the first commit of the file
    first_commit = commits[-1]
    return first_commit['commit']['committer']['date']

def main():
    # Load saved state
    state = load_state()
    page = state['current_page']
    processed_repos = state['processed_repos']
    processed_files = set(state.get('processed_files', []))
    repo_function_counts = state.get('repo_function_counts', {})
    
    session = create_session()
    
    with open(OUTPUT_FILE, 'a') as out_file:  # Open in append mode
        while True:
            # Search for eligible repositories
            search_url = f"https://api.github.com/search/repositories?q={SEARCH_QUERY}&per_page=100&page={page}"
            response = fetch_github_api(search_url, session=session)
            
            if not response or response.status_code != 200:
                print(f"Search failed, waiting {NETWORK_ERROR_SLEEP} seconds before retrying")
                time.sleep(NETWORK_ERROR_SLEEP)
                continue
                
            data = response.json()
            if not data.get('items'):
                print(f"No search results on page {page}, exiting")
                break
                
            print(f"Processing page {page}, total {len(data['items'])} repositories")
            
            for repo_item in data['items']:
                repo_full_name = repo_item['full_name']
                
                # Skip already processed repositories
                if repo_full_name in processed_repos:
                    print(f"Skipping processed repository: {repo_full_name}")
                    continue
                
                stars = repo_item['stargazers_count']
                print(f"\nProcessing repository: {repo_full_name} (Stars: {stars})")
                
                # Get or initialize repository function counter
                repo_function_count = repo_function_counts.get(repo_full_name, 0)
                
                # Get repository's default branch
                repo_url = f"https://api.github.com/repos/{repo_full_name}"
                repo_response = fetch_github_api(repo_url, session=session)
                if not repo_response or repo_response.status_code != 200:
                    print(f"Failed to get repository information, skipping {repo_full_name}")
                    continue
                default_branch = repo_response.json().get('default_branch', 'main')
                
                # Get list of Python files in the repository (recursive retrieval)
                contents_url = f"https://api.github.com/repos/{repo_full_name}/contents?ref={default_branch}"
                stack = [contents_url]
                
                try:
                    while stack:
                        # Check if function count limit is reached
                        if repo_function_count >= FUNCTIONS_PER_REPO:
                            print(f"Collected {repo_function_count} functions from {repo_full_name}, limit reached, moving to next repository")
                            break
                        
                        current_url = stack.pop()
                        contents_response = fetch_github_api(current_url, session=session)
                        
                        if not contents_response or contents_response.status_code != 200:
                            print(f"Failed to get directory contents, skipping {current_url}")
                            continue
                        
                        items = contents_response.json()
                        
                        for item in items:
                            # Check if function count limit is reached
                            if repo_function_count >= FUNCTIONS_PER_REPO:
                                print(f"Collected {repo_function_count} functions from {repo_full_name}, limit reached, moving to next repository")
                                break
                                
                            if item['type'] == 'dir':
                                # Recursively process subdirectories
                                stack.append(item['url'])
                            elif item['name'].endswith('.py'):
                                file_path = item['path']
                                file_id = f"{repo_full_name}/{file_path}"
                                
                                if file_id in processed_files:
                                    continue
                                
                                # Get file content
                                download_url = item['download_url']
                                file_response = fetch_github_api(download_url, session=session)
                                
                                if not file_response or file_response.status_code != 200:
                                    print(f"Failed to get file content, skipping {file_id}")
                                    continue
                                
                                # Get SHA while retrieving file content (for querying commit history)
                                content_data = file_response.json() if 'json' in file_response.headers.get('Content-Type', '') else {}
                                sha = content_data.get('sha', '')
                                
                                # Get file creation date
                                creation_date_str = get_file_creation_date(repo_full_name, file_path, sha, session=session)
                                if not creation_date_str:
                                    print(f"Failed to get file creation date, skipping {file_id}")
                                    continue
                                
                                creation_date = datetime.strptime(creation_date_str, "%Y-%m-%dT%H:%M:%SZ")
                                
                                # Strictly check file creation date
                                if creation_date < datetime(2024, 1, 1):
                                    print(f"File {file_id} was created on {creation_date_str}, which is before 2024-01-01, skipping")
                                    continue
                                
                                # Parse functions in the file
                                source_code = file_response.text
                                functions = parse_functions(source_code)
                                
                                if not functions:
                                    print(f"No functions found in file {file_id}")
                                    continue
                                
                                print(f"Parsed {len(functions)} functions from file {file_id}")
                                
                                # Write function records and update counter
                                for func in functions:
                                    if repo_function_count >= FUNCTIONS_PER_REPO:
                                        break
                                        
                                    record = {
                                        'function': func['code'],
                                        'creation_date': creation_date_str,
                                        'repo': repo_full_name,
                                        'file_path': file_path,
                                        'stars': stars,
                                        'label': 0
                                    }
                                    out_file.write(json.dumps(record) + '\n')
                                    repo_function_count += 1
                                    
                                    # Save state after every 10 functions processed
                                    if repo_function_count % 10 == 0:
                                        state = {
                                            'current_page': page,
                                            'processed_repos': processed_repos,
                                            'processed_files': list(processed_files),
                                            'repo_function_counts': repo_function_counts
                                        }
                                        save_state(state)
                                        print(f"State saved: Collected {repo_function_count} functions from repository {repo_full_name}")
                                
                                # Mark file as processed
                                processed_files.add(file_id)
                                
                        # Save state after processing each directory
                        state = {
                            'current_page': page,
                            'processed_repos': processed_repos,
                            'processed_files': list(processed_files),
                            'repo_function_counts': repo_function_counts
                        }
                        save_state(state)
                    
                except KeyboardInterrupt:
                    print("User interruption detected, saving current state...")
                    state = {
                        'current_page': page,
                        'processed_repos': processed_repos,
                        'processed_files': list(processed_files),
                        'repo_function_counts': repo_function_counts
                    }
                    save_state(state)
                    print("State saved, program exiting. Next run will resume from the last interruption point.")
                    return
                    
                except Exception as e:
                    print(f"Unexpected error occurred while processing repository {repo_full_name}: {str(e)}")
                    print("Saving current state and skipping this repository...")
                    state = {
                        'current_page': page,
                        'processed_repos': processed_repos,
                        'processed_files': list(processed_files),
                        'repo_function_counts': repo_function_counts
                    }
                    save_state(state)
                    continue
                
                # Update repository function count
                repo_function_counts[repo_full_name] = repo_function_count
                
                # Mark repository as processed
                processed_repos.append(repo_full_name)
                
                # Save state
                state = {
                    'current_page': page,
                    'processed_repos': processed_repos,
                    'processed_files': list(processed_files),
                    'repo_function_counts': repo_function_counts
                }
                save_state(state)
                
                print(f"Completed processing repository {repo_full_name}, collected {repo_function_count} functions in total")
            
            # Check if there is a next page
            if 'next' in response.links:
                page += 1
                # Save page number state
                state = {
                    'current_page': page,
                    'processed_repos': processed_repos,
                    'processed_files': list(processed_files),
                    'repo_function_counts': repo_function_counts
                }
                save_state(state)
                print(f"State saved: About to process page {page}")
            else:
                print(f"No next page, processing completed")
                break

if __name__ == "__main__":
    main()

'''
Usage Instructions

1. Run the script normally:
    ```bash
    python collect_script.py
    ```

2. To start over from scratch, delete the state file:
    ```bash
    rm crawler_state.json
    ```

3. After network recovery, simply re-run the script to resume from the interruption point

4. Adjust parameters:
    ```python
    # Increase retry count or extend waiting time
    MAX_RETRIES = 15
    NETWORK_ERROR_SLEEP = 60
    ```
'''