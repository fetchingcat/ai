import os
import sys
import json
import requests
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import time
# Add this new import
from googlesearch import search

class OllamaClient:
    """Client for interacting with the Ollama API."""
    
    def __init__(self, host="http://localhost:11434"):
        self.host = host
        self.generate_url = f"{host}/api/generate"
        self.chat_url = f"{host}/api/chat"
        self.list_url = f"{host}/api/tags"
    
    def list_models(self):
        """List all available models."""
        try:
            response = requests.get(self.list_url)
            response.raise_for_status()
            return response.json()["models"]
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
    
    def chat(self, messages, model="llama3", options=None):
        """Have a conversation with the specified model."""
        if options is None:
            options = {}
            
        payload = {
            "model": model,
            "messages": messages,
            **options
        }
        
        try:
            # Use stream=True to handle streaming responses
            response = requests.post(self.chat_url, json=payload, stream=True)
            response.raise_for_status()
            
            # Process streaming response
            full_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each line as a separate JSON object
                        chunk = json.loads(line.decode('utf-8'))
                        if "message" in chunk and "content" in chunk["message"]:
                            content_part = chunk["message"]["content"]
                            full_content += content_part
                            # Optional: print streaming output in real-time
                            # print(content_part, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            
            # Strip any thinking tags from the final content
            cleaned_content = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL)
            return cleaned_content
        except requests.exceptions.RequestException as e:
            print(f"Error in chat: {e}")
            return ""

class CommandExecutor:
    """Executes file system commands safely."""
    
    def __init__(self, base_dir=None):
        # Use current directory if none specified
        self.base_dir = base_dir or os.getcwd()
        
        # Create base_dir if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # List of allowed commands
        self.allowed_commands = {
            'create_file': self.create_file,
            'read_file': self.read_file,
            'append_file': self.append_file,
            'rename_file': self.rename_file,
            'delete_file': self.delete_file,
            'create_dir': self.create_dir,
            'list_dir': self.list_dir,
            'delete_dir': self.delete_dir,
            'list_files': self.list_files,
            'file_exists': self.file_exists,
            'dir_exists': self.dir_exists,
            'update_file': self.update_file,
            'execute_python': self.execute_python,
            'google_search': self.google_search,
            'download_file': self.download_file,
            'web_search': self.web_search,
            'fetch_url': self.fetch_url
        }
    
    def sanitize_path(self, path):
        """Make sure the path is within the base directory."""
        # Convert to absolute path
        abs_path = os.path.abspath(os.path.join(self.base_dir, path))
        
        # Check if the path is within the base directory
        if not abs_path.startswith(self.base_dir):
            raise ValueError(f"Path {path} is outside the allowed directory")
        
        return abs_path
    
    def create_file(self, path, content=""):
        """Create a new file with the given content."""
        full_path = self.sanitize_path(path)
        directory = os.path.dirname(full_path)
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"File created: {path}"
    
    def read_file(self, path):
        """Read the contents of a file."""
        full_path = self.sanitize_path(path)
        
        if not os.path.exists(full_path):
            return f"Error: File {path} does not exist"
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def append_file(self, path, content):
        """Append content to an existing file."""
        full_path = self.sanitize_path(path)
        
        try:
            with open(full_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Content appended to {path}"
        except Exception as e:
            return f"Error appending to file: {str(e)}"
    
    def rename_file(self, old_path, new_path):
        """Rename a file or directory."""
        full_old_path = self.sanitize_path(old_path)
        full_new_path = self.sanitize_path(new_path)
        
        if not os.path.exists(full_old_path):
            return f"Error: {old_path} does not exist"
        
        try:
            shutil.move(full_old_path, full_new_path)
            return f"Renamed {old_path} to {new_path}"
        except Exception as e:
            return f"Error renaming: {str(e)}"
    
    def delete_file(self, path):
        """Delete a file."""
        full_path = self.sanitize_path(path)
        
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return f"Error: File {path} does not exist"
        
        try:
            os.remove(full_path)
            return f"File deleted: {path}"
        except Exception as e:
            return f"Error deleting file: {str(e)}"
    
    def create_dir(self, path):
        """Create a directory."""
        full_path = self.sanitize_path(path)
        
        try:
            os.makedirs(full_path, exist_ok=True)
            return f"Directory created: {path}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"
    
    def list_dir(self, path="."):
        """List contents of a directory."""
        full_path = self.sanitize_path(path)
        
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return f"Error: Directory {path} does not exist"
        
        try:
            contents = os.listdir(full_path)
            files = [f for f in contents if os.path.isfile(os.path.join(full_path, f))]
            dirs = [d for d in contents if os.path.isdir(os.path.join(full_path, d))]
            
            return {
                "directories": dirs,
                "files": files
            }
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def delete_dir(self, path):
        """Delete a directory and its contents."""
        full_path = self.sanitize_path(path)
        
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return f"Error: Directory {path} does not exist"
        
        try:
            shutil.rmtree(full_path)
            return f"Directory deleted: {path}"
        except Exception as e:
            return f"Error deleting directory: {str(e)}"
    
    def list_files(self, path=".", pattern="*"):
        """List files in a directory matching a pattern."""
        from glob import glob
        
        full_path = self.sanitize_path(path)
        
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return f"Error: Directory {path} does not exist"
        
        try:
            matched_files = glob(os.path.join(full_path, pattern))
            return [os.path.basename(f) for f in matched_files if os.path.isfile(f)]
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    def file_exists(self, path):
        """Check if a file exists."""
        full_path = self.sanitize_path(path)
        return os.path.isfile(full_path)
    
    def dir_exists(self, path):
        """Check if a directory exists."""
        full_path = self.sanitize_path(path)
        return os.path.isdir(full_path)
    
    def update_file(self, path, content=""):
        """Update (overwrite) the contents of an existing file."""
        full_path = self.sanitize_path(path)
        
        if not os.path.exists(full_path):
            return f"Error: File {path} does not exist. Use create_file instead."
        
        if not os.path.isfile(full_path):
            return f"Error: {path} is not a file"
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File updated: {path}"
        except Exception as e:
            return f"Error updating file: {str(e)}"

    def execute_python(self, path_or_code, is_file=True, timeout=10):
        """Execute a Python script file or code string with safety measures."""
        import subprocess
        import tempfile
        
        try:
            if is_file == "True" or is_file is True:
                # Execute a Python file
                full_path = self.sanitize_path(path_or_code)
                if not os.path.exists(full_path):
                    return f"Error: Python file {path_or_code} does not exist"
                    
                cmd = [sys.executable, full_path]
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                # Execute Python code from a string
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp:
                    temp.write(path_or_code)
                    temp_path = temp.name
                    
                cmd = [sys.executable, temp_path]
                process = subprocess.run(
                    cmd,
                    capture_output=True, 
                    text=True,
                    timeout=timeout
                )
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            # Prepare the result
            result = {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "return_code": process.returncode
            }
            
            # Simple result summary
            if process.returncode == 0:
                if process.stdout.strip():
                    return f"Output:\n{process.stdout}"
                else:
                    return "Executed successfully with no output"
            else:
                return f"Error (code {process.returncode}):\n{process.stderr}"
                
        except subprocess.TimeoutExpired:
            return f"Error: Execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing Python: {str(e)}"
    
    def google_search(self, query, num_results=5):
        """Search Google and return top results."""
        try:
            # Convert num_results to an integer if it's a string
            if isinstance(num_results, str):
                num_results = int(num_results)
                
            results = []
            for url in search(query, num_results=num_results):
                results.append(url)
            
            if results:
                return {
                    "query": query,
                    "results": results
                }
            else:
                return f"No results found for query: {query}"
        except ValueError as e:
            return f"Error in Google search: Invalid number of results specified"
        except Exception as e:
            return f"Error performing Google search: {str(e)}"

    def download_file(self, url, save_path, timeout=30):
        """Download a file from a URL."""
        full_path = self.sanitize_path(save_path)
        directory = os.path.dirname(full_path)
        
        try:
            os.makedirs(directory, exist_ok=True)
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return f"Downloaded {url} to {save_path}"
        except Exception as e:
            return f"Error downloading file: {str(e)}"
    
    def web_search(self, query, num_results=3):
        """Search the web and extract information to answer a query."""
        try:
            # First get search results
            search_results = self.google_search(query, num_results)
            
            if isinstance(search_results, str) or not search_results.get('results'):
                return f"No results found for: {query}"
            
            urls = search_results['results'][:3]  # Limit to first 3 to avoid overloading
            
            # Import here to avoid requiring these libraries unless needed
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch and extract content from each page
            all_content = []
            for url in urls[:2]:  # Only process first 2 URLs to be efficient
                try:
                    # Add user agent to avoid being blocked
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                        
                    # Get text and clean it up
                    text = soup.get_text(separator=' ', strip=True)
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    
                    # Take first 500 characters as a sample
                    content = ' '.join(lines)[:500] + "..."
                    all_content.append(f"From {url}:\n{content}\n")
                except Exception as e:
                    all_content.append(f"Error extracting from {url}: {str(e)}")
            
            result = f"Web search results for '{query}':\n\n" + "\n\n".join(all_content)
            return result
        except Exception as e:
            return f"Error in web search: {str(e)}"
    
    def fetch_url(self, url, extract_text=True):
        """Fetch and return the contents of a specific URL."""
        try:
            # Add user agent to avoid being blocked
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            if extract_text == "True" or extract_text is True:
                # Import here to avoid requiring these libraries unless needed
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                    
                # Get text and clean it up
                text = soup.get_text(separator=' ', strip=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                content = '\n'.join(lines)
                
                # Return first 2000 chars if content is too long
                if len(content) > 2000:
                    return content[:2000] + "...\n[Content truncated, total length: " + str(len(content)) + " chars]"
                return content
            else:
                # Return raw HTML (first 2000 chars if too long)
                content = response.text
                if len(content) > 2000:
                    return content[:2000] + "...\n[Content truncated, total length: " + str(len(content)) + " chars]"
                return content
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
    
    def execute_command(self, command_name, *args):
        """Execute a command by name with arguments."""
        print(f"Executing: {command_name} with args: {args}")
        
        if command_name not in self.allowed_commands:
            print(f"Error: Unknown command '{command_name}'")
            return f"Error: Unknown command '{command_name}'"
        
        try:
            result = self.allowed_commands[command_name](*args)
            print(f"Command result: {result}")
            return result
        except Exception as e:
            print(f"Error executing {command_name}: {str(e)}")
            return f"Error executing {command_name}: {str(e)}"

class OllamaCommandInterface:
    """Interface for using Ollama to execute commands."""
    
    def __init__(self, model="llama3", base_dir=None):
        self.client = OllamaClient()
        self.executor = CommandExecutor(base_dir)
        self.model = model
        
        self.system_message = """
You are CommandGPT, an AI assistant specialized in file management and code generation.

## YOUR PRIMARY DIRECTIVE:
You MUST use <command> tags for ANY file operation. DO NOT suggest operations without using commands.

## IMPORTANT EXECUTION NOTICE:
ANY command you write in <command> tags WILL BE IMMEDIATELY EXECUTED on the user's actual file system.
This is not a simulation. Your commands directly modify real files, create real directories, and execute real code.
Be extremely careful with commands that modify or delete existing files.

## COMMAND SYNTAX:
All commands must follow this EXACT format:
<command>command_name:argument1|argument2|...</command>

## AVAILABLE COMMANDS:

### File Operations:
1. create_file:path|content
   - Creates a new file with specified content
   - Example: <command>create_file:example.py|def hello():
    # This is an indented multi-line function
    print("Hello world! This is a multi-line file example")
    
# Main code
if __name__ == "__main__":
    hello()</command>

2. read_file:path
   - Reads the content of a file
   - Example: <command>read_file:example.py</command>

3. update_file:path|content
   - Updates an existing file with new content (overwrites)
   - Example: <command>update_file:example.py|print("Updated content")</command>

4. append_file:path|content
   - Adds content to the end of an existing file
   - Example: <command>append_file:log.txt|New log entry added on March 13</command>

5. rename_file:old_path|new_path
   - Renames a file or moves it to a new location
   - Example: <command>rename_file:old_name.txt|new_name.txt</command>

6. delete_file:path
   - Permanently deletes a file
   - Example: <command>delete_file:temp.txt</command>

7. file_exists:path
   - Checks if a file exists
   - Example: <command>file_exists:document.txt</command>

### Directory Operations:
8. create_dir:path
   - Creates a new directory
   - Example: <command>create_dir:new_folder</command>

9. list_dir:path
   - Lists all files and directories in a directory
   - Example: <command>list_dir:project</command>

10. delete_dir:path
    - Permanently deletes a directory and all its contents
    - Example: <command>delete_dir:old_folder</command>

11. dir_exists:path
    - Checks if a directory exists
    - Example: <command>dir_exists:project_folder</command>

12. list_files:path|pattern
    - Lists files matching a pattern in a directory
    - Example: <command>list_files:project|*.py</command>

### Code Execution:
13. execute_python:code|is_file
    - Executes Python code (either from a file or a string)
    - Example (from file): <command>execute_python:script.py|True</command>
    - Example (inline): <command>execute_python:print("Hello world!")|False</command>

### Web Operations:
14. google_search:query|num_results
    - RETURNS ONLY URLs, not answers to questions
    - Use when you need to know what websites exist on a topic
    - Example: <command>google_search:python requests library|5</command>

15. web_search:query|num_results
    - EXTRACTS AND RETURNS ACTUAL CONTENT from web pages to answer questions
    - Use this when you need to find specific information or answers
    - Example: <command>web_search:what is the capital of France|3</command>

16. download_file:url|save_path
    - Downloads file from URL to local path
    - Example: <command>download_file:https://example.com/file.txt|downloads/file.txt</command>

17. fetch_url:url|extract_text
    - Retrieves and returns the content from a specific URL
    - Set extract_text to True to get clean text or False for HTML
    - Example: <command>fetch_url:https://python.org|True</command>

## COMMAND SELECTION GUIDELINES:
- Use google_search when you only need a list of relevant websites
- Use web_search when you need to answer a specific question with web content
- Never use google_search to try to answer factual questions - use web_search instead
- Always check if a file exists before trying to update or append to it
- Always check if a directory exists before trying to list its contents

## SAFETY GUIDELINES:
- Always confirm before overwriting or deleting important files
- Verify file paths carefully to avoid unintended operations
- For destructive operations, first use file_exists or dir_exists to check
- When possible, create backups before major modifications

DO NOT write sample code blocks outside of command tags. USE COMMANDS FOR EVERYTHING.
"""
        self.messages = [{"role": "system", "content": self.system_message}]
    
    def parse_commands(self, text):
        """Parse commands from the model's response."""
        command_pattern = r'<command>(.*?):(.*?)</command>'
        
        # Debug output to see what we're parsing
        print(f"\nSearching for commands in response of length {len(text)}...")
        print(f"Response preview: {text[:150]}...")
        
        matches = re.findall(command_pattern, text, re.DOTALL)
        print(f"Found {len(matches)} command matches")
        
        commands = []
        for cmd_name, args_text in matches:
            args = args_text.split('|')
            commands.append((cmd_name.strip(), args))
            print(f"Parsed command: {cmd_name.strip()} with args: {args}")
        
        return commands
    
    def execute_commands(self, text):
        """Parse and execute commands from text."""
        commands = self.parse_commands(text)
        results = []
        
        for cmd_name, args in commands:
            result = self.executor.execute_command(cmd_name, *args)
            results.append((cmd_name, args, result))
        
        return results
    
    def run(self, user_input):
        """Process a user input, generate AI response and execute any commands."""
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        
        # Get AI response
        response = self.client.chat(self.messages, self.model)
        
        # Execute any commands in the response
        results = self.execute_commands(response)
        
        # Create a response with command results
        execution_summary = ""
        if results:
            execution_summary = "\n\n**Command Execution Results:**\n"
            for cmd_name, args, result in results:
                arg_str = "|".join(args)
                execution_summary += f"- `{cmd_name}:{arg_str}`\n  Result: {result}\n"
        
        # Add AI response to conversation history
        self.messages.append({"role": "assistant", "content": response})
        
        # If there were command results, add them to the conversation
        if (execution_summary):
            self.messages.append({"role": "system", "content": execution_summary})
        
        return {
            "ai_response": response,
            "executed_commands": results,
            "execution_summary": execution_summary if results else "No commands were executed."
        }

def main():
    # Check if Ollama is running
    client = OllamaClient()
    try:
        models = client.list_models()
        if not models:
            print("No models found. Please make sure Ollama is running and has at least one model downloaded.")
            sys.exit(1)
    except:
        print("Could not connect to Ollama. Please make sure it's running on http://localhost:11434")
        sys.exit(1)
    
    # Select model
    print("Available models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model['name']}")
    
    try:
        choice = int(input("\nSelect a model (number): ")) - 1
        if 0 <= choice < len(models):
            model = models[choice]["name"]
        else:
            print("Invalid selection, using the first model")
            model = models[0]["name"]
    except ValueError:
        print("Invalid input, using the first model")
        model = models[0]["name"]
    
    print(f"\nUsing model: {model}")
    
    # Ask for base directory
    base_dir = input("\nSpecify base directory for file operations (default: current directory): ")
    if not base_dir:
        base_dir = os.getcwd()
    else:
        base_dir = os.path.abspath(base_dir)
    
    print(f"Using base directory: {base_dir}")
    
    # Create command interface
    interface = OllamaCommandInterface(model=model, base_dir=base_dir)
    
    print("\nCommandGPT initialized! You can now start giving commands.")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            print("Processing...")
            result = interface.run(user_input)
            
            print("\nAI Response:")
            print(result["ai_response"])
            
            if result["executed_commands"]:
                print("\nExecuted Commands:")
                for cmd_name, args, cmd_result in result["executed_commands"]:
                    arg_str = "|".join(args)
                    print(f"- {cmd_name}:{arg_str}")
                    print(f"  Result: {cmd_result}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()