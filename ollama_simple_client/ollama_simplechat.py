import requests
import json
import sys

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        """Initialize the Ollama client with the API base URL."""
        self.base_url = base_url
        self.models_url = f"{base_url}/api/tags"
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"

    def list_models(self):
        """List all available models."""
        try:
            response = requests.get(self.models_url)
            response.raise_for_status()
            return response.json()["models"]
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []
            
    def generate(self, prompt, model="llama3", options=None):
        """Generate a completion for a prompt using the specified model."""
        if options is None:
            options = {}
            
        payload = {
            "model": model,
            "prompt": prompt,
            **options
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each line as a separate JSON object
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            chunk_text = chunk["response"] 
                            full_response += chunk_text
                            # Optional: Print streaming tokens in real-time
                            print(chunk_text, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
                        
            return full_response
        except requests.exceptions.RequestException as e:
            print(f"Error generating completion: {e}")
            return ""
            
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
                            # Optional: print streaming tokens in real-time
                            print(content_part, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            
            return full_content
        except requests.exceptions.RequestException as e:
            print(f"Error in chat: {e}")
            return ""

def interactive_chat(client, model="llama3"):
    """Run an interactive chat session with the model."""
    messages = []
    print(f"Starting chat with {model} (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        print("\nAI: ", end="", flush=True)
        try:
            response = client.chat(messages, model)
            #print(response)
            # Don't print the response again since it was already streamed
            print("")  # Just add a newline after the response
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\nError: {e}")
            
def main():
    client = OllamaClient()
    
    # Check if Ollama is running
    try:
        models = client.list_models()
        if not models:
            print("No models found. Please make sure Ollama is running and has at least one model downloaded.")
            sys.exit(1)
    except:
        print("Could not connect to Ollama. Please make sure it's running on http://localhost:11434")
        sys.exit(1)
    
    print("Available models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model['name']}")
        
    if len(models) > 1:
        try:
            choice = int(input("\nSelect a model (number): ")) - 1
            if 0 <= choice < len(models):
                selected_model = models[choice]["name"]
            else:
                print("Invalid selection, using the first model")
                selected_model = models[0]["name"]
        except ValueError:
            print("Invalid input, using the first model")
            selected_model = models[0]["name"]
    else:
        selected_model = models[0]["name"]
        
    print(f"\nUsing model: {selected_model}")
    
    # Example of direct completion
    print("\nExample generation:")
    prompt = "Explain quantum computing in one paragraph:"
    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)
    response = client.generate(prompt, selected_model)
    print("")  # Add a newline after the streamed response
    
    # Start interactive chat
    print("\nNow starting interactive chat...")
    interactive_chat(client, selected_model)
    
if __name__ == "__main__":
    main()