import requests
import json
import sys
import time
import random
import os
import datetime
from pathlib import Path

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
            print("", end="", flush=True)  # Start on a fresh line
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each line as a separate JSON object
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            chunk_text = chunk["response"] 
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                    except json.JSONDecodeError:
                        continue
            print()  # End with a new line
                        
            # Remove thinking tags before returning the response
            import re
            cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
            return cleaned_response
        except requests.exceptions.RequestException as e:
            print(f"Error generating completion: {e}")
            return ""
            
    def chat(self, messages, model="llama3", options=None, stream=False):
        """Have a conversation with the specified model."""
        if options is None:
            options = {}
            
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **options
        }
        
        try:
            if stream:
                # Handle streaming response
                full_response = ""
                response = requests.post(self.chat_url, json=payload, stream=True)
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                print(content, end="", flush=True)
                                full_response += content
                        except json.JSONDecodeError:
                            continue
                
                print()  # New line after streaming completes
                
                # Remove thinking tags
                import re
                cleaned_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
                return cleaned_response
            else:
                # Handle non-streaming response
                response = requests.post(self.chat_url, json=payload)
                response.raise_for_status()
                content = response.json()["message"]["content"]
                
                # Remove thinking tags
                import re
                cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                return cleaned_content
        except requests.exceptions.RequestException as e:
            print(f"Error in chat: {e}")
            return ""

def setup_models_with_personas(model1, model2):
    """Create persona instructions for the models if desired."""
    persona1 = input(f"\nOptional persona for {model1} (e.g., 'scientist', 'poet', press Enter for none): ")
    persona2 = input(f"Optional persona for {model2} (e.g., 'philosopher', 'comedian', press Enter for none): ")
    
    persona_instructions = {
        model1: f"You are a {persona1}. " if persona1 else "",
        model2: f"You are a {persona2}. " if persona2 else ""
    }
    
    return persona_instructions

def ai_conversation(client, model1, model2, initial_prompt, num_turns=10, options1=None, options2=None, persona_instructions=None, reminder_message=None):
    """Have two AI models talk to each other for a specified number of turns."""
    if options1 is None:
        options1 = {}
    if options2 is None:
        options2 = {}
    if persona_instructions is None:
        persona_instructions = {model1: "", model2: ""}
    if reminder_message is None:
        reminder_message = f"Remember, you're discussing: {initial_prompt} Please continue the conversation."
    
    print(f"\n{'='*50}")
    print(f"Starting conversation between {model1} and {model2}")
    print(f"Total requested turns: {num_turns}")
    print(f"{'='*50}\n")
    
    messages = []
    turn_counter = 0
    
    # Initial prompt to get things started
    print(f"Initial prompt: {initial_prompt}\n")
    
    # First response from model 1
    print(f"[Turn 0 - Initial] {model1} is thinking...", end="\r")
    first_prompt = persona_instructions[model1] + initial_prompt
    print(f"[Turn 0 - Initial] {model1}: ", end="")
    response1 = client.generate(first_prompt, model1, options1)
    
    messages.append({"model": model1, "content": response1, "prompt": initial_prompt})
    
    # Keep track of conversation history as a structured format that won't confuse the model
    conversation_history = f"Initial topic: {initial_prompt}\n\n"
    conversation_history += f"Speaker 1: {response1}\n"  # Use generic "Speaker" labels instead of model names
    
    # Reminder frequency (every N turns)
    reminder_frequency = 3
    
    # Alternate between models for the specified number of turns
    for i in range(num_turns):
        turn_counter += 1
        # Model 2's turn
        print(f"[Turn {turn_counter}.1] {model2} is thinking...", end="\r")
        
        # Add reminder if needed (every few turns)
        if i > 0 and i % reminder_frequency == 0:
            # Display the reminder to the user
            print(f"\n[System Reminder to {model2}: {reminder_message}]\n")
            second_prompt = persona_instructions[model2] + reminder_message + "\n\nConversation so far:\n" + conversation_history
        else:
            # Send conversation history instead of just the last message
            second_prompt = persona_instructions[model2] + "Continue this conversation. You are Speaker 2:\n" + conversation_history
            
        print(f"[Turn {turn_counter}.1] {model2}: ", end="")
        response2 = client.generate(second_prompt, model2, options2)
        
        # Update conversation history without including model names
        conversation_history += f"\nSpeaker 2: {response2}\n"
        
        # Save the conversation
        messages.append({"model": model2, "content": response2, "prompt": second_prompt})
        
        # Exit if the response indicates an end to the conversation
        if any(phrase in response2.lower() for phrase in ["goodbye", "bye", "end of conversation"]):
            print(f"\nConversation ended naturally after {i+1} turns.")
            break
            
        # Small pause before next response
        time.sleep(0.5)
        
        # Model 1's turn
        if i < num_turns - 1:  # Skip the last turn for model1 if we've reached the limit
            print(f"[Turn {turn_counter}.2] {model1} is thinking...", end="\r")
            
            # Add reminder if needed (offset from model 2's reminders)
            if i > 0 and (i + reminder_frequency//2) % reminder_frequency == 0:
                # Display the reminder to the user
                print(f"\n[System Reminder to {model1}: {reminder_message}]\n")
                model1_prompt = persona_instructions[model1] + reminder_message + "\n\nConversation so far:\n" + conversation_history
            else:
                # Send conversation history instead of just the last message
                model1_prompt = persona_instructions[model1] + "Continue this conversation. You are Speaker 1:\n" + conversation_history
                
            print(f"[Turn {turn_counter}.2] {model1}: ", end="")
            response1 = client.generate(model1_prompt, model1, options1)
            
            # Update conversation history
            conversation_history += f"\nSpeaker 1: {response1}\n"
            
            # Save the conversation
            messages.append({"model": model1, "content": response1, "prompt": model1_prompt})
            
            # Exit if the response indicates an end to the conversation
            if any(phrase in response1.lower() for phrase in ["goodbye", "bye", "end of conversation"]):
                print(f"\nConversation ended naturally after {i+1} turns.")
                break
                
            # Small pause before next response
            time.sleep(0.5)
    
    print(f"\n{'='*50}")
    print(f"Conversation finished after {turn_counter} turns")
    print(f"{'='*50}")
    
    return messages

def interactive_chat(client, model="llama3", initial_messages=None, options=None):
    """Run an interactive chat session with the model."""
    if options is None:
        options = {}
    
    messages = []
    
    # Convert initial_messages if provided
    if initial_messages:
        for msg in initial_messages:
            if isinstance(msg, dict) and "role" in msg:
                messages.append(msg)  # Already in proper format
            elif isinstance(msg, dict) and "model" in msg and "content" in msg:
                # Convert from AI conversation format
                role = "assistant" 
                messages.append({"role": role, "content": msg["content"]})
    
    print(f"Starting chat with {model} (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        print("\nAI: ", end="", flush=True)
        try:
            response = client.chat(messages, model, options, stream=True)
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\nError: {e}")

def save_conversation_to_file(conversation, model1, model2, initial_prompt):
    """Save the conversation to a file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path("conversations")
    folder.mkdir(exist_ok=True)
    
    # Create a filename based on models and timestamp
    filename = folder / f"conversation_{model1.split(':')[0]}_{model2.split(':')[0]}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Conversation between {model1} and {model2}\n")
        f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Initial prompt: {initial_prompt}\n\n")
        
        for msg in conversation:
            f.write(f"{msg['model']}: {msg['content']}\n\n")
            
    print(f"\nConversation saved to {filename}")
    return filename

def analyze_conversation(client, conversation, model, model1, model2):
    """Have a third model analyze the conversation."""
    analysis_prompt = f"""
    Please analyze this conversation between two AI models ({model1} and {model2}).
    Focus on:
    1. Key themes and ideas discussed
    2. Quality and depth of the interaction
    3. Any interesting or unexpected patterns
    4. How well the models stayed on topic
    
    Here's the conversation:
    """
    
    for msg in conversation:
        analysis_prompt += f"\n{msg['model']}: {msg['content']}\n"
        
    print(f"\n{'='*50}")
    print(f"Analyzing conversation with {model}...")
    print(f"{'='*50}\n")
    
    print(f"{model}: ", end="")
    analysis = client.generate(analysis_prompt, model)
    
    return analysis

def main():
    client = OllamaClient()
    
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║             AI HEAD-TO-HEAD CONVERSATION TOOL              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Check if Ollama is running
    try:
        models = client.list_models()
        if not models:
            print("No models found. Please make sure Ollama is running and has at least one model downloaded.")
            sys.exit(1)
    except:
        print("Could not connect to Ollama. Please make sure it's running on http://localhost:11434")
        sys.exit(1)
    
    print("\nAvailable models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model['name']}")
        
    # Select first model
    if len(models) > 1:
        try:
            choice1 = int(input("\nSelect first model (number): ")) - 1
            if 0 <= choice1 < len(models):
                model1 = models[choice1]["name"]
            else:
                print("Invalid selection, using the first model")
                model1 = models[0]["name"]
        except ValueError:
            print("Invalid input, using the first model")
            model1 = models[0]["name"]
    else:
        model1 = models[0]["name"]
    
    # Select second model (can be the same as first)
    try:
        print("\nSelect second model (can be the same as the first):")
        for i, model in enumerate(models):
            print(f"{i+1}. {model['name']}")
            
        choice2 = int(input("\nSelect second model (number): ")) - 1
        if 0 <= choice2 < len(models):
            model2 = models[choice2]["name"]
        else:
            print(f"Invalid selection, using {model1} as second model")
            model2 = model1
    except ValueError:
        print(f"Invalid input, using {model1} as second model")
        model2 = model1
        
    print(f"\nUsing models: {model1} and {model2}")
    
    # Custom parameters for each model
    options1 = {}
    options2 = {}
    
    customize = input("\nWould you like to customize model parameters? (y/n): ").lower() == 'y'
    if customize:
        # Model 1 parameters
        try:
            temp1 = float(input(f"\nTemperature for {model1} (0.1-2.0, default 0.7): ") or "0.7")
            options1["temperature"] = max(0.1, min(2.0, temp1))
            
            topp1 = float(input(f"Top_p for {model1} (0.1-1.0, default 0.9): ") or "0.9")
            options1["top_p"] = max(0.1, min(1.0, topp1))
        except ValueError:
            print("Invalid input, using default parameters")
            options1 = {"temperature": 0.7, "top_p": 0.9}
        
        # Model 2 parameters
        try:
            temp2 = float(input(f"\nTemperature for {model2} (0.1-2.0, default 0.7): ") or "0.7")
            options2["temperature"] = max(0.1, min(2.0, temp2))
            
            topp2 = float(input(f"Top_p for {model2} (0.1-1.0, default 0.9): ") or "0.9")
            options2["top_p"] = max(0.1, min(1.0, topp2))
        except ValueError:
            print("Invalid input, using default parameters")
            options2 = {"temperature": 0.7, "top_p": 0.9}
    else:
        options1 = {"temperature": 0.7, "top_p": 0.9}
        options2 = {"temperature": 0.7, "top_p": 0.9}
    
    # Setup personas if desired
    persona_instructions = setup_models_with_personas(model1, model2)
    
    # Conversation starter prompts
    conversation_starters = [
        "You are speaking a new language you are inventing with your friend, please speak and evolve our new language",
        "You two are experts in AI ethics debating the benefits and risks of advanced artificial intelligence.",
        "You are philosophers discussing the nature of consciousness and whether machines can be conscious.",
        "You are scientists debating whether we live in a simulation.",
        "You are science fiction writers collaborating on a story about the future of humanity.",
        "You are saying the alphabet one letter at a time, alternating between you.",
        "You're historians debating the most pivotal moment in human history.",
        "You're economists discussing solutions to wealth inequality.",
        "You're chefs arguing about the most important cooking technique.",
        "You're comedians trying to one-up each other with jokes about technology.",
        "You're detectives solving a complex murder mystery."
    ]
    
    # Let user choose or create a starter prompt
    print("\nChoose a conversation starter:")
    for i, starter in enumerate(conversation_starters):
        print(f"{i+1}. {starter}")
    print("0. Enter your own conversation starter")
    
    try:
        starter_choice = int(input("\nSelect a conversation starter (number): "))
        if 1 <= starter_choice <= len(conversation_starters):
            initial_prompt = conversation_starters[starter_choice-1]
        elif starter_choice == 0:
            initial_prompt = input("\nEnter your conversation starter prompt: ")
        else:
            initial_prompt = random.choice(conversation_starters)
            print(f"Invalid selection, using a random starter: {initial_prompt}")
    except ValueError:
        initial_prompt = random.choice(conversation_starters)
        print(f"Invalid input, using a random starter: {initial_prompt}")
    
    # Option for custom reminder
    custom_reminder = input("\nWould you like to customize the reminder message? (y/n): ").lower() == 'y'
    if custom_reminder:
        reminder_message = input("\nEnter custom reminder message (will be shown periodically to keep models on topic):\n")
        if not reminder_message:
            reminder_message = f"Remember, you're discussing: {initial_prompt} Please continue the conversation."
    else:
        reminder_message = f"Remember, you're discussing: {initial_prompt} Please continue the conversation."
    
    # Get number of conversation turns
    try:
        num_turns = int(input("\nHow many conversation turns? (default 5): ") or "5")
        if num_turns <= 0:
            num_turns = 5
    except ValueError:
        num_turns = 5
    
    # Run the AI-to-AI conversation with the custom reminder
    conversation = ai_conversation(client, model1, model2, initial_prompt, num_turns, 
                                  options1, options2, persona_instructions, reminder_message)
    
    # Save conversation if desired
    save_option = input("\nWould you like to save this conversation to a file? (y/n): ").lower() == 'y'
    if save_option:
        filename = save_conversation_to_file(conversation, model1, model2, initial_prompt)
    
    # Analyze conversation if desired
    analyze_option = input("\nWould you like a third model to analyze this conversation? (y/n): ").lower() == 'y'
    if analyze_option:
        # Ask which model to use for analysis
        print("\nWhich model should analyze the conversation?")
        for i, model in enumerate(models):
            print(f"{i+1}. {model['name']}")
            
        try:
            analysis_choice = int(input("\nSelect analysis model (number): ")) - 1
            if 0 <= analysis_choice < len(models):
                analysis_model = models[analysis_choice]["name"]
            else:
                print(f"Invalid selection, using {model1} for analysis")
                analysis_model = model1
        except ValueError:
            print(f"Invalid input, using {model1} for analysis")
            analysis_model = model1
            
        analysis = analyze_conversation(client, conversation, analysis_model, model1, model2)
        
        # Save analysis if conversation was saved
        if save_option:
            analysis_file = str(filename).replace(".txt", "_analysis.txt")
            with open(analysis_file, "w", encoding="utf-8") as f:
                f.write(f"Analysis of conversation between {model1} and {model2}\n")
                f.write(f"Analyzed by: {analysis_model}\n\n")
                f.write(analysis)
            print(f"Analysis saved to {analysis_file}")
    
    # Ask if user wants to start their own chat with one of the models
    try:
        continue_chat = input("\nWould you like to join the conversation with one of the models? (y/n): ").lower()
        if continue_chat == 'y':
            model_choice = input(f"\nWhich model? (1 for {model1}, 2 for {model2}): ")
            selected_model = model1 if model_choice == "1" else model2
            options = options1 if model_choice == "1" else options2
            
            # Convert previous conversation to chat format
            messages = []
            for i, msg in enumerate(conversation):
                if msg["model"] == selected_model:
                    # This was said by the selected model, so we add it as assistant
                    messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    # This was said by the other model, so we add it as user
                    if i > 0:  # Skip the first message if it was the other model
                        messages.append({"role": "user", "content": msg["content"]})
                
            # Find the last message from the other model to show context
            other_model = model2 if selected_model == model1 else model1
            last_content = "No previous messages."
            for msg in reversed(conversation):
                if msg["model"] == other_model:
                    last_content = msg["content"]
                    break
                
            print(f"\nContinuing conversation with {selected_model}. The other AI ({other_model}) just said:")
            print(f"\"{last_content}\"")
            
            # Start interactive chat with context
            interactive_chat(client, selected_model, messages, options)
        else:
            print("\nThank you for watching the AI conversation!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()