#aws_interactivebot.py
from awsbedrock_rag_textsummarizer import AWSRAGBot, test_aws_connection
import os

def print_header():
    """Print a cool header"""
    print("\n" + "="*60)
    print(" AWS BEDROCK RAG-POWERED Q&A BOT")
    print("="*60)
    print("Powered by: Amazon Bedrock + Claude AI")
    print("Ask questions about your documents!")
    print("Commands: 'quit' to exit, 'summary' for document summary")
    print("="*60 + "\n")

def select_model():
    """Let user choose which model to use"""
    print("\n Choose your AI model:")
    print("1. Claude 3 Haiku")
    print("2. Claude 3 Sonnet")
    print("3. Llama 3 70B")
    
    choice = input("\nEnter choice (1-3, or press Enter for default): ").strip()
    
    models = {
        "1": "anthropic.claude-3-haiku-20240307-v1:0",
        "2": "anthropic.claude-3-sonnet-20240229-v1:0",
        "3": "meta.llama3-70b-instruct-v1:0",
        "": "anthropic.claude-3-haiku-20240307-v1:0"  # default
    }
    
    model_id = models.get(choice, models["1"])
    model_name = {
        models["1"]: "Claude 3 Haiku",
        models["2"]: "Claude 3 Sonnet",
        models["3"]: "Llama 3 70B"
    }
    
    print(f"\n Selected: {model_name.get(model_id, 'Claude 3 Haiku')}")
    return model_id

def main():
    print_header()
    
    # Test AWS connection
    print(" Testing AWS Bedrock connection...\n")
    if not test_aws_connection():
        print("\n Cannot connect to AWS. Please check your credentials!")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Get file path from user
    file_path = input(" Enter the path to your document (PDF or TXT): ").strip()
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return
    
    # Select model
    model_id = select_model()
    
    # Create and load the bot
    print("\n Initializing bot with AWS Bedrock...\n")
    bot = AWSRAGBot(model_id=model_id)
    bot.process_document(file_path)
    
    # Show cost estimate
    print("\n Cost Estimate:")
    print("   Claude Haiku: ~$0.25 per 1M input tokens")
    print("   Your queries will cost fractions of a cent!")
    print("   Way cheaper than OpenAI! 🎉\n")
    
    # Interactive loop
    while True:
        print("\n" + "-"*60)
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n Goodbye! Thanks for using the AWS RAG Bot!")
            break
        
        if user_input.lower() == 'summary':
            print("\n Bot: Getting summary with Claude...\n")
            bot.summarize()
        else:
            print("\n Bot:")
            bot.ask(user_input)
    
    # Ask if they want to save
    save = input("\n Save the vector store for faster loading next time? (yes/no): ")
    if save.lower() in ['yes', 'y']:
        bot.save_vectorstore()
        print(" Saved! Next time, you can load it instantly.")

if __name__ == "__main__":
    main()