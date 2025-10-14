#awsbedrock_rag_textsummarizer.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import boto3

# Load environment variables (your AWS credentials)
load_dotenv()

class AWSRAGBot:
    def __init__(self, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the RAG Bot with AWS Bedrock
        
        Available models:
        - anthropic.claude-3-haiku-20240307-v1:0 (fast & cheap!)
        - anthropic.claude-3-sonnet-20240229-v1:0 (smarter)
        - meta.llama3-70b-instruct-v1:0 (alternative)
        """
        print("🔧 Initializing AWS Bedrock connection...")
        
        # Set up AWS Bedrock client
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        # Initialize embeddings (Amazon Titan)
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v2:0"
        )
        
        # Initialize LLM (Claude)
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id=model_id,
            model_kwargs={
                "temperature": 0.3,
                "max_tokens": 2000
            }
        )
        
        self.vectorstore = None
        self.qa_chain = None
        print(" AWS Bedrock connected!")
        
    def load_documents(self, file_path):
        """
        Step 1: Load your documents (PDF or TXT)
        """
        print(f" Loading document: {file_path}")
        
        # Check file type and use appropriate loader
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Only PDF and TXT files supported!")
        
        documents = loader.load()
        print(f" Loaded {len(documents)} pages/sections")
        return documents
    
    def split_documents(self, documents):
        """
        Step 2: Split documents into smaller chunks
        """
        print("✂️  Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f" Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks):
        """
        Step 3: Create embeddings with Amazon Titan and store in FAISS
        """
        print(" Creating embeddings with Amazon Titan...")
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print(" Vector store created!")
        
    def setup_qa_chain(self):
        """
        Step 4: Set up the Question-Answering chain with Claude
        """
        print(" Setting up QA chain with Claude...")
        
        # Custom prompt template
        template = """You are a helpful AI assistant powered by Claude. Use the following context to answer the question.
        If you don't know the answer based on the context, say "I don't have enough information to answer that."
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print(" QA chain ready!")
    
    def process_document(self, file_path):
        """
        Complete pipeline: Load → Split → Embed → Setup QA
        """
        documents = self.load_documents(file_path)
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.setup_qa_chain()
        print("\n🎉 Bot is ready! You can now ask questions.\n")
    
    def ask(self, question):
        """
        Ask a question and get an answer from Claude!
        """
        if not self.qa_chain:
            return " Please load a document first using process_document()!"
        
        print(f"\n Question: {question}")
        result = self.qa_chain.invoke({"query": question})
        
        answer = result['result']
        sources = result['source_documents']
        
        print(f"\n💡 Answer: {answer}")
        print(f"\n📚 Found answer in {len(sources)} document sections")
        
        return answer
    
    def summarize(self):
        """
        Get a summary of the entire document using Claude
        """
        if not self.qa_chain:
            return " Please load a document first!"
        
        summary_question = "Please provide a comprehensive summary of the main topics and key points in this document."
        return self.ask(summary_question)
    
    def save_vectorstore(self, path="vectorstore_aws"):
        """
        Save the vector store
        """
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f" Vector store saved to {path}")
    
    def load_vectorstore(self, path="vectorstore_aws"):
        """
        Load a previously saved vector store
        """
        self.vectorstore = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.setup_qa_chain()
        print(f" Vector store loaded from {path}")


def test_aws_connection():
    """
    Test if AWS credentials are working
    """
    try:
        bedrock = boto3.client('bedrock', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
        models = bedrock.list_foundation_models()
        print(" AWS Connection successful!")
        print(f" Available models: {len(models['modelSummaries'])} found")
        return True
    except Exception as e:
        print(f" AWS Connection failed: {str(e)}")
        print("\n Troubleshooting:")
        print("1. Check your AWS credentials in .env file")
        print("2. Make sure you have Bedrock access in your AWS region")
        print("3. Verify IAM permissions for Bedrock")
        return False


def main():
    """
    Example usage with AWS Bedrock
    """
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Rest of your code...
    # Test AWS connection first
    print(" Testing AWS connection...\n")
    if not test_aws_connection():
        return
    
    print("\n" + "="*60 + "\n")
    
    # Create the bot with AWS Bedrock
    bot = AWSRAGBot()
    
    # Process a document
    file_path = "/Users/pranjal/Documents/AI-projects/rag-textsummarizer/sample_document.txt" # Change this to your file!
    
    if os.path.exists(file_path):
        bot.process_document(file_path)
        
        # Get a summary
        print("\n" + "="*50)
        print("GETTING SUMMARY WITH CLAUDE")
        print("="*50)
        bot.summarize()
        
        # Ask some questions
        print("\n" + "="*50)
        print("ASKING QUESTIONS")
        print("="*50)
        
        bot.ask("What are the main topics covered?")
        bot.ask("Tell me the summary of the document")
        
        # Save the vectorstore
        bot.save_vectorstore()
    else:
        print(f" File not found: {file_path}")
        print("Please create a PDF or TXT file and update the file_path variable!")


if __name__ == "__main__":
    main()