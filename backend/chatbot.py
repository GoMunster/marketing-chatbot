# chatbot.py

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

class MarketingChatbot:
    def __init__(self, resources_dir: str, openai_api_key: str):
        """
        Initialize the marketing chatbot
        
        Args:
            resources_dir (str): Path to directory containing company documents
            openai_api_key (str): OpenAI API key
        """
        self.resources_dir = resources_dir
        self.openai_api_key = openai_api_key
        
        # Initialize components
        self._load_documents()
        self._create_vector_store()
        self._setup_chat_chain()
    
    def _load_documents(self):
        """Load and process all company documents"""
        # Create a loader for all txt files in the resources directory
        loader = DirectoryLoader(
            self.resources_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        # Load all documents
        documents = loader.load()
        
        # Split documents into smaller chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Characters per chunk
            chunk_overlap=200  # Overlap between chunks to maintain context
        )
        
        self.documents = text_splitter.split_documents(documents)
        print(f"Loaded {len(self.documents)} document chunks")
    
    def _create_vector_store(self):
        """Create vector store from processed documents"""
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        # Create vector store from documents
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=embeddings
        )
    
    def _setup_chat_chain(self):
        """Set up the conversational chain with memory"""
        # Initialize the language model
        llm = ChatOpenAI(
            temperature=0.7,  # Controls response creativity (0.0 - 1.0)
            openai_api_key=self.openai_api_key
        )
        
        # Set up conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the conversation chain
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # Return top 3 most relevant chunks
            ),
            memory=memory,
            verbose=True
        )
    
    def get_response(self, query: str) -> str:
        """
        Get chatbot response for user query
        
        Args:
            query (str): User's question or message
            
        Returns:
            str: Chatbot's response
        """
        # Add custom instructions to the query
        enhanced_query = f"""Please answer the following question based on our company's information. 
        If you're not sure about something, be honest about it. Question: {query}"""
        
        # Get response from the chat chain
        response = self.chat_chain({"question": enhanced_query})
        return response["answer"]
    
    def add_document(self, file_path: str):
        """
        Add a new document to the knowledge base
        
        Args:
            file_path (str): Path to the new document
        """
        # Load new document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        new_docs = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(new_docs)
        print(f"Added new document: {file_path}")

# Example usage
if __name__ == "__main__":
    # This code only runs if you execute chatbot.py directly
    openai_key = os.getenv("OPENAI_API_KEY")
    chatbot = MarketingChatbot(
        resources_dir="./company_resources/knowledge_base",
        openai_api_key=openai_key
    )
    
    # Test the chatbot
    question = "What social media services do you offer?"
    response = chatbot.get_response(question)
    print(f"Q: {question}")
    print(f"A: {response}")
