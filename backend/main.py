# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional

# Import our chatbot class
from chatbot import MarketingChatbot

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Marketing Chatbot API",
    description="API for handling marketing chatbot interactions"
)

# Configure CORS (Cross-Origin Resource Sharing)
# This is necessary to allow your frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot with your OpenAI API key
chatbot = MarketingChatbot(
    resources_dir="./company_resources/knowledge_base",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Define the structure for chat messages
class ChatMessage(BaseModel):
    message: str
    context: Optional[dict] = None  # Optional context data

# Define the chat endpoint
@app.post("/api/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """
    Main endpoint for chat interactions
    
    Args:
        chat_message: Contains the user's message and optional context
        
    Returns:
        JSON response containing the chatbot's reply
    """
    try:
        # Get response from chatbot
        response = chatbot.get_response(chat_message.message)
        
        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        # Log the error (you should set up proper logging)
        print(f"Error processing message: {str(e)}")
        
        # Return error to client
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your message"
        )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Simple health check endpoint to verify the API is running
    """
    return {
        "status": "healthy",
        "service": "Marketing Chatbot API"
    }

# For testing the API directly
@app.get("/")
async def root():
    """
    Root endpoint for API verification
    """
    return {
        "message": "Marketing Chatbot API is running",
        "status": "active"
    }

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Make server accessible from outside
        port=8000,       # Port to run the server on
        reload=True      # Enable auto-reload on code changes
    )
