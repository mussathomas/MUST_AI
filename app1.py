import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import io
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from pathlib import Path
import tempfile

# Attempt to import FAISS and SentenceTransformer
try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    st.warning("FAISS library not available. Using TF-IDF fallback for embeddings.")
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    st.warning("SentenceTransformers library not available. Using TF-IDF fallback for embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Custom CSS for UI (same as original)
st.markdown("""
<style>
/* Chat Container */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    max-width: 100%;
    padding: 0 1rem;
}
/* Base chat message styling */
.chat-message {
    padding: 0.75rem 1rem;
    border-radius: 18px;
    margin-bottom: 0.5rem;
    max-width: 80%;
    word-wrap: break-word;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    position: relative;
    clear: both;
}
/* User messages - aligned right */
.user-message {
    background: linear-gradient(135deg, #2196F3, #1976D2);
    color: white;
    margin-left: auto;
    margin-right: 0;
    border-bottom-right-radius: 4px;
    float: right;
    text-align: left;
}
.user-message::after {
    content: '';
    position: absolute;
    bottom: 0;
    right: -8px;
    width: 0;
    height: 0;
    border: 8px solid transparent;
    border-bottom-color: #1976D2;
    border-right: 0;
    border-bottom-right-radius: 16px;
}
/* Assistant messages - aligned left */
.assistant-message {
    background-color: #f5f5f5;
    color: #333;
    margin-left: 0;
    margin-right: auto;
    border-bottom-left-radius: 4px;
    border-left: 3px solid #4CAF50;
    float: left;
    text-align: left;
}
.assistant-message::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: -8px;
    width: 0;
    height: 0;
    border: 8px solid transparent;
    border-bottom-color: #f5f5f5;
    border-left: 0;
    border-bottom-left-radius: 16px;
}
/* Message container to handle floating */
.message-container {
    width: 100%;
    overflow: hidden;
    margin-bottom: 1rem;
}
/* Greeting message */
.greeting-message {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    border-left: 4px solid #4CAF50;
    font-style: italic;
    margin: 0 auto;
    max-width: 90%;
    text-align: center;
    float: none;
}
.greeting-message::after {
    display: none;
}
/* Typing indicator */
.typing-indicator {
    background-color: #f8f9fa;
    border-left: 4px solid #FFC107;
    animation: pulse 1.5s ease-in-out infinite alternate;
    margin-left: 0;
    margin-right: auto;
    float: left;
}
.typing-indicator::after {
    border-bottom-color: #f8f9fa;
}
.typing-text {
    font-family: 'Courier New', monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.typing-cursor {
    display: inline-block;
    background-color: #333;
    width: 2px;
    height: 1em;
    animation: blink 1s step-end infinite;
    margin-left: 2px;
}
/* Animations */
@keyframes pulse {
    from { border-left-color: #FFC107; }
    to { border-left-color: #FF9800; }
}
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}
/* Button styling */
.stButton > button {
    width: 100%;
}
/* History items */
.prompt-history-item, .chart-history-item {
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 12px;
    background-color: #f8f9fa;
    border: 1px solid #ddd;
    cursor: pointer;
    transition: all 0.3s ease;
    word-wrap: break-word;
}
.prompt-history-item:hover, .chart-history-item:hover {
    background-color: #e3f2fd;
    border-color: #2196F3;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.chart-history-item.active {
    background-color: #d1ecf1;
    border-color: #0dcaf0;
    box-shadow: 0 2px 4px rgba(13, 202, 240, 0.2);
}
/* Responsive Design */
@media (max-width: 768px) {
    .chat-message {
        max-width: 90%;
        padding: 0.6rem 0.8rem;
        font-size: 14px;
    }
    
    .chat-container {
        padding: 0 0.5rem;
        gap: 0.5rem;
    }
    
    .user-message::after, .assistant-message::after {
        border-width: 6px;
    }
    
    .prompt-history-item, .chart-history-item {
        padding: 0.6rem;
        margin: 0.3rem 0;
        font-size: 14px;
    }
}
@media (max-width: 480px) {
    .chat-message {
        max-width: 95%;
        padding: 0.5rem 0.7rem;
        font-size: 13px;
        border-radius: 14px;
    }
    
    .user-message {
        border-bottom-right-radius: 3px;
    }
    
    .assistant-message {
        border-bottom-left-radius: 3px;
    }
    
    .greeting-message {
        max-width: 98%;
        font-size: 13px;
    }
}
@media (min-width: 1200px) {
    .chat-message {
        max-width: 70%;
    }
    
    .chat-container {
        max-width: 1000px;
        margin: 0 auto;
    }
}
/* Dark mode support (if needed) */
@media (prefers-color-scheme: dark) {
    .assistant-message {
        background-color: #2d3748;
        color: #e2e8f0;
        border-left-color: #4CAF50;
    }
    
    .assistant-message::after {
        border-bottom-color: #2d3748;
    }
    
    .prompt-history-item, .chart-history-item {
        background-color: #2d3748;
        color: #e2e8f0;
        border-color: #4a5568;
    }
    
    .prompt-history-item:hover, .chart-history-item:hover {
        background-color: #4a5568;
        border-color: #2196F3;
    }
}
/* Clearfix for floating elements */
.clearfix::after {
    content: "";
    display: table;
    clear: both;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables with default values"""
    defaults = {
        'pdf_text': "",
        'pdf_chunks': [],
        'chat_history': [],
        'embeddings_ready': False,
        'vector_store': None,
        'gemini_model': None,
        'available_pdfs': [],
        'current_pdf_name': "",
        'greeting_shown': False,
        'app_initialized': False,
        'auto_load_attempted': False,
        'typing_response': "",
        'is_typing': False,
        'current_response_index': 0,
        'ai_assistant': None,
        'chart_history': [],
        'active_chart_index': None,
        'extracted_tables': {},
        'all_pdf_texts': {},
        'all_pdf_chunks': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Call initialize_session_state at the start
initialize_session_state()

class LocalPDFManager:
    """Manages local PDF files in a specified directory"""
    
    def __init__(self, pdf_folder="pdfs"):
        self.pdf_folder = Path(pdf_folder)
        self.ensure_folder_exists()
    
    def ensure_folder_exists(self):
        """Create the PDF folder if it doesn't exist"""
        try:
            self.pdf_folder.mkdir(exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create PDF folder {self.pdf_folder}: {str(e)}")
    
    def list_pdf_files(self):
        """List all PDF files in the folder with metadata"""
        try:
            pdf_files = []
            if self.pdf_folder.exists():
                for pdf_path in self.pdf_folder.glob("*.pdf"):
                    try:
                        file_stats = pdf_path.stat()
                        pdf_files.append({
                            'name': pdf_path.name,
                            'path': str(pdf_path),
                            'size': file_stats.st_size,
                            'modified': datetime.fromtimestamp(file_stats.st_mtime),
                            'size_mb': round(file_stats.st_size / (1024 * 1024), 2)
                        })
                    except Exception as e:
                        st.warning(f"Could not access file {pdf_path.name}: {str(e)}")
                        continue
                
                # Sort by modification date (newest first)
                pdf_files.sort(key=lambda x: x['modified'], reverse=True)
            
            return pdf_files
        except Exception as e:
            st.error(f"Error listing PDF files: {str(e)}")
            return []
    
    def load_pdf(self, pdf_path):
        """Load a PDF file and return as BytesIO object"""
        try:
            with open(pdf_path, 'rb') as file:
                return io.BytesIO(file.read())
        except Exception as e:
            st.error(f"Error loading PDF {pdf_path}: {str(e)}")
            return None
    
    def get_folder_info(self):
        """Get information about the PDF folder"""
        try:
            if not self.pdf_folder.exists():
                return {
                    'exists': False, 
                    'path': str(self.pdf_folder.absolute()), 
                    'pdf_count': 0, 
                    'total_size_mb': 0
                }
            
            pdf_files = list(self.pdf_folder.glob("*.pdf"))
            total_size = 0
            
            for f in pdf_files:
                try:
                    total_size += f.stat().st_size
                except:
                    continue
            
            return {
                'exists': True,
                'path': str(self.pdf_folder.absolute()),
                'pdf_count': len(pdf_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            st.error(f"Error getting folder info: {str(e)}")
            return {
                'exists': False, 
                'path': str(self.pdf_folder.absolute()), 
                'pdf_count': 0, 
                'total_size_mb': 0
            }

class PDFAIAssistant:
    """Main AI assistant class for PDF analysis and chat"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.embedding_model = None
        self.faiss_index = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.model = None
        self.chunk_metadata = []
        self.all_chunks = []  # Store all chunks for retrieval
        self.embedding_method = "none"  # Track which method is being used
        
        # Initialize components
        self.setup_gemini()
        self.setup_embeddings()
    
    def setup_gemini(self):
        """Initialize Google Gemini AI model"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            st.session_state.gemini_model = self.model
            st.success("Gemini AI model initialized successfully.")
            return True
        except Exception as e:
            st.error(f"Failed to set up Gemini AI: {str(e)}")
            return False
    
    def setup_embeddings(self):
        """Set up embedding system with proper fallback"""
        try:
            # Try FAISS with SentenceTransformers first
            if FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    # Test the model to make sure it works
                    test_embedding = self.embedding_model.encode(["test"])
                    embedding_dim = test_embedding.shape[1]
                    
                    # Initialize FAISS index with correct dimensions
                    self.faiss_index = faiss.IndexFlatL2(embedding_dim)
                    self.embedding_method = "faiss"
                    st.success("FAISS with SentenceTransformers initialized successfully.")
                    return True
                    
                except Exception as e:
                    st.warning(f"FAISS setup failed: {str(e)}. Falling back to TF-IDF.")
                    self.embedding_model = None
                    self.faiss_index = None
            
            # Fallback to TF-IDF
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            self.embedding_method = "tfidf"
            st.info("Using TF-IDF for document embeddings (fallback mode).")
            return True
            
        except Exception as e:
            st.error(f"Failed to set up embeddings: {str(e)}")
            return False
    
    def extract_pdf_text_and_tables(self, pdf_file, pdf_name):
        """Extract text and tables from PDF file with improved table handling"""
        try:
            full_text = ""
            tables_data = []
            
            # Create temporary file for PDF processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                with pdfplumber.open(tmp_file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        
                        # Extract tables with improved settings
                        page_tables = page.extract_tables({
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 4,
                            "intersection_tolerance": 3,
                        })
                        for table_num, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                try:
                                    # Clean table data: replace None with '', remove empty rows/columns
                                    cleaned_table = [[cell if cell is not None else '' for cell in row] for row in table]
                                    df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                                    df = df.replace(r'^\s*$', np.nan, regex=True)  # Replace empty strings with NaN
                                    df = df.dropna(how='all')  # Drop fully empty rows
                                    df = df.loc[:, df.notna().any()]  # Drop fully empty columns
                                    
                                    if not df.empty:
                                        table_info = {
                                            'page': page_num + 1,
                                            'table_num': table_num + 1,
                                            'dataframe': df,
                                            'rows': len(df),
                                            'columns': len(df.columns),
                                            'content': f"Table {table_num + 1} on Page {page_num + 1}:\n{df.to_string()}"
                                        }
                                        tables_data.append(table_info)
                                        full_text += f"\n--- Table {table_num + 1} on Page {page_num + 1} ---\n{df.to_string()}\n"
                                except Exception as table_error:
                                    st.warning(f"Could not process table {table_num + 1} on page {page_num + 1}: {str(table_error)}")
                                    continue
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                # Create text chunks
                chunks = self.create_text_chunks(full_text, pdf_name)
                
                # Store extracted tables
                if tables_data:
                    st.session_state.extracted_tables[pdf_name] = tables_data
                
                st.success(f"Extracted text and {len(tables_data)} tables from {pdf_name} successfully.")
                return full_text, chunks
                
            except Exception as pdf_error:
                # Clean up temporary file even if processing fails
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                raise pdf_error
                
        except Exception as e:
            st.error(f"Error extracting PDF {pdf_name}: {str(e)}")
            return "", []
    
    def create_text_chunks(self, text, pdf_name, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks for better context"""
        try:
            chunks = []
            words = text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': len(chunks),
                        'pdf_name': pdf_name,
                        'start_word': i,
                        'end_word': min(i + chunk_size, len(words))
                    })
            
            return chunks
        except Exception as e:
            st.error(f"Error creating text chunks: {str(e)}")
            return []
    
    def create_embeddings_for_all_pdfs(self):
        """Create embeddings for all loaded PDF chunks"""
        try:
            # Collect all chunks from all PDFs
            self.all_chunks = []
            all_texts = []
            self.chunk_metadata = []
            
            for pdf_name, chunks in st.session_state.all_pdf_chunks.items():
                for chunk in chunks:
                    self.all_chunks.append(chunk)
                    all_texts.append(chunk['text'])
                    self.chunk_metadata.append({
                        'pdf_name': pdf_name,
                        'chunk_id': chunk['chunk_id']
                    })
            
            if not all_texts:
                st.warning("No text chunks available to create embeddings.")
                return False
            
            # Use the appropriate embedding method
            if self.embedding_method == "faiss" and self.faiss_index is not None:
                try:
                    # Generate embeddings in batches to avoid memory issues
                    batch_size = 32
                    all_embeddings = []
                    
                    for i in range(0, len(all_texts), batch_size):
                        batch_texts = all_texts[i:i + batch_size]
                        batch_embeddings = self.embedding_model.encode(batch_texts)
                        all_embeddings.append(batch_embeddings)
                    
                    # Combine all embeddings
                    embeddings = np.vstack(all_embeddings).astype('float32')
                    
                    # Reset FAISS index and add embeddings
                    embedding_dim = embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatL2(embedding_dim)
                    self.faiss_index.add(embeddings)  # Fixed: Changed from add(embeddings) to add(embeddings)
                    
                    st.success(f"FAISS embeddings created for {len(all_texts)} chunks.")
                    return True
                    
                except Exception as faiss_error:
                    st.warning(f"FAISS failed: {str(faiss_error)}. Falling back to TF-IDF.")
                    self.embedding_method = "tfidf"
                    self.faiss_index = None
                    self.embedding_model = None
            
            # TF-IDF fallback
            if self.embedding_method == "tfidf":
                try:
                    if self.vectorizer is None:
                        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
                    
                    self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                    st.success(f"TF-IDF embeddings created for {len(all_texts)} chunks.")
                    return True
                    
                except Exception as tfidf_error:
                    st.error(f"TF-IDF also failed: {str(tfidf_error)}")
                    return False
            
            return False
                
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieve most relevant text chunks for a query"""
        try:
            if not self.all_chunks:
                st.warning("No chunks available for retrieval.")
                return []
            
            # Try FAISS first
            if self.embedding_method == "faiss" and self.faiss_index is not None:
                try:
                    query_embedding = self.embedding_model.encode([query]).astype('float32')
                    
                    # Ensure we don't request more results than available
                    k = min(top_k, self.faiss_index.ntotal)
                    if k <= 0:
                        return []
                    
                    distances, indices = self.faiss_index.search(query_embedding, k)
                    
                    relevant_chunks = []
                    for idx in indices[0]:
                        if 0 <= idx < len(self.all_chunks):
                            relevant_chunks.append(self.all_chunks[idx]['text'])
                    
                    return relevant_chunks
                    
                except Exception as faiss_error:
                    st.warning(f"FAISS retrieval failed: {str(faiss_error)}. Using TF-IDF fallback.")
                    self.embedding_method = "tfidf"
            
            # TF-IDF fallback
            if self.embedding_method == "tfidf" and self.tfidf_matrix is not None:
                try:
                    query_vec = self.vectorizer.transform([query])
                    similarities = cosine_similarity(query_vec, self.tfidf_matrix)
                    
                    # Get top indices
                    top_indices = similarities[0].argsort()[-top_k:][::-1]
                    
                    relevant_chunks = []
                    for idx in top_indices:
                        if 0 <= idx < len(self.all_chunks) and similarities[0][idx] > 0:
                            relevant_chunks.append(self.all_chunks[idx]['text'])
                    
                    return relevant_chunks
                    
                except Exception as tfidf_error:
                    st.warning(f"TF-IDF retrieval failed: {str(tfidf_error)}")
            
            # Last resort: return first few chunks
            return [chunk['text'] for chunk in self.all_chunks[:top_k]]
            
        except Exception as e:
            st.error(f"Error retrieving relevant chunks: {str(e)}")
            return []
    
    def is_general_greeting_or_chat(self, user_query):
        """Check if user query is a general greeting or casual chat"""
        try:
            greeting_patterns = [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                'how are you', 'whats up', "what's up", 'greetings', 'hola', 'welcome',
                'thanks', 'thank you', 'please', 'excuse me', 'sorry', 'mambo', 'hellow',
                'inakuajee', 'salaam'
            ]
            
            query_lower = user_query.lower().strip()
            return any(pattern in query_lower for pattern in greeting_patterns)
        except Exception as e:
            st.error(f"Error processing greeting check: {str(e)}")
            return False
    
    def generate_general_response(self, user_query):
        """Generate response for general greetings and casual chat"""
        try:
            query_lower = user_query.lower().strip()
            
            if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'mambo', 'hellow', 'salaam']):
                response = "Hi there! I'm MUST AI, your friendly assistant for Mbeya University of Science and Technology. Ready to help with your questions or dive into any documents you have—let me know what's up!"
            
            elif 'how are you' in query_lower or 'inakuajee' in query_lower:
                response = "I'm doing awesome, thanks for asking! I'm MUST AI, here to support you with all things MUST. What's on your mind today?"
            
            elif any(q in query_lower for q in ['who are you', 'what are you']):
                response = "I'm MUST AI, a smart assistant built for Mbeya University of Science and Technology! I can answer questions, analyze documents, and more. How can I assist you today?"
            
            else:
                response = f"Hey! I'm MUST AI, your go-to helper for Mbeya University of Science and Technology. I can answer general questions or analyze documents for you. What do you want to explore?"
            
            response += "\n\nDo you have any further questions?"
            return response
        
        except Exception as e:
            st.error(f"Error generating general response: {str(e)}")
            return "Sorry, something went wrong. Please try again!"
    
    def generate_greeting(self, pdf_names):
        """Generate greeting message when PDFs are loaded"""
        try:
            pdf_list = ', '.join(pdf_names)
            greeting = f"Hi! I'm MUST AI, I can answer questions about MUST. What would you like to know?"
            greeting += "\n\nDo you have any further questions?"
            return greeting
        except Exception as e:
            st.error(f"Error generating greeting: {str(e)}")
            return "Welcome to MUST AI! I couldn't load the PDF names, but I'm here to help. What's your question?"
    
    def generate_response(self, user_query, relevant_context="", enable_viz=False):
        """Generate AI response using Gemini model"""
        try:
            # Handle general greetings
            if not relevant_context and self.is_general_greeting_or_chat(user_query):
                return self.generate_general_response(user_query)
            
            # Prepare visualization instructions
            viz_instruction = """
            If asked to create charts or visualizations, provide detailed suggestions and sample code.
            """ if enable_viz else """
            Do not suggest visualizations or charts unless specifically requested by the user.
            Focus on providing clear, informative text responses.
            """
            
            # System prompt (modified to reference 'current knowledge')
            system_prompt = f"""You are MUST AI, A helpful AI assistant that specializes in analyzing and answering questions about MUST (Mbeya University of Science and Technology).
            
            Use your current knowledge as the knowledge base to answer questions accurately and comprehensively.
            {viz_instruction}
            
            If your current knowledge doesn't contain relevant information, say so politely and offer general guidance about MUST or suggest what kind of information might be helpful."""
            
            # Context prompt (modified to reference 'current knowledge')
            context_prompt = f"""
            Current Knowledge:
            {relevant_context}
            
            User Question: {user_query}
            
            Please provide a helpful, accurate, and detailed response based on your current knowledge. If the knowledge doesn't directly answer the question, provide what relevant information you can and suggest related topics.
            """
            
            # Generate response
            response = self.model.generate_content(f"{system_prompt}\n\n{context_prompt}")
            ai_response = response.text
            ai_response += "\n\nDo you have any further questions?"
            return ai_response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"Sorry, I ran into an issue: {str(e)}. Please try rephrasing your question!"
    
    def suggest_chart_from_response(self, response_text, pdf_context):
        """Suggest chart creation based on response content"""
        try:
            chart_prompt = f"""
            Based on this response: {response_text}
            And this context: {pdf_context[:1000]}
            
            If there's numerical data that could be visualized, extract it and format as JSON:
            {{"chart_type": "bar/pie/line/scatter", "data": {{"labels": [...], "values": [...]}}, "title": "chart title"}}
            
            If no suitable data, return: {{"chart_type": "none", "reason": "No numerical data found"}}
            """
            
            chart_response = self.model.generate_content(chart_prompt)
            return self.parse_chart_suggestion(chart_response.text)
        except Exception as e:
            st.error(f"Error suggesting chart: {str(e)}")
            return None
    
    def parse_chart_suggestion(self, response):
        """Parse chart suggestion from AI response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            st.error(f"Error parsing chart suggestion: {str(e)}")
            return None

def create_chart_from_data(chart_data, chart_title=None):
    
    """Create Plotly chart from data"""
    try:
        if not chart_data or chart_data.get('chart_type') == 'none':
            return None
        
        chart_type = chart_data.get('chart_type', 'bar')
        data = chart_data.get('data', {})
        title = chart_data.get('title', chart_title or "Data from PDF Analysis")
        
        # Validate data
        if 'labels' not in data or 'values' not in data:
            st.warning("Invalid chart data: Missing labels or values.")
            return None
        
        labels = data['labels']
        values = data['values']
        
        # Create chart based on type
        if chart_type == 'bar':
            fig = px.bar(x=labels, y=values, title=title)
        elif chart_type == 'pie':
            fig = px.pie(values=values, names=labels, title=title)
        elif chart_type == 'line':
            fig = px.line(x=labels, y=values, title=title)
        elif chart_type == 'scatter':
            fig = px.scatter(x=range(len(values)), y=values, hover_name=labels, title=title)
        else:
            fig = px.bar(x=labels, y=values, title=title)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def add_chart_to_history(chart_fig, title, query):
    """Add chart to history for later viewing"""
    try:
        chart_item = {
            'timestamp': datetime.now(),
            'title': title,
            'query': query,
            'figure': chart_fig
        }
        st.session_state.chart_history.append(chart_item)
        st.session_state.active_chart_index = len(st.session_state.chart_history) - 1
    except Exception as e:
        st.error(f"Error adding chart to history: {str(e)}")

def display_chat_message(message, index):
    """Display individual chat message with proper formatting"""
    try:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            message_class = "greeting-message" if message.get("is_greeting", False) else "assistant-message"
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <strong>MUST AI:</strong> <span class="typing-text">{message["content"]}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Display chart if available
            if "chart_data" in message:
                chart_fig = create_chart_from_data(
                    message["chart_data"], 
                    f"Chart from: {message.get('query', '')}"
                )
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)
                    add_chart_to_history(
                        chart_fig, 
                        message.get('query', 'AI Generated Chart'), 
                        message.get('query', '')
                    )
    
    except Exception as e:
        st.error(f"Error displaying chat message: {str(e)}")

def initialize_ai_assistant():
    """Initialize AI assistant if not already done"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and st.session_state.ai_assistant is None:
            st.session_state.ai_assistant = PDFAIAssistant(api_key)
            return True
        return st.session_state.ai_assistant is not None
    except Exception as e:
        st.error(f"Error initializing AI assistant: {str(e)}")
        return False

def auto_load_all_pdfs():
    """Automatically load all available PDFs from the designated folder"""
    try:
        if (not st.session_state.auto_load_attempted and 
            not st.session_state.embeddings_ready and 
            st.session_state.available_pdfs):
            
            if initialize_ai_assistant():
                ai_assistant = st.session_state.ai_assistant
                loaded_names = []
                with st.spinner("Automatically loading all PDFs..."):
                    for pdf_info in st.session_state.available_pdfs:
                        pdf_file = pdf_manager.load_pdf(pdf_info['path'])
                        if pdf_file:
                            pdf_text, pdf_chunks = ai_assistant.extract_pdf_text_and_tables(
                                pdf_file, pdf_info['name']
                            )
                            if pdf_text:
                                st.session_state.all_pdf_texts[pdf_info['name']] = pdf_text
                                st.session_state.all_pdf_chunks[pdf_info['name']] = pdf_chunks
                                loaded_names.append(pdf_info['name'])
                    
                    if loaded_names:
                        if ai_assistant.create_embeddings_for_all_pdfs():
                            st.session_state.embeddings_ready = True
                            st.session_state.vector_store = ai_assistant
                            
                            # Add greeting to chat
                            greeting = ai_assistant.generate_greeting(loaded_names)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": greeting,
                                "timestamp": datetime.now(),
                                "is_greeting": True
                            })
                            st.session_state.greeting_shown = True
                            return True
        
        st.session_state.auto_load_attempted = True
        return False
    except Exception as e:
        st.error(f"Error auto-loading PDFs: {str(e)}")
        return False

# Initialize PDF Manager
pdf_folder_name = os.getenv('PDF_FOLDER_PATH', 'pdfs')
pdf_manager = LocalPDFManager(pdf_folder_name)

# Auto-initialize on first run
if not st.session_state.get('app_initialized', False):
    st.session_state.available_pdfs = pdf_manager.list_pdf_files()
    initialize_ai_assistant()
    st.session_state.app_initialized = True
    auto_load_all_pdfs()

# Sidebar - Prompt and Chart History
with st.sidebar:
    st.header("📜 Prompt History")
    
    try:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    # Create a truncated version of the prompt for display
                    truncated_content = (message['content'][:30] + '...' 
                                       if len(message['content']) > 30 
                                       else message['content'])
                    
                    if st.button(f"🗨️ {truncated_content}", key=f"prompt_{i}"):
                        # Note: We can't directly set chat input, so we'll add it to history
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": message['content'],
                            "timestamp": datetime.now()
                        })
                        st.rerun()
                    
                    # Show timestamp
                    st.caption(f"🕒 {message['timestamp'].strftime('%H:%M')}")
        else:
            st.info("No prompts yet. Start chatting to build your history!")
    
    except Exception as e:
        st.error(f"Error displaying prompt history: {str(e)}")
    
    st.divider()
    
    # Chart History Section (enhanced to mimic Cloud AI style with expandable previews)
    st.header("📊 Chart History")
    
    try:
        if st.session_state.chart_history:
            for i, chart_item in enumerate(st.session_state.chart_history):
                with st.expander(f"📈 {chart_item['title'][:30] + '...' if len(chart_item['title']) > 30 else chart_item['title']}"):
                    st.plotly_chart(chart_item['figure'], use_container_width=True)
                    st.caption(f"🕒 {chart_item['timestamp'].strftime('%H:%M')} - Query: {chart_item['query'][:50]}...")
                    
                    if st.button("🗑️ Remove", key=f"remove_chart_{i}"):
                        st.session_state.chart_history.pop(i)
                        if st.session_state.active_chart_index == i:
                            st.session_state.active_chart_index = None
                        st.rerun()
        else:
            st.info("No charts generated yet. Ask for visualizations to see them here!")
    
    except Exception as e:
        st.error(f"Error displaying chart history: {str(e)}")
    
    st.divider()
    
    # Document Settings
    with st.expander("⚙️ Document Settings"):
        try:
            folder_info = pdf_manager.get_folder_info()
            st.info(f"**Folder:** `{folder_info['path']}`")
            
            if folder_info['exists']:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("PDF Files", folder_info['pdf_count'])
                with col2:
                    st.metric("Total Size", f"{folder_info['total_size_mb']} MB")
            
            # Refresh button
            if st.button("🔄 Refresh PDF List", type="primary"):
                st.session_state.available_pdfs = pdf_manager.list_pdf_files()
                st.rerun()
            
            # Available PDFs
            if st.session_state.available_pdfs:
                st.subheader("Available PDFs")
                selected_pdfs = []
                
                for pdf in st.session_state.available_pdfs:
                    is_loaded = pdf['name'] in st.session_state.all_pdf_texts
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        status = "✅ Loaded" if is_loaded else "📄 Available"
                        st.text(f"{pdf['name']}\n{status} - {pdf['size_mb']}MB")
                    
                    with col2:
                        if not is_loaded:
                            if st.button("📖", key=f"load_{pdf['name']}", help="Load this PDF"):
                                selected_pdfs.append(pdf)
                
                # Process selected PDFs
                if selected_pdfs and initialize_ai_assistant():
                    with st.spinner(f"Loading {len(selected_pdfs)} PDF(s)..."):
                        ai_assistant = st.session_state.ai_assistant
                        loaded_count = 0
                        
                        for pdf_info in selected_pdfs:
                            pdf_file = pdf_manager.load_pdf(pdf_info['path'])
                            if pdf_file:
                                pdf_text, pdf_chunks = ai_assistant.extract_pdf_text_and_tables(
                                    pdf_file, pdf_info['name']
                                )
                                if pdf_text:
                                    st.session_state.all_pdf_texts[pdf_info['name']] = pdf_text
                                    st.session_state.all_pdf_chunks[pdf_info['name']] = pdf_chunks
                                    loaded_count += 1
                        
                        if loaded_count > 0:
                            if ai_assistant.create_embeddings_for_all_pdfs():
                                st.session_state.embeddings_ready = True
                                st.session_state.vector_store = ai_assistant
                                
                                # Generate greeting
                                loaded_names = [pdf['name'] for pdf in selected_pdfs[:loaded_count]]
                                greeting = ai_assistant.generate_greeting(loaded_names)
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": greeting,
                                    "timestamp": datetime.now(),
                                    "is_greeting": True
                                })
                                
                                st.success(f"✅ Loaded {loaded_count} PDF(s) successfully!")
                                st.rerun()
            
            # Show loaded documents
            if st.session_state.all_pdf_texts:
                st.subheader("Loaded Documents")
                for pdf_name in st.session_state.all_pdf_texts.keys():
                    st.text(f"• {pdf_name}")
                
                if st.button("🗑️ Clear All Documents"):
                    # Reset all PDF-related session state
                    st.session_state.all_pdf_texts = {}
                    st.session_state.all_pdf_chunks = {}
                    st.session_state.extracted_tables = {}
                    st.session_state.chat_history = []
                    st.session_state.embeddings_ready = False
                    st.session_state.current_pdf_name = ""
                    st.session_state.greeting_shown = False
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error in document settings: {str(e)}")

# Main Chat Interface
st.title("🤖 MUST AI Assistant")
st.markdown("Chat with an AI that understands about MUST")

# Show info message if no PDFs available
if not st.session_state.available_pdfs and pdf_manager.get_folder_info()['exists']:
    st.info(f"📂 Add PDF files to the `{pdf_folder_name}` folder in your working directory to get started with document analysis.")

# Initial greeting display
if not st.session_state.greeting_shown and initialize_ai_assistant():
    try:
        initial_greeting = st.session_state.ai_assistant.generate_general_response("Hello")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": initial_greeting,
            "timestamp": datetime.now(),
            "is_general_response": True
        })
        st.session_state.greeting_shown = True
    except Exception as e:
        st.error(f"Error displaying initial greeting: {str(e)}")

# Example Questions (moved above chat history to show conversations below them)
if initialize_ai_assistant():
    try:
        st.subheader("💡 Example Questions")
        
        if st.session_state.embeddings_ready:
            # Questions for when documents are loaded
            example_col1, example_col2 = st.columns(2)
            
            with example_col1:
                if st.button("📊 Create visualization from data"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "Can you create charts from the data in my documents?",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
                
                if st.button("📋 Summarize all documents"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "Please summarize the key points from all loaded documents",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
            
            with example_col2:
                if st.button("🔍 Find specific information"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "What specific data or statistics are mentioned across all documents?",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
                
                if st.button("💡 Generate insights"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "What patterns or insights can you identify across all documents?",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
        
        else:
            # Questions for general MUST information
            example_col1, example_col2 = st.columns(2)
            
            with example_col1:
                if st.button("🏫 Tell me about MUST"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "Tell me about Mbeya University of Science and Technology",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
                
                if st.button("📚 Academic programs"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "What academic programs does MUST offer?",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
            
            with example_col2:
                if st.button("🎓 Student life"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "What is student life like at MUST?",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
                
                if st.button("📖 How to use documents"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "How can I upload and analyze multiple documents?",
                        "timestamp": datetime.now()
                    })
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error displaying example questions: {str(e)}")

# Chat History Display (now below example questions)
chat_container = st.container()
with chat_container:
    try:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                display_chat_message(message, i)
    except Exception as e:
        st.error(f"Error displaying chat history: {str(e)}")

# Chat Input
if initialize_ai_assistant() and os.getenv('GEMINI_API_KEY'):
    user_input = st.chat_input("Ask me anything about MUST...")
    
    if user_input:
        try:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Show typing indicator
            typing_placeholder = st.empty()
            typing_placeholder.markdown(f"""
            <div class="chat-message typing-indicator">
                <strong>MUST AI:</strong>
                <span class="typing-text">Thinking<span class="typing-cursor">|</span></span>
            </div>
            """, unsafe_allow_html=True)
            
            # Get AI assistant
            ai_assistant = st.session_state.ai_assistant
            
            # Initialize variables
            relevant_chunks = []
            context = ""
            is_general_response = False
            
            # Check if it's a general greeting/chat
            if ai_assistant.is_general_greeting_or_chat(user_input):
                ai_response = ai_assistant.generate_general_response(user_input)
                is_general_response = True
            else:
                # Retrieve relevant context if embeddings are ready
                if st.session_state.embeddings_ready and st.session_state.vector_store:
                    relevant_chunks = ai_assistant.retrieve_relevant_chunks(user_input)
                    context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
                
                # Check for visualization keywords
                viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'diagram', 'figure']
                enable_viz = any(keyword in user_input.lower() for keyword in viz_keywords)
                
                # Generate response
                ai_response = ai_assistant.generate_response(user_input, context, enable_viz)
                
                # Suggest chart if visualization was requested
                chart_data = None
                if enable_viz and context:
                    chart_data = ai_assistant.suggest_chart_from_response(ai_response, context)
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            # Prepare response data
            response_data = {
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now(),
                "is_new_response": True,
                "query": user_input
            }
            
            if is_general_response:
                response_data["is_general_response"] = True
            
            # Add chart data if available
            if 'chart_data' in locals() and chart_data and chart_data.get('chart_type') != 'none':
                response_data["chart_data"] = chart_data
            
            # Add response to history
            st.session_state.chat_history.append(response_data)
            st.rerun()
        
        except Exception as e:
            st.error(f"Error processing user input: {str(e)}")
            # Clear typing indicator on error
            if 'typing_placeholder' in locals():
                typing_placeholder.empty()

elif not os.getenv('GEMINI_API_KEY'):
    st.info("👈 Please add GEMINI_API_KEY to your .env file to get started.")
else:
    st.info("🤖 Setting up AI assistant...")

# Footer
st.divider()
st.markdown("**MUST AI Assistant** - Your intelligent companion for Mbeya University of Science and Technology")
st.markdown("Developed by Adomaster the ML Engineer")
st.markdown("Contact me +255 536 369 49 or mussathomas98@gmail.com")

# Debug information (only show if in development)
if os.getenv('DEBUG', 'False').lower() == 'true':
    with st.expander("🔧 Debug Information"):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("PDFs Loaded:", len(st.session_state.all_pdf_texts))
        st.write("Embeddings Ready:", st.session_state.embeddings_ready)
        st.write("Chat History Length:", len(st.session_state.chat_history))
        st.write("Available PDFs:", len(st.session_state.available_pdfs))