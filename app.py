import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import pdfplumber
import chromadb
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

# Attempt to import SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    st.error("The 'sentence_transformers' library is not installed. Falling back to TF-IDF for embeddings. To use SentenceTransformer, install it with: `pip install sentence-transformers`")
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load environment variables
load_dotenv()
# Custom CSS for UI
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
        self.collection = None
        self.client = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.model = None
        
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
        """Set up embedding system (ChromaDB with SentenceTransformers or TF-IDF fallback)"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.client = chromadb.Client()
                    
                    # Try to get or create collection
                    try:
                        self.collection = self.client.get_or_create_collection("pdf_documents")
                        st.success("Successfully accessed or created ChromaDB collection 'pdf_documents'.")
                    except Exception as collection_error:
                        st.warning(f"Failed to access or create ChromaDB collection: {str(collection_error)}. Falling back to TF-IDF.")
                        self.embedding_model = None
                        self.collection = None
                        self.client = None
                        # Initialize TF-IDF as fallback
                        self.vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
                        st.info("Using TF-IDF for document embeddings (fallback mode).")
                        return True
                    
                    return True
                except Exception as e:
                    st.warning(f"ChromaDB setup failed: {str(e)}. Falling back to TF-IDF.")
                    self.embedding_model = None
                    self.collection = None
                    self.client = None
                    # Initialize TF-IDF as fallback
                    self.vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
                    st.info("Using TF-IDF for document embeddings (fallback mode).")
                    return True
            else:
                # Fallback to TF-IDF if SentenceTransformers is not available
                self.vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
                st.info("Using TF-IDF for document embeddings (fallback mode).")
                return True
        except Exception as e:
            st.error(f"Failed to set up embeddings: {str(e)}")
            return False
    
    def extract_pdf_text_and_tables(self, pdf_file, pdf_name):
        """Extract text and tables from PDF file"""
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
                        
                        # Extract tables
                        page_tables = page.extract_tables()
                        for table_num, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                try:
                                    # Create DataFrame from table
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                    table_info = {
                                        'page': page_num + 1,
                                        'table_num': table_num + 1,
                                        'dataframe': df,
                                        'rows': len(table),
                                        'columns': len(table[0]) if table else 0,
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
                chunks = self.create_text_chunks(full_text)
                
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
    
    def create_text_chunks(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks for better context"""
        try:
            chunks = []
            words = text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append({
                        'text': chunk,
                        'chunk_id': len(chunks),
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
            all_chunks = []
            all_texts = []
            
            for pdf_name, chunks in st.session_state.all_pdf_chunks.items():
                all_chunks.extend(chunks)
                all_texts.extend([chunk['text'] for chunk in chunks])
            
            if not all_texts:
                st.warning("No text chunks available to create embeddings.")
                return False
            
            # Use ChromaDB with SentenceTransformers if available
            if self.collection and self.embedding_model:
                try:
                    # Generate embeddings
                    embeddings = self.embedding_model.encode(all_texts)
                    
                    # Process in batches to avoid memory issues
                    batch_size = 100
                    for i in range(0, len(all_texts), batch_size):
                        batch_texts = all_texts[i:i + batch_size]
                        batch_embeddings = embeddings[i:i + batch_size]
                        batch_ids = [f"chunk_{j}" for j in range(i, min(i + batch_size, len(all_texts)))]
                        
                        # Create metadata for each chunk
                        batch_metadatas = []
                        for j in range(i, min(i + batch_size, len(all_texts))):
                            # Find which PDF this chunk belongs to
                            pdf_name = "unknown"
                            current_idx = 0
                            for p_name, chunks in st.session_state.all_pdf_chunks.items():
                                if j < current_idx + len(chunks):
                                    pdf_name = p_name
                                    break
                                current_idx += len(chunks)
                            
                            batch_metadatas.append({
                                'pdf_name': pdf_name, 
                                'chunk_id': j
                            })
                        
                        # Upsert to ChromaDB
                        self.collection.upsert(
                            embeddings=batch_embeddings.tolist(),
                            documents=batch_texts,
                            ids=batch_ids,
                            metadatas=batch_metadatas
                        )
                    
                    st.success("Embeddings created successfully for all PDFs using ChromaDB.")
                    return True
                
                except Exception as chroma_error:
                    st.warning(f"ChromaDB failed: {str(chroma_error)}. Falling back to TF-IDF.")
                    # Fall back to TF-IDF
                    if self.vectorizer is None:
                        self.vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
                    self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                    st.success("TF-IDF embeddings created as fallback.")
                    return True
            
            else:
                # Use TF-IDF fallback
                if self.vectorizer is None:
                    self.vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
                
                self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                st.success("TF-IDF embeddings created successfully.")
                return True
                
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieve most relevant text chunks for a query"""
        try:
            # Try ChromaDB first
            if self.collection and self.embedding_model:
                try:
                    query_embedding = self.embedding_model.encode([query])
                    results = self.collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=top_k
                    )
                    return results['documents'][0] if results['documents'] else []
                except Exception as chroma_error:
                    st.warning(f"ChromaDB query failed: {str(chroma_error)}. Using TF-IDF fallback.")
            
            # Fallback to TF-IDF
            if self.vectorizer and self.tfidf_matrix is not None:
                try:
                    query_vec = self.vectorizer.transform([query])
                    similarities = cosine_similarity(query_vec, self.tfidf_matrix)
                    top_indices = similarities[0].argsort()[-top_k:][::-1]
                    
                    all_chunks = self.get_all_chunks()
                    return [all_chunks[idx]['text'] for idx in top_indices if idx < len(all_chunks)]
                except Exception as tfidf_error:
                    st.warning(f"TF-IDF query failed: {str(tfidf_error)}")
            
            st.warning("No valid vector store available for retrieval.")
            return []
            
        except Exception as e:
            st.error(f"Error retrieving relevant chunks: {str(e)}")
            return []
    
    def get_all_chunks(self):
        """Get all text chunks from all loaded PDFs"""
        try:
            all_chunks = []
            for chunks in st.session_state.all_pdf_chunks.values():
                all_chunks.extend(chunks)
            return all_chunks
        except Exception as e:
            st.error(f"Error retrieving all chunks: {str(e)}")
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
                return "Hi there! I'm MUST AI, your friendly assistant for Mbeya University of Science and Technology. Ready to help with your questions or dive into any documents you have—let me know what's up! 😊"
            
            elif 'how are you' in query_lower or 'inakuajee' in query_lower:
                return "I'm doing awesome, thanks for asking! I'm MUST AI, here to support you with all things MUST. What's on your mind today?"
            
            elif any(q in query_lower for q in ['who are you', 'what are you']):
                return "I'm MUST AI, a smart assistant built for Mbeya University of Science and Technology! I can answer questions, analyze documents, and more. How can I assist you today?"
            
            else:
                return f"Hey! I'm MUST AI, your go-to helper for Mbeya University of Science and Technology. I can answer general questions or analyze documents for you. What do you want to explore?"
        
        except Exception as e:
            st.error(f"Error generating general response: {str(e)}")
            return "Sorry, something went wrong. Please try again!"
    
    def generate_greeting(self, pdf_names):
        """Generate greeting message when PDFs are loaded"""
        try:
            pdf_list = ', '.join(pdf_names)
            return f"Hi! I'm MUST AI, and I've loaded your PDFs: {pdf_list}. I can analyze them together to answer your questions or provide insights. What would you like to know?"
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
            
            # System prompt
            system_prompt = f"""You are MUST AI, A helpful AI assistant that specializes in analyzing and answering questions about MUST (Mbeya University of Science and Technology).
            
            Use the provided context as the knowledge base to answer questions accurately and comprehensively.
            {viz_instruction}
            
            If the context doesn't contain relevant information, say so politely and offer general guidance about MUST or suggest what kind of information might be helpful."""
            
            # Context prompt
            context_prompt = f"""
            Context from PDF Documents:
            {relevant_context}
            
            User Question: {user_query}
            
            Please provide a helpful, accurate, and detailed response based on the PDF content. If the context doesn't directly answer the question, provide what relevant information you can and suggest related topics.
            """
            
            # Generate response
            response = self.model.generate_content(f"{system_prompt}\n\n{context_prompt}")
            return response.text
            
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

def auto_load_first_pdf():
    """Automatically load the first available PDF"""
    try:
        if (not st.session_state.auto_load_attempted and 
            not st.session_state.embeddings_ready and 
            st.session_state.available_pdfs):
            
            if initialize_ai_assistant():
                first_pdf = st.session_state.available_pdfs[0]
                pdf_file = pdf_manager.load_pdf(first_pdf['path'])
                
                if pdf_file:
                    ai_assistant = st.session_state.ai_assistant
                    pdf_text, pdf_chunks = ai_assistant.extract_pdf_text_and_tables(
                        pdf_file, first_pdf['name']
                    )
                    
                    if pdf_text:
                        st.session_state.all_pdf_texts[first_pdf['name']] = pdf_text
                        st.session_state.all_pdf_chunks[first_pdf['name']] = pdf_chunks
                        st.session_state.current_pdf_name = first_pdf['name']
                        
                        if ai_assistant.create_embeddings_for_all_pdfs():
                            st.session_state.embeddings_ready = True
                            st.session_state.vector_store = ai_assistant
                            
                            # Add greeting to chat
                            greeting = ai_assistant.generate_greeting([first_pdf['name']])
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
        st.error(f"Error auto-loading first PDF: {str(e)}")
        return False

# Initialize PDF Manager
pdf_folder_name = os.getenv('PDF_FOLDER_PATH', 'pdfs')
pdf_manager = LocalPDFManager(pdf_folder_name)

# Auto-initialize on first run
if not st.session_state.get('app_initialized', False):
    st.session_state.available_pdfs = pdf_manager.list_pdf_files()
    initialize_ai_assistant()
    st.session_state.app_initialized = True
    auto_load_first_pdf()

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
    
    # Chart History Section
    st.header("📊 Chart History")
    
    try:
        if st.session_state.chart_history:
            for i, chart_item in enumerate(st.session_state.chart_history):
                is_active = i == st.session_state.active_chart_index
                
                # Truncate title for display
                truncated_title = (chart_item['title'][:30] + '...' 
                                 if len(chart_item['title']) > 30 
                                 else chart_item['title'])
                
                if st.button(
                    f"📈 {truncated_title}",
                    key=f"chart_{i}",
                    help=f"Click to view this chart\nQuery: {chart_item['query'][:50]}..."
                ):
                    st.session_state.active_chart_index = i
                
                # Show timestamp and query
                truncated_query = (chart_item['query'][:40] + '...' 
                                 if len(chart_item['query']) > 40 
                                 else chart_item['query'])
                st.caption(f"🕒 {chart_item['timestamp'].strftime('%H:%M')} - {truncated_query}")
            
            # Display active chart
            if st.session_state.active_chart_index is not None:
                active_chart = st.session_state.chart_history[st.session_state.active_chart_index]
                st.subheader("Active Chart")
                st.plotly_chart(active_chart['figure'], use_container_width=True)
                
                if st.button("🗑️ Remove This Chart", type="secondary"):
                    st.session_state.chart_history.pop(st.session_state.active_chart_index)
                    st.session_state.active_chart_index = (None if not st.session_state.chart_history 
                                                         else 0)
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

# Chat History Display
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
    user_input = st.chat_input("Ask me anything about MUST or your PDFs...")
    
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

# Example Questions
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