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
import os
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import asyncio
import functools
from cachetools import TTLCache

# Attempt to import FAISS and SentenceTransformer
try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Cache for embeddings and PDF chunks (TTL of 1 hour)
embedding_cache = TTLCache(maxsize=100, ttl=3600)
chunk_cache = TTLCache(maxsize=100, ttl=3600)

# Custom CSS for UI (unchanged)
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    max-width: 100%;
    padding: 0 1rem;
}
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
.message-container {
    width: 100%;
    overflow: hidden;
    margin-bottom: 1rem;
}
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
@keyframes pulse {
    from { border-left-color: #FFC107; }
    to { border-left-color: #FF9800; }
}
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}
.stButton > button {
    width: 100%;
}
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
.clearfix::after {
    content: "";
    display: table;
    clear: both;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
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

initialize_session_state()

class LocalPDFManager:
    def __init__(self, pdf_folder="pdfs"):
        self.pdf_folder = Path(pdf_folder)
        self.ensure_folder_exists()
    
    def ensure_folder_exists(self):
        try:
            self.pdf_folder.mkdir(exist_ok=True)
        except Exception:
            pass
    
    @st.cache_data(show_spinner=False)
    def list_pdf_files(_self):
        try:
            pdf_files = []
            if _self.pdf_folder.exists():
                for pdf_path in _self.pdf_folder.glob("*.pdf"):
                    file_stats = pdf_path.stat()
                    pdf_files.append({
                        'name': pdf_path.name,
                        'path': str(pdf_path),
                        'size': file_stats.st_size,
                        'modified': datetime.fromtimestamp(file_stats.st_mtime),
                        'size_mb': round(file_stats.st_size / (1024 * 1024), 2)
                    })
                pdf_files.sort(key=lambda x: x['modified'], reverse=True)
            return pdf_files
        except Exception:
            return []
    
    def load_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                return io.BytesIO(file.read())
        except Exception:
            return None
    
    @st.cache_data(show_spinner=False)
    def get_folder_info(_self):
        try:
            if not _self.pdf_folder.exists():
                return {'exists': False, 'path': str(_self.pdf_folder.absolute()), 'pdf_count': 0, 'total_size_mb': 0}
            pdf_files = list(_self.pdf_folder.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdf_files if f.exists())
            return {
                'exists': True,
                'path': str(_self.pdf_folder.absolute()),
                'pdf_count': len(pdf_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception:
            return {'exists': False, 'path': str(_self.pdf_folder.absolute()), 'pdf_count': 0, 'total_size_mb': 0}

class PDFAIAssistant:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embedding_model = None
        self.faiss_index = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.model = None
        self.chunk_metadata = []
        self.all_chunks = []
        self.embedding_method = "none"
        self.setup_gemini()
        self.setup_embeddings()
    
    def setup_gemini(self):
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            st.session_state.gemini_model = self.model
            return True
        except Exception:
            return False
    
    def setup_embeddings(self):
        if FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                test_embedding = self.embedding_model.encode(["test"])
                self.faiss_index = faiss.IndexFlatL2(test_embedding.shape[1])
                self.embedding_method = "faiss"
                return True
            except Exception:
                self.embedding_model = None
                self.faiss_index = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.embedding_method = "tfidf"
        return True
    
    async def extract_pdf_text_and_tables(self, pdf_file, pdf_name):
        try:
            full_text = ""
            tables_data = []
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            try:
                with pdfplumber.open(tmp_file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        page_tables = page.extract_tables({
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 4,
                            "intersection_tolerance": 3,
                        })
                        for table_num, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                cleaned_table = [[cell if cell is not None else '' for cell in row] for row in table]
                                df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                                df = df.replace(r'^\s*$', np.nan, regex=True).dropna(how='all')
                                df = df.loc[:, df.notna().any()]
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
                os.unlink(tmp_file_path)
                chunks = self.create_text_chunks(full_text, pdf_name)
                if tables_data:
                    st.session_state.extracted_tables[pdf_name] = tables_data
                return full_text, chunks
            except Exception:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                return "", []
        except Exception:
            return "", []
    
    @functools.lru_cache(maxsize=100)
    def create_text_chunks(self, text, pdf_name, chunk_size=1000, overlap=200):
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
        except Exception:
            return []
    
    async def create_embeddings_for_all_pdfs(self):
        try:
            self.all_chunks = []
            all_texts = []
            self.chunk_metadata = []
            for pdf_name, chunks in st.session_state.all_pdf_chunks.items():
                for chunk in chunks:
                    self.all_chunks.append(chunk)
                    all_texts.append(chunk['text'])
                    self.chunk_metadata.append({'pdf_name': pdf_name, 'chunk_id': chunk['chunk_id']})
            if not all_texts:
                return False
            cache_key = hash(tuple(all_texts))
            if cache_key in embedding_cache:
                self.faiss_index, self.tfidf_matrix = embedding_cache[cache_key]
                return True
            if self.embedding_method == "faiss" and self.faiss_index is not None:
                batch_size = 32
                all_embeddings = []
                for i in range(0, len(all_texts), batch_size):
                    batch_texts = all_texts[i:i + batch_size]
                    batch_embeddings = await asyncio.get_event_loop().run_in_executor(None, 
                        lambda: self.embedding_model.encode(batch_texts))
                    all_embeddings.append(batch_embeddings)
                embeddings = np.vstack(all_embeddings).astype('float32')
                embedding_dim = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(embedding_dim)
                self.faiss_index.add(embeddings)
                embedding_cache[cache_key] = (self.faiss_index, None)
                return True
            if self.embedding_method == "tfidf":
                if self.vectorizer is None:
                    self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
                self.tfidf_matrix = await asyncio.get_event_loop().run_in_executor(None, 
                    lambda: self.vectorizer.fit_transform(all_texts))
                embedding_cache[cache_key] = (None, self.tfidf_matrix)
                return True
            return False
        except Exception:
            return False
    
    async def retrieve_relevant_chunks(self, query, top_k=5):
        try:
            if not self.all_chunks:
                return []
            cache_key = hash(query + str(top_k))
            if cache_key in chunk_cache:
                return chunk_cache[cache_key]
            if self.embedding_method == "faiss" and self.faiss_index is not None:
                query_embedding = await asyncio.get_event_loop().run_in_executor(None, 
                    lambda: self.embedding_model.encode([query]).astype('float32'))
                k = min(top_k, self.faiss_index.ntotal)
                if k <= 0:
                    return []
                distances, indices = await asyncio.get_event_loop().run_in_executor(None, 
                    lambda: self.faiss_index.search(query_embedding, k))
                relevant_chunks = [self.all_chunks[idx]['text'] for idx in indices[0] if 0 <= idx < len(self.all_chunks)]
                chunk_cache[cache_key] = relevant_chunks
                return relevant_chunks
            if self.embedding_method == "tfidf" and self.tfidf_matrix is not None:
                query_vec = await asyncio.get_event_loop().run_in_executor(None, 
                    lambda: self.vectorizer.transform([query]))
                similarities = await asyncio.get_event_loop().run_in_executor(None, 
                    lambda: cosine_similarity(query_vec, self.tfidf_matrix))
                top_indices = similarities[0].argsort()[-top_k:][::-1]
                relevant_chunks = [self.all_chunks[idx]['text'] for idx in top_indices 
                                 if 0 <= idx < len(self.all_chunks) and similarities[0][idx] > 0]
                chunk_cache[cache_key] = relevant_chunks
                return relevant_chunks
            return [chunk['text'] for chunk in self.all_chunks[:top_k]]
        except Exception:
            return []
    
    def is_general_greeting_or_chat(self, user_query):
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                            'how are you', 'whats up', "what's up", 'greetings', 'hola', 'welcome',
                            'thanks', 'thank you', 'please', 'excuse me', 'sorry', 'mambo', 'hellow',
                            'inakuajee', 'salaam']
        return any(pattern in user_query.lower().strip() for pattern in greeting_patterns)
    
    def generate_general_response(self, user_query):
        query_lower = user_query.lower().strip()
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'mambo', 'hellow', 'salaam']):
            return "Hi! I'm MUST AI, ready to assist with Mbeya University of Science and Technology queries. What's up?"
        elif 'how are you' in query_lower or 'inakuajee' in query_lower:
            return "I'm doing great, thanks! What's on your mind today?"
        elif any(q in query_lower for q in ['who are you', 'what are you']):
            return "I'm MUST AI, your assistant for MUST-related questions and document analysis. How can I help?"
        return "Hey! I'm MUST AI, here to help with MUST info or documents. What do you want to explore?"
    
    def generate_greeting(self, pdf_names):
        pdf_list = ', '.join(pdf_names)
        return f"Hi! I'm MUST AI, ready to answer questions about MUST. Loaded PDFs: {pdf_list}. Ask away!"
    
    async def generate_response(self, user_query, relevant_context="", enable_viz=False):
        try:
            if not relevant_context and self.is_general_greeting_or_chat(user_query):
                return self.generate_general_response(user_query) + "\n\nAny more questions?"
            viz_instruction = "Provide visualization suggestions if requested." if enable_viz else "Focus on text responses."
            system_prompt = f"""You are MUST AI, specializing in Mbeya University of Science and Technology.
            Use available knowledge to answer accurately. {viz_instruction}
            If knowledge is limited, offer general MUST guidance."""
            context_prompt = f"Knowledge:\n{relevant_context}\n\nQuestion: {user_query}\n\nRespond helpfully."
            response = await asyncio.get_event_loop().run_in_executor(None, 
                lambda: self.model.generate_content(f"{system_prompt}\n\n{context_prompt}"))
            return response.text + "\n\nAny more questions?"
        except Exception:
            return "Sorry, I ran into an issue. Try rephrasing your question!"
    
    async def suggest_chart_from_response(self, response_text, pdf_context):
        try:
            chart_prompt = f"""
            Response: {response_text}
            Context: {pdf_context[:1000]}
            Extract numerical data for visualization as JSON:
            {{"chart_type": "bar/pie/line/scatter", "data": {{"labels": [...], "values": [...]}}, "title": "chart title"}}
            If none, return: {{"chart_type": "none", "reason": "No numerical data"}}
            """
            chart_response = await asyncio.get_event_loop().run_in_executor(None, 
                lambda: self.model.generate_content(chart_prompt))
            json_match = re.search(r'\{.*\}', chart_response.text, re.DOTALL)
            return json.loads(json_match.group()) if json_match else None
        except Exception:
            return None

def create_chart_from_data(chart_data, chart_title=None):
    try:
        if not chart_data or chart_data.get('chart_type') == 'none':
            return None
        chart_type = chart_data.get('chart_type', 'bar')
        data = chart_data.get('data', {})
        title = chart_data.get('title', chart_title or "Data Analysis")
        if 'labels' not in data or 'values' not in data:
            return None
        labels, values = data['labels'], data['values']
        if chart_type == 'bar':
            return px.bar(x=labels, y=values, title=title)
        elif chart_type == 'pie':
            return px.pie(values=values, names=labels, title=title)
        elif chart_type == 'line':
            return px.line(x=labels, y=values, title=title)
        elif chart_type == 'scatter':
            return px.scatter(x=range(len(values)), y=values, hover_name=labels, title=title)
        return px.bar(x=labels, y=values, title=title)
    except Exception:
        return None

def add_chart_to_history(chart_fig, title, query):
    if chart_fig:
        st.session_state.chart_history.append({
            'timestamp': datetime.now(),
            'title': title,
            'query': query,
            'figure': chart_fig
        })
        st.session_state.active_chart_index = len(st.session_state.chart_history) - 1

def display_chat_message(message, index):
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
        if "chart_data" in message:
            chart_fig = create_chart_from_data(message["chart_data"], f"Chart from: {message.get('query', '')}")
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
                add_chart_to_history(chart_fig, message.get('query', 'AI Chart'), message.get('query', ''))

async def initialize_ai_assistant():
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and st.session_state.ai_assistant is None:
        st.session_state.ai_assistant = PDFAIAssistant(api_key)
        return True
    return st.session_state.ai_assistant is not None

async def auto_load_all_pdfs():
    if (not st.session_state.auto_load_attempted and 
        not st.session_state.embeddings_ready and 
        st.session_state.available_pdfs):
        if await initialize_ai_assistant():
            ai_assistant = st.session_state.ai_assistant
            loaded_names = []
            tasks = []
            for pdf_info in st.session_state.available_pdfs:
                pdf_file = pdf_manager.load_pdf(pdf_info['path'])
                if pdf_file:
                    tasks.append(ai_assistant.extract_pdf_text_and_tables(pdf_file, pdf_info['name']))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for pdf_info, result in zip(st.session_state.available_pdfs, results):
                if isinstance(result, tuple) and result[0]:
                    st.session_state.all_pdf_texts[pdf_info['name']] = result[0]
                    st.session_state.all_pdf_chunks[pdf_info['name']] = result[1]
                    loaded_names.append(pdf_info['name'])
            if loaded_names and await ai_assistant.create_embeddings_for_all_pdfs():
                st.session_state.embeddings_ready = True
                st.session_state.vector_store = ai_assistant
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

# Initialize PDF Manager
pdf_folder_name = os.getenv('PDF_FOLDER_PATH', 'pdfs')
pdf_manager = LocalPDFManager(pdf_folder_name)

# Auto-initialize
if not st.session_state.get('app_initialized', False):
    st.session_state.available_pdfs = pdf_manager.list_pdf_files()
    asyncio.run(initialize_ai_assistant())
    st.session_state.app_initialized = True
    asyncio.run(auto_load_all_pdfs())

# Sidebar
with st.sidebar:
    st.header("📜 Prompt History")
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                truncated_content = (message['content'][:30] + '...' if len(message['content']) > 30 else message['content'])
                if st.button(f"🗨️ {truncated_content}", key=f"prompt_{i}"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": message['content'],
                        "timestamp": datetime.now()
                    })
                    st.rerun()
                st.caption(f"🕒 {message['timestamp'].strftime('%H:%M')}")
    else:
        st.info("No prompts yet.")
    
    st.divider()
    st.header("📊 Chart History")
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
        st.info("No charts yet.")
    
    st.divider()
    with st.expander("⚙️ Document Settings"):
        folder_info = pdf_manager.get_folder_info()
        st.info(f"**Folder:** `{folder_info['path']}`")
        if folder_info['exists']:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PDF Files", folder_info['pdf_count'])
            with col2:
                st.metric("Total Size", f"{folder_info['total_size_mb']} MB")
        if st.button("🔄 Refresh PDF List", type="primary"):
            st.session_state.available_pdfs = pdf_manager.list_pdf_files()
            st.rerun()
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
                    if not is_loaded and st.button("📖", key=f"load_{pdf['name']}", help="Load this PDF"):
                        selected_pdfs.append(pdf)
            if selected_pdfs and asyncio.run(initialize_ai_assistant()):
                ai_assistant = st.session_state.ai_assistant
                loaded_count = 0
                for pdf_info in selected_pdfs:
                    pdf_file = pdf_manager.load_pdf(pdf_info['path'])
                    if pdf_file:
                        pdf_text, pdf_chunks = asyncio.run(ai_assistant.extract_pdf_text_and_tables(pdf_file, pdf_info['name']))
                        if pdf_text:
                            st.session_state.all_pdf_texts[pdf_info['name']] = pdf_text
                            st.session_state.all_pdf_chunks[pdf_info['name']] = pdf_chunks
                            loaded_count += 1
                if loaded_count > 0 and await ai_assistant.create_embeddings_for_all_pdfs():
                    st.session_state.embeddings_ready = True
                    st.session_state.vector_store = ai_assistant
                    loaded_names = [pdf['name'] for pdf in selected_pdfs[:loaded_count]]
                    greeting = ai_assistant.generate_greeting(loaded_names)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": greeting,
                        "timestamp": datetime.now(),
                        "is_greeting": True
                    })
                    st.rerun()
        if st.session_state.all_pdf_texts:
            st.subheader("Loaded Documents")
            for pdf_name in st.session_state.all_pdf_texts.keys():
                st.text(f"• {pdf_name}")
            if st.button("🗑️ Clear All Documents"):
                st.session_state.all_pdf_texts = {}
                st.session_state.all_pdf_chunks = {}
                st.session_state.extracted_tables = {}
                st.session_state.chat_history = []
                st.session_state.embeddings_ready = False
                st.session_state.current_pdf_name = ""
                st.session_state.greeting_shown = False
                st.rerun()

# Main Chat Interface
st.title("🤖 MUST AI Assistant")
st.markdown("Chat with an AI that understands MUST")

if not st.session_state.available_pdfs and pdf_manager.get_folder_info()['exists']:
    st.info(f"📂 Add PDFs to `{pdf_folder_name}` folder to start analysis.")

if not st.session_state.greeting_shown and asyncio.run(initialize_ai_assistant()):
    initial_greeting = st.session_state.ai_assistant.generate_general_response("Hello")
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": initial_greeting,
        "timestamp": datetime.now(),
        "is_general_response": True
    })
    st.session_state.greeting_shown = True

# Example Questions
if asyncio.run(initialize_ai_assistant()):
    st.subheader("💡 Example Questions")
    example_col1, example_col2 = st.columns(2)
    if st.session_state.embeddings_ready:
        with example_col1:
            if st.button("📊 Create visualization"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Create charts from document data",
                    "timestamp": datetime.now()
                })
                st.rerun()
            if st.button("📋 Summarize documents"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Summarize key points from all documents",
                    "timestamp": datetime.now()
                })
                st.rerun()
        with example_col2:
            if st.button("🔍 Find information"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "What data or statistics are in the documents?",
                    "timestamp": datetime.now()
                })
                st.rerun()
            if st.button("💡 Generate insights"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "Identify patterns or insights across documents",
                    "timestamp": datetime.now()
                })
                st.rerun()
    else:
        with example_col1:
            if st.button("🏫 About MUST"):
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
            if st.button("📖 Use documents"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "How to upload and analyze documents?",
                    "timestamp": datetime.now()
                })
                st.rerun()

# Chat History
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        display_chat_message(message, i)

# Chat Input
if asyncio.run(initialize_ai_assistant()) and os.getenv('GEMINI_API_KEY'):
    user_input = st.chat_input("Ask about MUST...")
    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="chat-message typing-indicator">
            <strong>MUST AI:</strong> <span class="typing-text">Thinking<span class="typing-cursor">|</span></span>
        </div>
        """, unsafe_allow_html=True)
        ai_assistant = st.session_state.ai_assistant
        relevant_chunks = []
        context = ""
        is_general_response = False
        if ai_assistant.is_general_greeting_or_chat(user_input):
            ai_response = ai_assistant.generate_general_response(user_input)
            is_general_response = True
        else:
            if st.session_state.embeddings_ready and st.session_state.vector_store:
                relevant_chunks = asyncio.run(ai_assistant.retrieve_relevant_chunks(user_input))
                context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
            viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'diagram', 'figure']
            enable_viz = any(keyword in user_input.lower() for keyword in viz_keywords)
            ai_response = asyncio.run(ai_assistant.generate_response(user_input, context, enable_viz))
            chart_data = None
            if enable_viz and context:
                chart_data = asyncio.run(ai_assistant.suggest_chart_from_response(ai_response, context))
        typing_placeholder.empty()
        response_data = {
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now(),
            "is_new_response": True,
            "query": user_input
        }
        if is_general_response:
            response_data["is_general_response"] = True
        if 'chart_data' in locals() and chart_data and chart_data.get('chart_type') != 'none':
            response_data["chart_data"] = chart_data
        st.session_state.chat_history.append(response_data)
        st.rerun()
elif not os.getenv('GEMINI_API_KEY'):
    st.info("Please add GEMINI_API_KEY to your .env file.")
else:
    st.info("Setting up AI assistant...")

# Footer
st.divider()
st.markdown("**MUST AI Assistant** - Your intelligent companion for Mbeya University of Science and Technology")
st.markdown("Developed by Adomaster the ML Engineer")
st.markdown("Contact: +255 536 369 49 or mussathomas98@gmail.com")

# Debug Info (only in development)
if os.getenv('DEBUG', 'false').lower() == 'true':
    with st.expander("🔧 Debug Information"):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("PDFs Loaded:", len(st.session_state.all_pdf_texts))
        st.write("Embeddings Ready:", st.session_state.embeddings_ready)
        st.write("Chat History Length:", len(st.session_state.chat_history))
        st.write("Available PDFs:", len(st.session_state.available_pdfs))