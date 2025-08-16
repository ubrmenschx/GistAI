import os
import re
import validators
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import tempfile
import io

# Load environment variables
load_dotenv()

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def load_youtube_content(url):
    """Load YouTube content with fallback methods"""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL")
    
    # Try multiple methods silently
    methods = [
        lambda: YoutubeLoader.from_youtube_url(url, add_video_info=True).load(),
        lambda: YoutubeLoader.from_youtube_url(f"https://www.youtube.com/watch?v={video_id}", add_video_info=False).load(),
        lambda: YoutubeLoader.from_youtube_url(url, add_video_info=True, language=["en", "auto"]).load()
    ]
    
    for method in methods:
        try:
            docs = method()
            if docs and docs[0].page_content.strip():
                return docs, "transcript"
        except:
            continue
    
    # Try youtube-transcript-api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            transcript = transcript_list.find_generated_transcript(['en'])
        
        transcript_data = transcript.fetch()
        full_text = " ".join([item['text'] for item in transcript_data])
        doc = Document(page_content=full_text, metadata={"source": url})
        return [doc], "transcript"
    except:
        pass
    
    # Fallback to video info
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
        
        if response.status_code == 200:
            title_match = re.search(r'"title":"([^"]+)"', response.text)
            title = title_match.group(1) if title_match else f"YouTube Video {video_id}"
            
            desc_match = re.search(r'"shortDescription":"([^"]+)"', response.text)
            description = desc_match.group(1) if desc_match else "No description available."
            
            content = f"Title: {title}\n\nDescription: {description}"
            doc = Document(page_content=content, metadata={"source": url, "title": title})
            return [doc], "basic_info"
    except:
        pass
    
    raise Exception("Unable to extract content from YouTube video")

def load_pdf_content(uploaded_file):
    """Load PDF content from uploaded file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if not documents:
            raise Exception("No content extracted from PDF")
        
        # Split large documents if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Combine all pages into one document for summarization
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        # Split if too large
        if len(full_text) > 10000:
            split_docs = text_splitter.split_documents(documents)
            return split_docs
        else:
            return documents
            
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

# Streamlit page configuration
st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="🎯",
    layout="centered"
)

# Custom CSS - Enhanced for new features
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
    }
    
    /* Remove Streamlit default styling */
    .stApp > header {
        background: transparent;
    }
    
    /* Compact Header - Elegant & Professional */
    .main-header {
        text-align: center;
        padding: 2rem 1.5rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-radius: 16px;
        margin: 1rem 0 2rem 0;
        backdrop-filter: blur(30px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 
            0 10px 25px rgba(0, 0, 0, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.05) inset;
        position: relative;
    }
    
    /* Larger, Bold Title */
    .main-header h1 {
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 50%, #cbd5e1 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }
    
    .main-header p {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        opacity: 0.95;
        max-width: 550px;
        margin: 0 auto !important;
        line-height: 1.5;
    }
    
    /* Compact Content Type Section */
    .content-type-section {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        padding: 1.8rem;
        border-radius: 16px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 2rem;
    }
    
    .content-type-section h3 {
        color: #f1f5f9 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
        text-align: center;
        letter-spacing: -0.01em;
    }
    
    /* Radio button styling - Enhanced */
    .stRadio > div {
        display: flex !important;
        flex-direction: row !important;
        gap: 1rem !important;
        justify-content: center !important;
        flex-wrap: wrap !important;
    }
    
    .stRadio > div > label {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.8) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 16px !important;
        padding: 1.2rem 2rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 180px !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        position: relative !important;
        overflow: hidden !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stRadio > div > label::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .stRadio > div > label:hover {
        border: 1px solid #3b82f6 !important;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(30, 41, 59, 0.9) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2) !important;
        color: #f1f5f9 !important;
    }
    
    .stRadio > div > label:hover::before {
        left: 100%;
    }
    
    .stRadio > div > label > div:first-child {
        display: none !important;
    }
    
    /* Selected state styling */
    .stRadio > div > label[data-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        border: 1px solid #3b82f6 !important;
        color: white !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Input Section */
    .input-section {
        background: rgba(15, 23, 42, 0.6);
        padding: 2rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }
    
    .input-section h3 {
        color: #e2e8f0 !important;
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Input Field Styling */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 16px !important;
        padding: 14px 18px !important;
        transition: all 0.2s ease !important;
        font-weight: 400 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #64748b !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 2px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #3b82f6 !important;
        background: rgba(59, 130, 246, 0.05) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        width: 100% !important;
        margin-top: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.35) !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.2) !important;
        border-radius: 12px !important;
        color: #22c55e !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
        border-radius: 12px !important;
        color: #ef4444 !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 12px !important;
        color: #3b82f6 !important;
    }
    
    /* Summary Container */
    .summary-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        margin: 2rem 0;
        color: #f8fafc !important;
        line-height: 1.7;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .summary-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    
    .summary-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        color: #e2e8f0 !important;
        letter-spacing: -0.01em;
    }
    
    .summary-container p {
        color: #cbd5e1 !important;
        margin-bottom: 1rem;
    }
    
    .summary-container * {
        color: #cbd5e1 !important;
    }
    
    /* Enhanced Metrics Styling - Centered */
    .metric-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        text-align: center !important;
        position: relative;
        overflow: hidden;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.4), transparent);
    }
    
    .metric-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Center align all metric content */
    .metric-container * {
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .metric-container [data-testid="metric-container"] {
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    .metric-container [data-testid="metric-container"] > div {
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto !important;
    }
    
    /* Fix metric label and value alignment */
    .metric-container [data-testid="metric-container"] [data-testid="metric-container-label"] {
        text-align: center !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .metric-container [data-testid="metric-container"] [data-testid="metric-container-value"] {
        text-align: center !important;
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    /* Footer */
    .footer {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 3rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        text-align: center;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Fix text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
<div class="main-header">
    <h1>🎯 AI Content Summarizer</h1>
    <p>Transform YouTube videos, web articles, and PDF documents into intelligent summaries</p>
</div>
''', unsafe_allow_html=True)

# Content type selection
st.markdown('''
<div class="content-type-section">
    <h3>📂 Choose Content Type</h3>
</div>
''', unsafe_allow_html=True)

content_type = st.radio(
    "Select the type of content you want to summarize:",
    ["🎥 YouTube Video", "🌐 Website/Article", "📄 PDF Document"],
    horizontal=True,
    label_visibility="collapsed"
)

# Dynamic input section based on selection
if content_type == "🎥 YouTube Video":
    st.markdown('''
    <div class="input-section">
        <h3>🎬 Enter YouTube Video URL</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    generic_url = st.text_input(
        "YouTube URL",
        placeholder="✨ Paste your YouTube video URL here (e.g., https://www.youtube.com/watch?v=...)",
        label_visibility="collapsed",
        key="youtube_url"
    )
    
    # URL validation for YouTube
    url_valid = False
    if generic_url:
        if "youtube.com" in generic_url or "youtu.be" in generic_url:
            st.success("🎥 YouTube video detected")
            url_valid = True
        else:
            st.error("❌ Please enter a valid YouTube URL")

elif content_type == "🌐 Website/Article":
    st.markdown('''
    <div class="input-section">
        <h3>🔗 Enter Website URL</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    generic_url = st.text_input(
        "Website URL",
        placeholder="✨ Paste your website URL here (e.g., https://example.com/article)",
        label_visibility="collapsed",
        key="website_url"
    )
    
    # URL validation for websites
    url_valid = False
    if generic_url:
        if validators.url(generic_url):
            st.success("🌐 Website URL detected")
            url_valid = True
        else:
            st.error("❌ Please enter a valid website URL")

else:  # PDF Document
    st.markdown('''
    <div class="input-section">
        <h3>📁 Upload PDF Document</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to summarize its content",
        label_visibility="collapsed"
    )
    
    url_valid = uploaded_file is not None
    if uploaded_file:
        st.success(f"📄 PDF uploaded: {uploaded_file.name}")
        generic_url = None  # Not needed for PDF

# Summarize button
generate_summary = st.button("🚀 Generate Summary")

# Auto-generate summary when user enters a valid URL or uploads a file
should_generate = False

if content_type in ["🎥 YouTube Video", "🌐 Website/Article"]:
    # Check if URL changed and is valid
    if generic_url and url_valid:
        current_input_key = f"{content_type}_{generic_url}"
        if st.session_state.get('last_processed_input') != current_input_key:
            should_generate = True
            st.session_state.last_processed_input = current_input_key

elif content_type == "📄 PDF Document":
    # Check if file is uploaded
    if uploaded_file:
        current_input_key = f"{content_type}_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get('last_processed_input') != current_input_key:
            should_generate = True
            st.session_state.last_processed_input = current_input_key

# Generate summary automatically or when button is clicked
if generate_summary or should_generate:
    if content_type in ["🎥 YouTube Video", "🌐 Website/Article"] and not generic_url.strip():
        st.error("📝 Please enter a URL to get started!")
    elif content_type == "📄 PDF Document" and not uploaded_file:
        st.error("📝 Please upload a PDF file to get started!")
    elif content_type in ["🎥 YouTube Video", "🌐 Website/Article"] and not url_valid:
        st.error("🔗 Please enter a valid URL")
    else:
        # Check API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("🔑 GROQ API key not found. Please add GROQ_API_KEY to your .env file")
        else:
            try:
                # Show loading
                with st.spinner(f"Processing {content_type.lower()}..."):
                    # Initialize model
                    llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)
                    
                    # Prompt template
                    prompt_template = """
                    Provide a comprehensive and well-structured summary of the following content in approximately 300 words:
                    
                    Content: {text}
                    
                    Focus on:
                    - Main points and key insights
                    - Important details and context
                    - Clear, organized structure
                    - Actionable information if applicable
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                    
                    # Load content based on type
                    if content_type == "🎥 YouTube Video":
                        try:
                            docs, content_type_info = load_youtube_content(generic_url)
                        except Exception:
                            st.error("❌ Could not extract content from YouTube video")
                            st.info("💡 Try a different video or check if it has captions available")
                            st.stop()
                    
                    elif content_type == "🌐 Website/Article":
                        try:
                            loader = UnstructuredURLLoader(
                                urls=[generic_url],
                                ssl_verify=False,
                                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                            )
                            docs = loader.load()
                            if not docs:
                                raise Exception("No content extracted")
                        except Exception:
                            st.error("❌ Could not extract content from website")
                            st.info("💡 Make sure the website is accessible and allows content extraction")
                            st.stop()
                    
                    else:  # PDF Document
                        try:
                            docs = load_pdf_content(uploaded_file)
                        except Exception as e:
                            st.error(f"❌ Could not process PDF: {str(e)}")
                            st.info("💡 Make sure the PDF is not password protected and contains readable text")
                            st.stop()
                    
                    # Generate summary
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)
                
                # Display results
                if summary and summary.strip():
                    st.markdown("---")
                    
                    # Display summary
                    st.markdown(f"""
                    <div class="summary-container">
                        <div class="summary-title">✨ AI-Generated Summary ✨</div>
                        <div style="color: #ffffff !important; position: relative; z-index: 1;">
                            {summary.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Stats - Centered and Enhanced
                    st.markdown('<div style="display: flex; justify-content: center; margin: 2rem 0;">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
                    total_content = " ".join([doc.page_content for doc in docs])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div data-testid="metric-container">
                                <div data-testid="metric-container-label">📄 {"Pages" if content_type=="📄 PDF Document" else "Documents"}</div>
                                <div data-testid="metric-container-value">{len(docs)}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div data-testid="metric-container">
                                <div data-testid="metric-container-label">📊 Summary Words</div>
                                <div data-testid="metric-container-value">{len(summary.split())}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div data-testid="metric-container">
                                <div data-testid="metric-container-label">📝 Source Words</div>
                                <div data-testid="metric-container-value">{len(total_content.split())}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show content type info
                    content_icons = {
                        "🎥 YouTube Video": "🎬",
                        "🌐 Website/Article": "🌐", 
                        "📄 PDF Document": "📄"
                    }
                    st.info(f"{content_icons[content_type]} Successfully summarized {content_type.lower()}")
                    
                    st.success("✅ Summary completed!")
                else:
                    st.error("❌ Failed to generate summary")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; font-weight: 500;">
        ✨ Powered by LangChain & Groq ✨
    </p>
</div>
""", unsafe_allow_html=True)