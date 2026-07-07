# app.py
# ----------------------------
# Web interface for your ICU Protocol RAG System
# This creates a user-friendly chat interface for medical professionals
# ----------------------------

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from src.search import ProtocolSearcher

# Load environment variables
load_dotenv()

# Initialize Navigator AI client
@st.cache_resource
def get_ai_client():
    """Initialize the Navigator AI client (cached for performance)"""
    return OpenAI(
        api_key=os.getenv("NAVIGATOR_API_KEY"),
        base_url=os.getenv("NAVIGATOR_BASE_URL"),
    )

@st.cache_resource
def get_protocol_searcher():
    """Initialize the protocol searcher (cached for performance)"""
    return ProtocolSearcher()

def generate_ai_response(question, protocol_context):
    """
    Generate an AI response using Navigator + retrieved protocol information
    """
    client = get_ai_client()
    
    # Create a prompt that combines the question with relevant protocol info
    prompt = f"""You are an AI assistant helping ICU medical professionals with protocol questions.

RELEVANT ICU PROTOCOL INFORMATION:
{protocol_context}

CLINICAL QUESTION: {question}

Please provide a helpful, accurate response based on the protocol information above. If the protocols don't contain enough information to fully answer the question, say so and provide what information is available. Always mention that this is based on the provided protocols and encourage consulting with senior staff for complex cases.

Include specific dosing, duration, and approval requirements when mentioned in the protocols."""

    try:
        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Lower temperature for more consistent medical responses
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating AI response: {str(e)}"

# Streamlit App Configuration
st.set_page_config(
    page_title="ICU Protocol Assistant", 
    page_icon="🏥",
    layout="wide"
)

# App Header
st.title("🏥 ICU Protocol Assistant")
st.subheader("AI-powered search through ICU protocols and procedures")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("📋 About")
    st.write("This tool helps ICU staff quickly find relevant protocol information using AI-powered search.")
    st.write("**Current Protocols:**")
    st.write("• UFH Antibiotic Protocol")
    st.write("• ICU Delirium Protocol")
    
    st.header("🔍 How to Use")
    st.write("1. Ask a clinical question")
    st.write("2. Review relevant protocol sections")
    st.write("3. Get AI-powered guidance")
    
    st.markdown("---")
    st.caption("⚠️ Always consult senior staff for complex cases")

# Main interface
st.header("💬 Ask a Clinical Question")

# Initialize searcher
try:
    searcher = get_protocol_searcher()
    st.success(f"✅ Loaded {len(searcher.text_chunks)} protocol sections")
except Exception as e:
    st.error(f"❌ Error loading protocols: {str(e)}")
    st.stop()

# Question input
question = st.text_input(
    "Enter your clinical question:",
    placeholder="e.g., What antibiotic should I use for pneumonia?",
    help="Ask about dosing, duration, indications, or approval requirements"
)

# Process question when submitted
if question:
    with st.spinner("🔍 Searching protocols..."):
        # Search for relevant protocol information
        protocol_context = searcher.answer_question(question, top_k=3)
        
        # Create two columns for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📄 Relevant Protocol Sections")
            if "No relevant protocol information found" in protocol_context:
                st.warning("❌ No relevant protocol information found for this question.")
            else:
                # Display the protocol context in a nice format
                st.text_area(
                    "Protocol Information:",
                    protocol_context,
                    height=400,
                    help="Raw protocol text matching your question"
                )
        
        with col2:
            st.header("🤖 AI Guidance")
            with st.spinner("🧠 Generating AI response..."):
                ai_response = generate_ai_response(question, protocol_context)
                st.markdown(ai_response)

# Example questions section
st.markdown("---")
st.header("💡 Example Questions")

example_cols = st.columns(2)
with example_cols[0]:
    st.write("**Antibiotic Questions:**")
    if st.button("What antibiotic for pneumonia?"):
        st.session_state.example_question = "What antibiotic should I use for pneumonia?"
    if st.button("How long should treatment last?"):
        st.session_state.example_question = "How long should antibiotic treatment last?"
        
with example_cols[1]:
    st.write("**Protocol Questions:**")
    if st.button("When do I need ID approval?"):
        st.session_state.example_question = "When do I need ID approval?"
    if st.button("What are the dosing guidelines?"):
        st.session_state.example_question = "What is the dosing for antibiotics?"

# Handle example question clicks
if "example_question" in st.session_state:
    st.rerun()

# Footer
st.markdown("---")
st.caption("🏥 ICU Protocol Assistant • University of Florida • Powered by Navigator AI")