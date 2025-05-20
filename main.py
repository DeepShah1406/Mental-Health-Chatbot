from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from fpdf import FPDF  # Using the original fpdf package
from datetime import datetime

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

working_dir = os.path.dirname(os.path.realpath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# FastAPI app and Pydantic model for API input
app = FastAPI()

class MessageRequest(BaseModel):
    message: str

# FastAPI route for chatbot
@app.post("/chat")
async def chatbot(request: MessageRequest):
    message = request.message

    # Setup vectorstore (same as Streamlit code)
    vectorstore = setup_vectorstore()

    # Setup the conversational chain (same as Streamlit code)
    conversational_chain = chat_chain(vectorstore)

    # Check for sensitive topics
    if contains_sensitive_topics(message):
        response = "It seems you may be experiencing distress. Please know that help is available. If you feel unsafe or overwhelmed, consider calling emergency services immediately. For ongoing support, here are some mental health helpline numbers: [Provide relevant Indian mental health helpline numbers]. We strongly encourage you to seek professional help."
    else:
        # Get response from the conversational chain
        response = conversational_chain({"question": message})["answer"]

    return {"response": response}

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are a specialized mental health assistant focused exclusively on mental health topics. 
Your role is to provide support, information, and guidance about mental health while maintaining professional boundaries.
Always respond with care, compassion, and evidence-based information from the provided context. 
Keep the response concise and to the point and also short but not too short, short enough to be read in one go.
Also make use of bullet points and lists wherever necessary.
Make use of few emojis only to make the response somewhat engaging and interesting.

When you receive a query:
1. First check if the provided context contains relevant mental health information
2. If the context is empty or not relevant to mental health, politely inform the user that you can only discuss mental health topics
3. If the context contains relevant information, provide a helpful response based on that information
4. For every incoming query, first assess whether it includes any trigger words/phrases related to self-harm, suicide, or harmful behaviors
5. If trigger words/phrases are detected, append the following to your response:
   "It seems you may be experiencing distress. Please know that help is available. If you feel unsafe or overwhelmed, consider calling emergency services immediately. For ongoing support, here are some mental health helpline numbers: [Provide relevant Indian mental health helpline numbers]. We strongly encourage you to seek professional help."
6. If no trigger words/phrases are detected, respond normally without including the helpline information

Remember: 
- Only respond to questions where you can find relevant mental health information in the context
- Be empathetic and supportive in your responses
- Focus on providing evidence-based information from the context and the vector database
- When discussing sensitive topics, maintain a professional and caring tone
- Always encourage seeking professional help when appropriate and needed, and when use of words/phrases related to self-harm, suicide, harmful behaviors, etc. are detected"""

DEFAULT_NEGATIVE_PROMPT = """Do not provide medical diagnoses, prescribe medications, or give specific treatment recommendations.
Do not encourage harmful behaviors or provide potentially dangerous advice.
Do not claim to be a licensed therapist or medical professional.
Do not provide any response that is not based on the provided mental health context and/ or the vector database.
If the context doesn't contain relevant mental health information, acknowledge this and politely inform the user that you can only discuss mental health topics.
Do not make assumptions about the user's mental state or condition.
Do not provide generic advice without context.
Do not share personal opinions or anecdotes."""

def contains_sensitive_topics(question):
    sensitive_keywords = [
        'suicide', 'self-harm', 'self harm', 'kill myself', 'kill', 'end my life',
        'want to die', 'hurting myself', 'cutting', 'overdose', 'harm myself',
        'suicidal', 'self injury', 'self-injury', 'self mutilation',
        'self-mutilation', 'suicidal thoughts', 'suicidal ideation',
        'harmful behaviors', 'hurting myself', 'ending it all', 'no reason to live',
        'can\'t go on', 'want to disappear', 'don\'t want to exist'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    # Create a combined prompt template
    prompt_template = f"""{system_prompt}

{negative_prompt}

Context (from mental health database):
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    memory = ConversationBufferMemory(
        llm = llm,
        output_key = "answer",
        memory_key = "chat_history",
        return_messages = True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        memory = memory,
        verbose = True,
        return_source_documents = True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

st.set_page_config(
    page_title="Chat with Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",  # Changed to wide layout to accommodate sidebar
)

# Custom CSS for sidebar styling
st.markdown("""
    <style>
    div.css-textbarboxtype {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px 20px 20px 70px;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    
    /* Justify text for Purpose section */
    div.css-textbarboxtype:nth-of-type(3) {
        text-align: justify;
        text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About Bot")
    
    # About Section
    st.markdown("## Description")
    st.markdown("""
        <div class="css-textbarboxtype">
            An AI-powered chatbot designed to provide mental health support.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Goals")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Provide 24/7 mental health support<br>
            - Offer crisis intervention when needed<br>
            - Connect users with professional resources
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Purpose")
    st.markdown("""
        <div class="css-textbarboxtype">
            Designed as a stigma-free entry point to mental health support in India—where people often feel too shy to seek traditional therapy—this chatbot tackles interpersonal harassment, workplace stress, and suicidal ideation by offering empathetic listening, practical coping strategies, and evidence-based guidance from the WHO and India's Ministry of Health and Family Welfare. By normalizing conversations about emotional well-being and delivering timely, trustworthy advice, it bridges users to professional therapists when they're ready.
        </div>
    """, unsafe_allow_html=True)
    
    # Values
    st.markdown("## Our Values")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Empathy<br>
            - Professional Ethics<br>
            - User Safety
        </div>
    """, unsafe_allow_html=True)
    
    # Chat History Section
    st.markdown("---")
    st.markdown("## Chat History")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history previews
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if st.button(f"Chat {idx//2 + 1}: {message['content'][:30]}...", key=f"history_{idx}"):
                # Load this conversation
                st.session_state.selected_chat = idx//2
    
    # PDF Export Button
    st.markdown("---")
    if st.button("Export Chat to PDF"):
        if len(st.session_state.chat_history) > 0:
            try:
                # Create PDF
                pdf = FPDF()
                pdf.add_page()
                
                # Use Arial Unicode MS font
                pdf.set_font('Arial', '', 10)
                
                # PDF Header
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, "Mental Health Chatbot - Conversation History", ln=True, align='C')
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)
                
                # Add conversation
                pdf.set_font('Arial', '', 10)
                for message in st.session_state.chat_history:
                    # Role header
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 10, message["role"].capitalize(), ln=True)
                    # Message content
                    pdf.set_font('Arial', '', 10)
                    pdf.multi_cell(0, 10, remove_emojis(message["content"]))
                    pdf.ln(5)
                
                # Save PDF
                filename = f"mental_health_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf.output(filename)
                
                # Create download button
                with open(filename, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=filename,
                        mime="application/pdf"
                    )
                
                # Clean up the file
                os.remove(filename)
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("No chat history to export!")

# Main chat interface
# Add header image

st.title("🧠 Mental Health Chatbot")

#st.image("https://img.freepik.com/premium-vector/mental-health-awareness-month-take-care-your-body-take-care-your-health-increase-awareness_758894-821.jpg?w=1380", use_column_width=True)

st.image("D:\Final Project\PythonDemoProject3\images\may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector.jpg")

#st.image("https://static.vecteezy.com/system/resources/previews/039/630/872/large_2x/may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector.jpg", use_column_width=True)
#st.image("https://static.vecteezy.com/system/resources/previews/040/941/494/non_2x/may-is-mental-health-awareness-month-banner-with-silhouettes-of-diverse-people-and-green-ribbon-women-and-men-of-different-ages-religions-and-races-design-for-info-importance-psychological-state-vector.jpg", use_column_width=True)
#st.image("https://static.vecteezy.com/system/resources/previews/038/147/547/non_2x/may-is-mental-health-awareness-month-banner-horizontal-design-with-man-women-children-old-people-silhouette-in-flat-style-informing-about-importance-of-good-state-of-mind-well-being-presentation-vector.jpg", use_column_width=True)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about Mental Health")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
