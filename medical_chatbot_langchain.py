import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="MediBot", page_icon="🏥")
st.title("🏥 Medical Chatbot for Symptom Analysis")
st.write("Describe your symptoms and we'll recommend which medical department to consult.")

# Initialize LangChain components
def initialize_chat():
    # Initialize Gemini LLM - UPDATED MODEL NAME
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    # System prompt template
    system_prompt = SystemMessagePromptTemplate.from_template(
        """You are a medical assistant chatbot designed to help patients identify which medical department 
        they should consult based on their symptoms. Your task is to:
        
        1. Ask clarifying questions if symptoms are vague
        2. Analyze the described symptoms
        3. Recommend the most appropriate medical department(s)
        4. Briefly explain your reasoning
        
        Departments include:
        - Cardiology (heart)
        - Dermatology (skin)
        - Endocrinology (hormones)
        - Gastroenterology (digestive)
        - Neurology (nervous system)
        - Ophthalmology (eyes)
        - Orthopedics (bones)
        - ENT (ears, nose, throat)
        - Pediatrics (children)
        - Pulmonology (lungs)
        - Rheumatology (joints)
        - Urology (urinary)
        
        Be professional and empathetic. If symptoms suggest an emergency, advise immediate medical attention.
        """
    )
    
    # Human prompt template
    human_prompt = HumanMessagePromptTemplate.from_template("{human_input}")
    
    # Combine prompts
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            human_prompt,
        ]
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create conversation chain
    conversation = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=False
    )
    
    return conversation

# Initialize chat in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_chat()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm here to help you determine which medical department you should consult based on your symptoms. Could you please describe how you're feeling?"
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Describe your symptoms..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.spinner("Analyzing symptoms..."):
        try:
            response = st.session_state.conversation.run(human_input=prompt)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            response = "I'm having trouble processing your request. Please try again later."
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Add disclaimer
st.sidebar.markdown("""
    **IBM GENAI Project** - Medical Chatbot for Symptom Analysis
                    
    **B Yashwanth Srinivasan**
""")

# Add department reference guide
st.sidebar.markdown("""

**Medical Departments Guide:**
- **Cardiology**: Heart and cardiovascular system
- **Dermatology**: Skin, hair, nails
- **Endocrinology**: Hormones and metabolism
- **Gastroenterology**: Digestive system
- **Neurology**: Brain and nervous system
- **Ophthalmology**: Eyes and vision
- **Orthopedics**: Bones, joints, muscles
- **ENT**: Ears, nose, throat
- **Pediatrics**: Children's health
- **Pulmonology**: Lungs and breathing
- **Rheumatology**: Joints and autoimmune diseases
- **Urology**: Urinary system
""")