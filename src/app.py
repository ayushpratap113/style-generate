import streamlit as st
from config import load_config
from faiss_utils import load_index
from llm_utils import get_llm, get_response
from prompt_utils import get_prompt_template

def main():
    st.title("Email/Report/Normal/Feedback Writer (Style-aware RAG with Word Limit)")

    # UI Controls
    style_choice = st.radio(
        "Select the response style", 
        ("Email Style", "Normal Style", "Report Style", "Feedback Style")
    )
    
    # Map style_choice to the style_key used to load the correct FAISS index
    style_key_map = {
        "Email Style": "mail",
        "Normal Style": "normal",
        "Report Style": "report",
        "Feedback Style": "feedback"
    }
    style_key = style_key_map[style_choice]

    temperature = st.slider("Select Temperature", 0.0, 1.0, 0.5) 
    external_knowledge = st.slider("Select external knowledge level (1 = minimal, 10 = maximum)", 1, 10, 5)
    
    # Instead of max tokens, we use word_limit
    word_limit = st.number_input("Word limit (approximate)", min_value=1, max_value=5000, value=50, step=5)
    
    # Larger, resizable input box for questions
    question = st.text_area(
        "Ask a question or request text",
        height=80,
        placeholder="Enter your question or request here..."
    )

    if st.button("Load Style Index & Generate Styled Response"):
        # 1. Load the relevant index for the chosen style
        st.write(f"Loading index for style: {style_choice}")
        faiss_index = load_index(style_key)
        if not faiss_index:
            st.error("No FAISS index loaded. Please ensure you have run admin.py for this style.")
            return
        
        # 2. Generate response if question is provided
        if question.strip() == "":
            st.warning("Please enter a question or prompt.")
        else:
            with st.spinner("Querying the style-based RAG system..."):
                llm = get_llm(temperature)
                response = get_response(
                    llm, 
                    faiss_index, 
                    question, 
                    external_knowledge, 
                    word_limit, 
                    style_choice
                )
                st.write(response)
                st.success("Done")

if __name__ == "__main__":
    load_config()
    main()