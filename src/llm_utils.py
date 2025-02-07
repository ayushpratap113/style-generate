import streamlit as st
from config import bedrock_client
from langchain.llms.bedrock import Bedrock
from prompt_utils import get_prompt_template

def get_llm(temperature: float):
    """
    Returns a Bedrock LLM with a large max_tokens_to_sample, 
    so it won't truncate early. We rely on the prompt 
    to enforce the word limit.
    """
    llm = Bedrock(
        model_id="anthropic.claude-v2:1", 
        client=bedrock_client,
        model_kwargs={
            'max_tokens_to_sample': 5000,  # Large enough to avoid premature truncation
            'temperature': temperature
        }
    )
    return llm

def get_response(llm, faiss_index, question, external_knowledge, word_limit, style_choice):
    """
    Retrieves relevant documents, constructs a prompt based on the selected style,
    and generates a response using the LLM.
    """
    # Step 1: Retrieve relevant documents from the FAISS index
    try:
        docs = faiss_index.similarity_search(question, k=5)  # Adjust 'k' as needed
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Error during similarity search: {str(e)}")
        return "An error occurred while retrieving information."

    # Step 2: Create the prompt using the selected style
    prompt_template = get_prompt_template(
        external_knowledge=external_knowledge,
        word_limit=word_limit,
        style_choice=style_choice
    )
    prompt = prompt_template.format(context=context, question=question)

    # Step 3: Generate the response using the LLM
    try:
        response = llm(prompt)
    except Exception as e:
        st.error(f"Error generating response from LLM: {str(e)}")
        return "An error occurred while generating the response."

    return response