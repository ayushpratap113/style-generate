from langchain.prompts import PromptTemplate

def get_prompt_template(external_knowledge: int, word_limit: int, style_choice: str) -> PromptTemplate:
    """
    Returns a PromptTemplate based on style_choice:
      - "Email Style"
      - "Normal Style"
      - "Report Style"
      - "Feedback Style"
    """
    if style_choice == "Email Style":
        # Email style prompt
        prompt_template = f"""
        Human: 
        You have read a set of emails in the context below. These emails have a unique writing style: tone, choice of words, and sentence structure. 
        Please answer and write the mail to the question **in the same writing style** as those emails, signed off as "Ayush". 
        Rely primarily on the given context. If the context doesn't have enough info, 
        you should use up to an external knowledge level of {external_knowledge} (scale 1-10). 
        
        Your response should be of {word_limit} words.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (in the same style and within {word_limit} words):
        """
    elif style_choice == "Report Style":
        # Report style prompt
        prompt_template = f"""
        Human:
        You are a professional and knowledgeable AI assistant. The user wants a detailed report like given in all the examples. 
        Please create a well-structured report with the following rules 
        1)
        2)
        3)
        Rely primarily on the given context. If the context doesn't have enough info, you may use up to 
        an external knowledge level of {external_knowledge} (scale 1-10), 
        but if you still cannot find the answer, say "I don't know."

        Limit the response to {word_limit} words.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (report style, within {word_limit} words):
        """
    elif style_choice == "Feedback Style":
        # Feedback style prompt
        prompt_template = f"""
        Human:
        The user needs feedback or critique based on the context below having same writing style: tone, choice of words, and sentence structure given for different person( like Feedback given for Person x).
        Please provide the answer to the two questions  
        1)  What is this person's super powers(s
        2) What growth idea(s) do you suggest for this person?
          
        If the context doesn't have enough info, you may use up to an external knowledge.

        Keep your response as less than 60 words(approx 60 words) for each question.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (constructive feedback, within 120 words for each question (60 words)):
        """
    else:
        # Normal style prompt
        prompt_template = f"""
        Human:
        You are a helpful and knowledgeable AI assistant. 
        Please answer the question below using the provided context. 
        If the context does not have enough information, 
        you may use up to an external knowledge level of {external_knowledge} (scale 1-10), 
        but if you still cannot find the answer, say "I don't know."
        
        Limit the response to {word_limit} words.

        <context>
        {{context}}
        </context>

        Question: {{question}}

        Assistant (helpful, within {word_limit} words):
        """
    
    return PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )