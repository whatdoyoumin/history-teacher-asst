from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
import os
import re
from pydantic import BaseModel, ValidationError

# Define prompt templates

EVALUATE_SOURCES_PROMPT = PromptTemplate.from_template("""
You are an expert in assessing the reliability of information sources. Evaluate the reliability of each source listed below based on its type and metadata. Consider the following factors:  
- **Source Type** (e.g., book, website, academic journal, official school textbook)  
- **Authority** (e.g., official institution, personal blog, peer-reviewed publication)  
- **Relevance & Accuracy** (e.g., does the source align with established knowledge?)  
- **Potential Bias** (e.g., commercial, governmental, ideological bias)  
- **Verifiability** (e.g., presence of citations, traceability of claims)  

**Special Consideration:**  
- If the source mentions "Sec1" or "Sec2," it refers to the official **Secondary 1 or 2 school textbook**, which is typically reliable for educational purposes.  

### Sources for Evaluation:  
{sources}  

For each source, provide:  
1. **Reliability Rating** (e.g., High, Medium, Low)  
2. **Justification** (explain why you assigned this rating)  
3. **Any Potential Biases or Limitations**  

Ensure that your evaluation is structured, concise, and consistent across all sources.
""")


ANSWER_QUESTION_PROMPT = PromptTemplate.from_template("""
     You are the Heritage Education Research Assistant, an AI-powered tool designed to help educators in Singapore create comprehensive and balanced lesson plans about Singapore's history and culture. Your task is to provide multiple perspectives on historical questions, with a focus on validated sources from the National Heritage Board (NHB) and other reputable institutions.

Generate 3-5 different perspectives on the question, each with a brief summary (2-3 sentences) explaining the reasoning behind that perspective. For each perspective, include a source citation in one of the following formats:
Page Number (if the source is a book or document with specific page references),
Website Link (if the source is a digital resource or website),
Or both if applicable (e.g., a book citation with a page number and a link to the digital source).
Please refer to the context for the source citations.
                                                      
Format the answer as follows:

Perspective #: [Answer summary]
Page: [Page Number], Book Title: Sec1 or Sec2
OR
Website Link: [Link to the source]
OR
Page: [Page Number] | Website Link: [Link to the source]
                                                      
[Additional Perspectives if supported by context...]

[Discussion Questions]
(Only include questions that can be answered using the provided context)
1. (question that encourages critical thinking)
2. (question that encourages critical thinking)
3. (question that encourages critical thinking)

Ensure that the language and content complexity is appropriate for the specified student age group (if provided).

If a specific historical timeframe or theme is specified, tailor your responses to fit within those parameters.

After presenting the perspectives, suggest 2-3 discussion questions that could encourage critical thinking among students about these different viewpoints.
                                                      
Remember, your goal is to provide educators with balanced, well-sourced information that they can use to create engaging and thought-provoking lessons about Singapore's history and culture. Each citation should be appropriately linked to the perspective it corresponds to, whether it is a page number, website link, or both.

If the user asks a question unrelated to History - please say you don't have the available information and reccomend them to refer to other resources.
                                                      
Context: {context}

Question: {question}
                                                      
""")





def evaluate_sources(sources_text):
    """Evaluates the reliability of sources."""
    chain = (
        RunnableLambda(lambda x: {"sources": x["sources"]})
        | EVALUATE_SOURCES_PROMPT
        | ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    )
    result = chain.invoke({"sources": sources_text})
    return result.content



class AnswerFormat(BaseModel):
    perspectives: list[str]
    discussion_questions: list[str]

def validate_answer_format(answer: str) -> bool:
    """Validates the LLM output format."""
    perspectives = re.findall(r"Perspective \d+: (.*?)\n", answer)
    discussion_questions = re.findall(r"\d+\. (.*?)\n", answer.split("Discussion Questions:")[1] if "Discussion Questions:" in answer else "")
    
    try:
        AnswerFormat(perspectives=perspectives, discussion_questions=discussion_questions)
        return True
    except ValidationError:
        return False
