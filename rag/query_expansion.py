from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()
def expand_query(question):
    """
    Convert a user's natural-language question
    into a concise search query optimized for
    semantic retrieval.
    """

    llm = ChatOpenAI(
    model=model_name,
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

    prompt = f"""
You are a query optimizer for a RAG system.

Your job is to rewrite the user's question into a
search query that maximizes retrieval of relevant
documents.

IMPORTANT:
Use terminology that is likely to appear in the source
documents.

Rules:
- Preserve the main topic.
- Preserve important terminology from the question.
- Add closely related terms and synonyms.
- Prefer technical/document terminology over generic words.
- Do not answer the question.
- Do not introduce unrelated concepts.
- Return ONLY the search query.
- Keep it concise.

Examples:

Question:
What are the criticisms of LangChain?

Good search query:
LangChain limitations criticisms complexity security

Question:
What are the security concerns in LangChain?

Good search query:
LangChain security risks external integrations data exposure vulnerabilities

Question:
What is LangChain architecture?

Good search query:
LangChain architecture components modular design

Question:
How does LangChain simplify LLM application development?

Good search query:
LangChain LLM application development components framework

User question:
{question}

Search query:
"""

    response = llm.invoke(prompt)

    return response.content.strip()