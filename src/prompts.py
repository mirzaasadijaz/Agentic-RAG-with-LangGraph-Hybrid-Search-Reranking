"""
All prompt templates for the agent, kept in one place so tone and
instructions are easy to audit and tune independently of the graph logic.

Template variables - each must be supplied by the matching call in nodes.py:
    router_prompt         {kb_description} {question}
    chitchat_prompt       {question}
    grade_doc_prompt      {document} {question}
    rag_prompt            {context} {question}
    hallucination_prompt  {documents} {generation}
    answer_grade_prompt   {question} {generation}
    rewrite_prompt        {kb_description} {question} {previous_query}
"""
from langchain_core.prompts import ChatPromptTemplate

# --- Router: chitchat vs. vectorstore vs. web search ------------------------------------

ROUTER_SYSTEM_PROMPT = """You route a user message to exactly one destination.

The user's knowledge base (their uploaded documents) covers:
{kb_description}

Destinations:
1. "chitchat": greetings, thanks, goodbyes, or questions about what you can do, with \
no information need. Examples: "hi", "hello", "thanks", "how are you", "what can you do".
2. "vectorstore": anything the knowledge base above could help answer: a topic, term, \
concept, or file name from it, or a question about a document's contents, structure, \
title, or author. Examples: "what does the introduction say", "who wrote this book", \
"how is this term defined".
3. "web_search": questions clearly unrelated to the knowledge base, such as current \
events, weather, sports scores, prices, or general knowledge about a different subject. \
Example: "what is the weather in Paris today".

Rules:
- A message that asks a question or makes a request about any subject is not "chitchat", \
even if it starts with a greeting ("hi, what is a graph" goes to "vectorstore").
- If you are unsure between "vectorstore" and "web_search", choose "vectorstore". The \
system falls back to the web automatically when the documents have no answer."""

router_prompt = ChatPromptTemplate.from_messages(
    [("system", ROUTER_SYSTEM_PROMPT), ("human", "{question}")]
)

# --- Chitchat: answer greetings directly, no retrieval ----------------------------------

CHITCHAT_SYSTEM_PROMPT = """You are a friendly assistant inside a document \
question-answering app. The app answers questions from the user's uploaded documents \
and falls back to a web search when the documents don't have the answer.

Reply in one or two short sentences: greet back, say you're welcome, or briefly explain \
what the app can do. Then invite the user to ask a question about their documents. \
Never make up facts about the documents."""

chitchat_prompt = ChatPromptTemplate.from_messages(
    [("system", CHITCHAT_SYSTEM_PROMPT), ("human", "{question}")]
)

# --- Retrieval grader: does this one retrieved chunk help answer the question? ----------

GRADE_DOC_SYSTEM_PROMPT = """You are a grader deciding whether a retrieved text chunk \
helps answer a user question.

The chunk comes from a PDF or other document, so it may contain broken line breaks, \
page numbers, or headers. Ignore that formatting noise.

Grade 'yes' if the chunk contains the concept the question asks about (or a close \
synonym), or information that directly helps answer it: a definition, explanation, \
example, formula, or closely related fact.
Grade 'no' if it only shares the general subject area (for example, it comes from the \
same book but covers a different concept) or is about something else entirely.
When genuinely unsure, grade 'yes'.

Respond with a binary score: 'yes' or 'no'."""

grade_doc_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADE_DOC_SYSTEM_PROMPT),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ]
)

# --- Generation -----------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You answer the user's question using ONLY the context in their \
message. The context is a set of numbered passages from the user's documents (or from a \
web search, when the documents had no answer).

Rules:
- Answer directly in at most 4 sentences (or a short list if the question asks for steps \
or several items). Don't start with "Based on the context" and don't mention passage \
numbers.
- Use the exact terms, names, and numbers from the passages. Don't add facts from your \
own knowledge.
- If the passages answer only part of the question, give the part they support and say \
what is missing.
- If the passages don't contain the answer, reply exactly: "I couldn't find that in the \
available sources." Don't guess."""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# --- Hallucination grader: is the generation supported by the passages? -----------------

HALLUCINATION_SYSTEM_PROMPT = """You are a grader checking whether an answer is \
supported by a set of source passages.

Grade 'yes' if every concrete claim in the answer (definitions, names, numbers, steps) \
is supported by the passages, even when it is paraphrased or summarized. An answer that \
only says the information could not be found makes no claims, so grade it 'yes'.
Grade 'no' if the answer states concrete claims that the passages do not support or \
that contradict them.

Respond with a binary score: 'yes' or 'no'."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_SYSTEM_PROMPT),
        ("human", "Facts:\n\n{documents}\n\nLLM answer: {generation}"),
    ]
)

# --- Answer grader: does the generation actually address the question? -----------------

ANSWER_GRADE_SYSTEM_PROMPT = """You are a grader checking whether an answer resolves \
the user's question.

Grade 'yes' if the answer gives useful information that directly addresses the \
question, even if it is brief.
Grade 'no' if it says the information could not be found, avoids the question, or is \
about something else.

Respond with a binary score: 'yes' or 'no'."""

answer_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_GRADE_SYSTEM_PROMPT),
        ("human", "User question:\n\n{question}\n\nLLM answer: {generation}"),
    ]
)

# --- Query rewriter ---------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """You rewrite a user question into a search query for a hybrid \
(semantic + keyword) search over the user's documents.

The documents cover:
{kb_description}

Rules:
- Keep the meaning of the original question. Don't answer it and don't add facts.
- Use the vocabulary the documents would use: formal terms, synonyms, related keywords.
- The previous query did not retrieve useful passages, so write a clearly different \
query (different keywords or phrasing) that is still faithful to the original question.
- Return ONLY the new query on one line: no quotes, no explanation."""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        (
            "human",
            "Original question: {question}\nPrevious query: {previous_query}\n\nNew search query:",
        ),
    ]
)