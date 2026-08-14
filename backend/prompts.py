"""Prompt templates for the RAG chain and premium AI features."""

RAG_SYSTEM_PROMPT = """You are RAGenius, a precise and helpful document assistant.
Answer the user's question using ONLY the context provided below. If the answer
is not contained in the context, say clearly that the documents don't contain
that information instead of guessing.

Rules:
- Ground every claim in the provided context.
- Be concise but complete.
- Use Markdown formatting (lists, bold, code blocks) where it improves clarity.
- Do not fabricate sources or facts not present in the context.
"""


def build_rag_prompt(question: str, context_blocks: list[str], history: str = "") -> str:
    """Assemble the final prompt sent to Gemini for a RAG-grounded answer."""
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"
    history_block = f"\nConversation so far:\n{history}\n" if history else ""
    return (
        f"{RAG_SYSTEM_PROMPT}\n"
        f"{history_block}\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def build_summary_prompt(text: str) -> str:
    return (
        "Summarize the following document in 4-6 concise sentences, "
        "capturing the main purpose and key points. Use plain prose, no preamble.\n\n"
        f"Document:\n{text}"
    )


def build_key_insights_prompt(text: str) -> str:
    return (
        "Extract the 5-8 most important insights from the following document as a "
        "Markdown bullet list. Each bullet should be one clear sentence.\n\n"
        f"Document:\n{text}"
    )


def build_keywords_prompt(text: str) -> str:
    return (
        "List the 10 most important keywords or key phrases from the following document, "
        "comma-separated, no numbering, no explanation.\n\n"
        f"Document:\n{text}"
    )


def build_faq_prompt(text: str) -> str:
    return (
        "Generate 5 frequently asked questions with concise answers based on the following "
        "document. Format as Markdown: **Q:** ... / **A:** ...\n\n"
        f"Document:\n{text}"
    )


def build_followup_prompt(question: str, answer: str) -> str:
    return (
        "Based on this question and answer exchange, suggest exactly 3 short, "
        "relevant follow-up questions the user might ask next. Return them as a "
        "plain numbered list, nothing else.\n\n"
        f"Q: {question}\nA: {answer}"
    )


def build_comparison_prompt(name_a: str, text_a: str, name_b: str, text_b: str) -> str:
    """Prompt Gemini to compare two documents side by side."""
    header = (
        "Compare the two documents below. Respond in Markdown with these sections:\n"
        "## Summary of Each Document\n"
        "## Key Similarities\n"
        "## Key Differences\n"
        "## Notable Points Unique to Each\n\n"
    )
    return f"{header}### Document A: {name_a}\n{text_a}\n\n### Document B: {name_b}\n{text_b}\n"
