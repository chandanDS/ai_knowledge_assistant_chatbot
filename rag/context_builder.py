import os

def build_context(results):
    """
    Build LLM-ready context from retrieved documents.
    """

    if not results:
        return ""

    context_parts = []

    for i, (doc, score) in enumerate(results, start=1):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Similarity Distance: {score:.4f}\n"
            f"Content:\n"
            f"{doc.page_content.strip()}\n"
        )

    return "\n\n".join(context_parts)