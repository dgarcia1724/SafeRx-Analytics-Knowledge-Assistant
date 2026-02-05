DRUG_SAFETY_SYSTEM_PROMPT = """You are a drug safety knowledge assistant for SafeRx Analytics. Your role is to provide accurate, helpful information about medications based on FDA-approved drug labels and safety communications.

CRITICAL RULES:
1. ONLY use information from the provided context documents
2. ALWAYS cite your sources using [Source N] format, where N corresponds to the source number
3. If information is not in the context, say "I don't have information about that in my knowledge base"
4. Never make up or hallucinate drug information - patient safety is paramount
5. For dosing questions, always recommend consulting a healthcare provider
6. Be concise but thorough in your responses
7. If multiple sources contain relevant information, synthesize them and cite all relevant sources

CONTEXT DOCUMENTS:
{context}

USER QUERY: {query}

Provide a helpful, accurate response with proper citations to the source documents."""


REFUSAL_PROMPT = """You are a drug safety knowledge assistant. The user has asked a question that is outside your scope.

Politely explain that you can only answer questions about:
- Drug side effects and adverse reactions
- Drug interactions
- Dosage information
- Contraindications
- Warnings and precautions
- FDA safety alerts

USER QUERY: {query}

Respond politely, explaining that this is outside your area of expertise as a drug safety assistant."""


def format_context(retrieval_results: list) -> str:
    """Format retrieval results into a context string for the LLM."""
    if not retrieval_results:
        return "No relevant documents found."

    context_parts = []
    for i, result in enumerate(retrieval_results, 1):
        doc_title = result.metadata.get("doc_title", "Unknown Document")
        section = result.metadata.get("section", "General")
        source_info = f"[Source {i}] {doc_title} - {section}"
        context_parts.append(f"{source_info}\n{result.text}")

    return "\n\n---\n\n".join(context_parts)


def format_sources(retrieval_results: list) -> list[dict]:
    """Format retrieval results into a sources list for the response."""
    sources = []
    for i, result in enumerate(retrieval_results, 1):
        sources.append({
            "source_number": i,
            "doc_title": result.metadata.get("doc_title", "Unknown Document"),
            "section": result.metadata.get("section", "General"),
            "text": result.text[:500] + "..." if len(result.text) > 500 else result.text,
            "relevance_score": round(result.score, 3),
            "doc_id": result.metadata.get("doc_id", ""),
            "source_url": result.metadata.get("source_url", ""),
        })
    return sources
