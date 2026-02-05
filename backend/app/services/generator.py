from openai import OpenAI
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.prompts.system import (
    DRUG_SAFETY_SYSTEM_PROMPT,
    REFUSAL_PROMPT,
    format_context,
)
from app.services.retriever import RetrievalResult


@dataclass
class GenerationResult:
    """Represents a generation result."""

    text: str
    model: str
    usage: dict


class GeneratorService:
    """Service for generating responses using OpenAI LLM."""

    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def generate(
        self,
        query: str,
        context_chunks: list[RetrievalResult],
        temperature: float = 0.3,
    ) -> GenerationResult:
        """Generate a response based on query and context."""
        # Format context from retrieval results
        context = format_context(context_chunks)

        # Build the prompt
        prompt = DRUG_SAFETY_SYSTEM_PROMPT.format(
            context=context,
            query=query,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
        )

        return GenerationResult(
            text=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def generate_refusal(self, query: str) -> GenerationResult:
        """Generate a polite refusal for out-of-scope queries."""
        prompt = REFUSAL_PROMPT.format(query=query)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )

        return GenerationResult(
            text=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    def is_drug_safety_query(self, query: str) -> bool:
        """Check if the query is related to drug safety (simple heuristic)."""
        drug_keywords = [
            "drug",
            "medication",
            "medicine",
            "side effect",
            "adverse",
            "interaction",
            "dose",
            "dosage",
            "contraindication",
            "warning",
            "precaution",
            "allergy",
            "reaction",
            "symptom",
            "prescription",
            "otc",
            "fda",
            "label",
            "safety",
            "risk",
            "take",
            "taking",
            "pill",
            "tablet",
            "capsule",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in drug_keywords)
