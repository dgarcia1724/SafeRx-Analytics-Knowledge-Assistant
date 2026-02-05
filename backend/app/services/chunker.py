import re
import tiktoken
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    metadata: dict
    chunk_index: int


class DocumentChunker:
    """Service for chunking documents into smaller pieces for embedding."""

    # Standard FDA drug label sections
    FDA_SECTIONS = [
        "INDICATIONS AND USAGE",
        "DOSAGE AND ADMINISTRATION",
        "DOSAGE FORMS AND STRENGTHS",
        "CONTRAINDICATIONS",
        "WARNINGS AND PRECAUTIONS",
        "ADVERSE REACTIONS",
        "DRUG INTERACTIONS",
        "USE IN SPECIFIC POPULATIONS",
        "OVERDOSAGE",
        "DESCRIPTION",
        "CLINICAL PHARMACOLOGY",
        "NONCLINICAL TOXICOLOGY",
        "CLINICAL STUDIES",
        "HOW SUPPLIED",
        "STORAGE AND HANDLING",
        "PATIENT COUNSELING INFORMATION",
        "BOXED WARNING",
        "WARNINGS",
        "PRECAUTIONS",
    ]

    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def _identify_sections(self, text: str) -> list[tuple[str, str]]:
        """Identify FDA label sections in the text."""
        sections = []
        current_section = "GENERAL"
        current_text = []

        lines = text.split("\n")

        for line in lines:
            line_upper = line.strip().upper()
            # Check if line is a section header
            is_section_header = False
            for section in self.FDA_SECTIONS:
                if section in line_upper and len(line.strip()) < len(section) + 20:
                    # Save previous section
                    if current_text:
                        sections.append((current_section, "\n".join(current_text)))
                    current_section = section
                    current_text = []
                    is_section_header = True
                    break

            if not is_section_header:
                current_text.append(line)

        # Add last section
        if current_text:
            sections.append((current_section, "\n".join(current_text)))

        return sections

    def _split_text_by_tokens(
        self, text: str, section_name: str, base_metadata: dict
    ) -> list[Chunk]:
        """Split text into chunks based on token count."""
        chunks = []

        # Split into sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            metadata={
                                **base_metadata,
                                "section": section_name,
                            },
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                for word in words:
                    word_token_count = self._count_tokens(word + " ")
                    if word_tokens + word_token_count > self.chunk_size:
                        if word_chunk:
                            chunk_text = " ".join(word_chunk)
                            chunks.append(
                                Chunk(
                                    text=chunk_text,
                                    metadata={
                                        **base_metadata,
                                        "section": section_name,
                                    },
                                    chunk_index=chunk_index,
                                )
                            )
                            chunk_index += 1
                        word_chunk = [word]
                        word_tokens = word_token_count
                    else:
                        word_chunk.append(word)
                        word_tokens += word_token_count

                if word_chunk:
                    current_chunk = word_chunk
                    current_tokens = word_tokens
                continue

            # Check if adding sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            metadata={
                                **base_metadata,
                                "section": section_name,
                            },
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Take last few sentences for overlap
                    overlap_tokens = 0
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        sent_tokens = self._count_tokens(sent)
                        if overlap_tokens + sent_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_tokens += sent_tokens
                        else:
                            break
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = overlap_tokens + sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "section": section_name,
                    },
                    chunk_index=chunk_index,
                )
            )

        return chunks

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        source_url: str = "",
    ) -> list[Chunk]:
        """Chunk a document into smaller pieces with metadata."""
        base_metadata = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "source_url": source_url,
        }

        # Identify sections in the document
        sections = self._identify_sections(text)

        all_chunks = []
        for section_name, section_text in sections:
            if not section_text.strip():
                continue
            section_chunks = self._split_text_by_tokens(
                section_text, section_name, base_metadata
            )
            all_chunks.extend(section_chunks)

        # Update chunk indices to be global
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)

        return all_chunks
