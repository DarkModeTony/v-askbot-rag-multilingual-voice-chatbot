import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    strategy: str
    char_count: int
    word_count: int
    parent_id: Optional[str] = None
    window_context: Optional[str] = None

class VastChunkingEngine:
    """
    Advanced Vast Chunking Engine implementing 5 distinct chunking strategies:
    1. Fixed-Size with Sliding Overlap (Baseline)
    2. Recursive Semantic Structure (Sentence/Paragraph boundaries)
    3. Metadata-Aware Contextual Chunking (Provenance & hierarchy enriched)
    4. Parent-Document / Hierarchical Chunking (Child vectors -> Parent context)
    5. Contextual Sentence-Window Chunking (Focal sentence with window span)
    """

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentence_endings = re.compile(r'(?<=[.?!|।])\s+')
        sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]
        if not sentences:
            sentences = [text.strip()]
        return sentences

    def chunk_fixed_size(self, text: str, metadata: Dict[str, Any], chunk_size: int = 150, overlap: int = 30) -> List[Chunk]:
        text = self._clean_text(text)
        words = text.split(' ')
        chunks = []
        doc_id = metadata.get('doc_id', 'doc')

        if len(words) <= chunk_size:
            chunk_text = ' '.join(words)
            return [Chunk(
                chunk_id=f"{doc_id}_fixed_0",
                text=chunk_text,
                metadata=metadata,
                strategy="fixed_size_overlap",
                char_count=len(chunk_text),
                word_count=len(words)
            )]

        step = max(1, chunk_size - overlap)
        chunk_idx = 0
        for i in range(0, len(words), step):
            window_words = words[i:i + chunk_size]
            if not window_words:
                break
            chunk_text = ' '.join(window_words)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_fixed_{chunk_idx}",
                text=chunk_text,
                metadata={**metadata, "chunk_index": chunk_idx, "start_token": i, "end_token": i + len(window_words)},
                strategy="fixed_size_overlap",
                char_count=len(chunk_text),
                word_count=len(window_words)
            ))
            chunk_idx += 1
            if i + chunk_size >= len(words):
                break

        return chunks

    def chunk_recursive_semantic(self, text: str, metadata: Dict[str, Any], target_chunk_words: int = 40) -> List[Chunk]:
        text = self._clean_text(text)
        sentences = self._split_sentences(text)
        chunks = []
        doc_id = metadata.get('doc_id', 'doc')

        current_chunk_sentences = []
        current_word_count = 0
        chunk_idx = 0

        for sentence in sentences:
            sent_words = len(sentence.split())
            if current_word_count + sent_words > target_chunk_words and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_semantic_{chunk_idx}",
                    text=chunk_text,
                    metadata={**metadata, "chunk_index": chunk_idx, "sentence_count": len(current_chunk_sentences)},
                    strategy="recursive_semantic",
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split())
                ))
                chunk_idx += 1
                current_chunk_sentences = [sentence]
                current_word_count = sent_words
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += sent_words

        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_semantic_{chunk_idx}",
                text=chunk_text,
                metadata={**metadata, "chunk_index": chunk_idx, "sentence_count": len(current_chunk_sentences)},
                strategy="recursive_semantic",
                char_count=len(chunk_text),
                word_count=len(chunk_text.split())
            ))

        return chunks

    def chunk_metadata_aware(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        text = self._clean_text(text)
        doc_id = metadata.get('doc_id', 'doc')
        lang = metadata.get('lang', 'en')
        domain = metadata.get('domain', 'general')
        query = metadata.get('query', '')

        header = f"[Domain: {domain.upper()} | Lang: {lang} | Provenance: MSMARCO-XI]"
        semantic_chunks = self.chunk_recursive_semantic(text, metadata, target_chunk_words=35)
        metadata_chunks = []
        for idx, sc in enumerate(semantic_chunks):
            augmented_text = f"{header} [Part {idx+1}/{len(semantic_chunks)}]\n{sc.text}"
            metadata_chunks.append(Chunk(
                chunk_id=f"{doc_id}_meta_{idx}",
                text=augmented_text,
                metadata={
                    **metadata,
                    "domain": domain,
                    "lang": lang,
                    "has_metadata_header": True,
                    "chunk_index": idx,
                    "total_chunks": len(semantic_chunks),
                    "associated_query": query
                },
                strategy="metadata_aware",
                char_count=len(augmented_text),
                word_count=len(augmented_text.split())
            ))
        return metadata_chunks

    def chunk_parent_document(self, text: str, metadata: Dict[str, Any], child_words: int = 20) -> List[Chunk]:
        text = self._clean_text(text)
        parent_id = metadata.get('doc_id', 'parent_doc')
        sentences = self._split_sentences(text)
        chunks = []

        for idx, sentence in enumerate(sentences):
            chunks.append(Chunk(
                chunk_id=f"{parent_id}_child_{idx}",
                text=sentence,
                metadata={
                    **metadata,
                    "is_child_chunk": True,
                    "parent_id": parent_id,
                    "child_index": idx,
                    "full_parent_passage": text
                },
                strategy="parent_document_hierarchical",
                char_count=len(sentence),
                word_count=len(sentence.split()),
                parent_id=parent_id
            ))
        return chunks

    def chunk_contextual_sentence_window(self, text: str, metadata: Dict[str, Any], window_size: int = 1) -> List[Chunk]:
        text = self._clean_text(text)
        doc_id = metadata.get('doc_id', 'doc')
        sentences = self._split_sentences(text)
        chunks = []

        for i, sentence in enumerate(sentences):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            window_ctx = ' '.join(sentences[start:end])

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_window_{i}",
                text=sentence,
                metadata={
                    **metadata,
                    "window_size": window_size,
                    "focal_sentence_index": i,
                    "window_range": [start, end]
                },
                strategy="contextual_sentence_window",
                char_count=len(sentence),
                word_count=len(sentence.split()),
                window_context=window_ctx
            ))
        return chunks

    def chunk_all_strategies(self, text: str, metadata: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "fixed_size_overlap": [asdict(c) for c in self.chunk_fixed_size(text, metadata)],
            "recursive_semantic": [asdict(c) for c in self.chunk_recursive_semantic(text, metadata)],
            "metadata_aware": [asdict(c) for c in self.chunk_metadata_aware(text, metadata)],
            "parent_document_hierarchical": [asdict(c) for c in self.chunk_parent_document(text, metadata)],
            "contextual_sentence_window": [asdict(c) for c in self.chunk_contextual_sentence_window(text, metadata)]
        }

    def evaluate_strategies(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        strategies = self.chunk_all_strategies(text, metadata)
        analytics = {}
        for name, chunks in strategies.items():
            counts = [c['word_count'] for c in chunks]
            avg_words = sum(counts) / len(counts) if counts else 0
            variance = sum((x - avg_words) ** 2 for x in counts) / len(counts) if counts else 0
            analytics[name] = {
                "num_chunks": len(chunks),
                "avg_words_per_chunk": round(avg_words, 1),
                "std_dev": round(variance ** 0.5, 2),
                "total_characters": sum(c['char_count'] for c in chunks),
                "granularity": "High" if len(chunks) > 4 else ("Medium" if len(chunks) > 2 else "Broad"),
                "best_use_case": {
                    "fixed_size_overlap": "Standard general-purpose baseline indexing",
                    "recursive_semantic": "Preserving natural grammatical sentences and paragraphs",
                    "metadata_aware": "Multi-domain Indic RAG with provenance lineage",
                    "parent_document_hierarchical": "High-precision retrieval with broad generation context",
                    "contextual_sentence_window": "Fine-grained focal fact matching with contextual cushion"
                }.get(name, "")
            }
        return {"strategies": strategies, "analytics": analytics}
