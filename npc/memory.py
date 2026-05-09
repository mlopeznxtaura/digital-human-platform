"""
LlamaIndex + Chroma long-term episodic memory for NPC characters.
Store and retrieve conversation memories with semantic search.
SDKs: LlamaIndex, ChromaDB, Ollama embeddings
"""
import os
import time
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not available. Install: pip install chromadb")

try:
    from llama_index.core import VectorStoreIndex, Document, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.embeddings import resolve_embed_model
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    print("Warning: llama-index not available. Install: pip install llama-index")


@dataclass
class Memory:
    id: str
    content: str
    session_id: str
    character: str
    timestamp: float
    importance: float = 1.0
    emotion: str = "neutral"


class NPCMemoryStore:
    """
    Long-term episodic memory for NPC characters.
    Stores conversation summaries and retrieves semantically relevant memories.
    Uses ChromaDB as vector store, LlamaIndex for indexing.
    """

    def __init__(
        self,
        character_name: str,
        persist_dir: str = "./memory_store",
        embedding_model: str = "local:BAAI/bge-small-en-v1.5",
        collection_prefix: str = "npc_memory",
    ):
        self.character_name = character_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._memories: List[Memory] = []     # In-memory fallback
        self._index = None
        self._collection = None

        if CHROMA_AVAILABLE and LLAMA_INDEX_AVAILABLE:
            self._init_chroma(collection_prefix, embedding_model)
        else:
            print(f"[Memory] Running in-memory fallback (Chroma/LlamaIndex not available)")

    def _init_chroma(self, collection_prefix: str, embedding_model: str):
        """Initialize ChromaDB + LlamaIndex vector store."""
        try:
            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collection_name = f"{collection_prefix}_{self.character_name}"
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            chroma_store = ChromaVectorStore(chroma_collection=self._collection)
            storage_context = StorageContext.from_defaults(vector_store=chroma_store)
            embed_model = resolve_embed_model(embedding_model)
            self._index = VectorStoreIndex(
                [], storage_context=storage_context, embed_model=embed_model
            )
            print(f"[Memory] ChromaDB initialized: {collection_name} | {self._collection.count()} memories")
        except Exception as e:
            print(f"[Memory] Chroma init failed: {e}. Using in-memory fallback.")

    def store(
        self,
        content: str,
        session_id: str = "global",
        emotion: str = "neutral",
        importance: float = 1.0,
    ) -> Memory:
        """Store a memory. Embeds and indexes for semantic retrieval."""
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            session_id=session_id,
            character=self.character_name,
            timestamp=time.time(),
            emotion=emotion,
            importance=importance,
        )
        self._memories.append(memory)

        if self._index is not None:
            try:
                doc = Document(
                    text=content,
                    metadata={
                        "id": memory.id,
                        "session_id": session_id,
                        "character": self.character_name,
                        "timestamp": memory.timestamp,
                        "emotion": emotion,
                        "importance": importance,
                    },
                )
                self._index.insert(doc)
            except Exception as e:
                print(f"[Memory] Index insert failed: {e}")

        return memory

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> List[str]:
        """Retrieve semantically relevant memories for a query."""
        if self._index is not None:
            try:
                retriever = self._index.as_retriever(similarity_top_k=top_k)
                nodes = retriever.retrieve(query)
                results = []
                for node in nodes:
                    meta = node.metadata
                    if min_importance and meta.get("importance", 1.0) < min_importance:
                        continue
                    if session_id and meta.get("session_id") != session_id:
                        if meta.get("session_id") != "global":
                            continue
                    results.append(node.text)
                return results
            except Exception as e:
                print(f"[Memory] Retrieval failed: {e}")

        # In-memory fallback: simple keyword match
        query_words = set(query.lower().split())
        scored = []
        for m in self._memories:
            if session_id and m.session_id != session_id and m.session_id != "global":
                continue
            if m.importance < min_importance:
                continue
            mem_words = set(m.content.lower().split())
            score = len(query_words & mem_words) / (len(query_words) + 1e-6)
            scored.append((score, m.content))
        scored.sort(reverse=True)
        return [c for _, c in scored[:top_k] if _ > 0]

    def get_session_memories(self, session_id: str) -> List[Memory]:
        """Get all memories from a specific session."""
        return [m for m in self._memories if m.session_id == session_id]

    def clear_session(self, session_id: str):
        """Remove all memories from a session."""
        self._memories = [m for m in self._memories if m.session_id != session_id]

    def summarize_session(self, session_id: str, llm=None) -> str:
        """Summarize a session's memories into a compact narrative."""
        mems = self.get_session_memories(session_id)
        if not mems:
            return ""
        combined = "
".join(m.content for m in mems[-20:])
        if llm:
            try:
                from langchain_core.messages import HumanMessage
                prompt = f"Summarize these conversation memories in 2-3 sentences:
{combined}"
                return llm.invoke([HumanMessage(content=prompt)]).content
            except Exception:
                pass
        return f"Session {session_id}: {len(mems)} memories recorded."

    def stats(self) -> Dict[str, Any]:
        return {
            "character": self.character_name,
            "total_memories": len(self._memories),
            "chroma_count": self._collection.count() if self._collection else 0,
            "sessions": len(set(m.session_id for m in self._memories)),
        }
