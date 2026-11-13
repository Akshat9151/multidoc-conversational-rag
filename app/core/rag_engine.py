import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.core.vector_store import VectorStoreManager
from app.core.memory import ConversationMemoryManager, memory_manager
from app.core.text_splitter import DocumentChunk
from app.models.schemas import Citation, QueryResponse
from app.config import settings

logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT = """You are an accurate, enterprise-grade AI Document Assistant.
Your objective is to answer the user's question accurately, concisely, and strictly based on the provided document excerpts.

CRITICAL INSTRUCTIONS:
1. GROUNDING: Base your answer ONLY on the provided context excerpts. Do not fabricate, assume, or extrapolate facts beyond what is stated.
2. UNAVAILABLE INFORMATION: If the provided excerpts do not contain the answer, you MUST explicitly respond:
   "Based on the uploaded documents, I could not find information regarding [rephrase user query]."
   Do NOT attempt to use external pre-trained knowledge if the documents do not support it.
3. CITATIONS: Whenever stating a fact, refer to the document name and page number, e.g. (Source: report.pdf, Page 2).
4. CONVERSATIONAL AWARENESS: Consider the chat history when resolving pronouns (it, that, they, previous figure).
"""

QUERY_REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up question to be a standalone, self-contained search query.
Do NOT answer the question. Only return the reformulated query text. If the question is already self-contained, return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Search Query:"""


class LLMClient:
    """Pluggable LLM interface supporting Google Gemini, Groq, and an offline synthesis fallback."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.gemini_key = settings.GEMINI_API_KEY
        self.groq_key = settings.GROQ_API_KEY
        self._gemini_model = None
        self._groq_client = None

        self._init_clients()

    def _init_clients(self):
        # Gemini Init
        if self.gemini_key and self.gemini_key != "your_google_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self._gemini_model = genai.GenerativeModel(settings.LLM_MODEL)
                logger.info("Initialized Google Gemini LLM client.")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")

        # Groq Init
        if self.groq_key and self.groq_key != "your_groq_api_key_here":
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_key)
                logger.info("Initialized Groq LLM client.")
            except Exception as e:
                logger.warning(f"Could not initialize Groq: {e}")

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Dispatches prompt generation to configured provider with offline fallback."""
        # 1. Try Gemini
        if self._gemini_model is not None:
            try:
                full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": settings.TEMPERATURE,
                        "max_output_tokens": settings.MAX_OUTPUT_TOKENS,
                    },
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")

        # 2. Try Groq
        if self._groq_client is not None:
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                resp = self._groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_OUTPUT_TOKENS,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Groq generation error: {e}")

        # 3. Intelligent Offline Fallback (Ensures zero runtime crashes if API keys are pending)
        logger.info("Using grounded extractive synthesis fallback (offline mode).")
        return self._offline_synthesis(prompt)

    def _offline_synthesis(self, prompt: str) -> str:
        """
        Extractive synthesis engine used when API key is missing.
        Parses the injected context chunks and produces an organized, cited summary.
        """
        if "No relevant context found" in prompt or "CONTEXT EXCERPTS:" not in prompt:
            return "Based on the uploaded documents, I could not find information regarding this query."

        context_part = prompt.split("CONTEXT EXCERPTS:")[-1].split("USER QUESTION:")[0].strip()
        lines = [line.strip() for line in context_part.splitlines() if line.strip()]

        synthesis_points = []
        for line in lines[:8]:
            if line.startswith("[Source:") or line.startswith("==="):
                continue
            if len(line) > 40:
                synthesis_points.append(f"• {line}")

        if not synthesis_points:
            return "Based on the uploaded documents, I could not find information regarding this query."

        return (
            "Based on the provided documents, here is the relevant grounded information:\n\n"
            + "\n".join(synthesis_points[:4])
            + "\n\n*(Note: Configure GEMINI_API_KEY in .env for full conversational LLM generation)*"
        )


class RAGEngine:
    """
    Complete Retrieval-Augmented Generation pipeline:
    - Contextual Query Reformulation
    - Top-K Vector Search with Cosine Similarity Filtering
    - Grounded Answer Generation
    - Structured Source Citations & Conversational Memory Tracking
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        memory: Optional[ConversationMemoryManager] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.vector_store = vector_store
        self.memory = memory or memory_manager
        self.llm = llm_client or LLMClient()

    def rewrite_query_with_history(self, session_id: str, question: str) -> str:
        """
        Contextualizes follow-up questions using recent chat memory.
        If user asks 'How does that compare to Q2?', rewrites to standalone subject.
        """
        history_text = self.memory.format_history_for_prompt(session_id, max_turns=3)
        if not history_text:
            return question

        # Pronoun & referential heuristic check
        referential_markers = [
            "that", "it", "they", "those", "previous", "earlier", "this", "compare",
            "what about", "how much", "and why", "explain more", "which one"
        ]
        needs_rewrite = any(m in question.lower() for m in referential_markers)

        if not needs_rewrite:
            return question

        prompt = QUERY_REWRITE_PROMPT.format(chat_history=history_text, question=question)
        rewritten = self.llm.generate(prompt)

        # Cleanup in case LLM added conversational prefix
        cleaned = rewritten.replace('"', '').replace("Standalone Search Query:", "").strip()
        logger.info(f"Rewrote query: '{question}' -> '{cleaned}'")
        return cleaned if cleaned else question

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_document: Optional[str] = None,
    ) -> QueryResponse:
        """
        Executes the full RAG pipeline for a user question.
        """
        start_time = time.time()
        active_session = self.memory.get_or_create_session(session_id)
        k = top_k or settings.TOP_K_RETRIEVAL

        # 1. Query Reformulation (Contextualization)
        rewritten_query = self.rewrite_query_with_history(active_session, question)

        # 2. Similarity Search & Retrieval
        retrieved_items: List[Tuple[DocumentChunk, float]] = (
            self.vector_store.similarity_search_with_score(
                query=rewritten_query, top_k=k, filter_doc=filter_document
            )
        )

        # 3. Evaluate Similarity Threshold
        valid_items = [
            (chunk, score)
            for chunk, score in retrieved_items
            if score >= settings.SIMILARITY_SCORE_THRESHOLD
        ]

        # 4. Handle Case: Information Unavailable in Documents
        if not valid_items:
            unavailable_msg = (
                f"Based on the uploaded documents, I could not find information "
                f"regarding '{question}'."
            )
            # Record turn in memory
            self.memory.add_user_message(active_session, question)
            self.memory.add_assistant_message(active_session, unavailable_msg, citations=[])

            latency = (time.time() - start_time) * 1000.0
            return QueryResponse(
                answer=unavailable_msg,
                session_id=active_session,
                grounded=False,
                citations=[],
                rewritten_query=rewritten_query if rewritten_query != question else None,
                latency_ms=round(latency, 2),
            )

        # 5. Build Structured Citations
        citations: List[Citation] = []
        context_blocks: List[str] = []

        for chunk, score in valid_items:
            doc_name = chunk.metadata.get("document_name", "unknown.pdf")
            page_num = chunk.metadata.get("page_number", 1)
            snippet = chunk.text[:250].strip() + ("..." if len(chunk.text) > 250 else "")

            citation = Citation(
                document_name=doc_name,
                page_number=page_num,
                snippet=snippet,
                relevance_score=round(score, 3),
            )
            citations.append(citation)

            block = f"[Source: {doc_name} | Page: {page_num}]\n{chunk.text}"
            context_blocks.append(block)

        # 6. Format LLM Grounded Prompt
        chat_history_str = self.memory.format_history_for_prompt(active_session, max_turns=3)
        joined_context = "\n\n---\n\n".join(context_blocks)

        user_prompt = f"""CONTEXT EXCERPTS:
{joined_context}

RECENT CONVERSATION HISTORY:
{chat_history_str if chat_history_str else "No previous conversation."}

USER QUESTION:
{question}

Provide an accurate, grounded response with document citations:"""

        # 7. Generate Answer via LLM
        answer = self.llm.generate(prompt=user_prompt, system_instruction=GROUNDED_SYSTEM_PROMPT)

        # 8. Update Conversational Memory
        self.memory.add_user_message(active_session, question)
        self.memory.add_assistant_message(active_session, answer, citations=citations)

        latency = (time.time() - start_time) * 1000.0

        return QueryResponse(
            answer=answer,
            session_id=active_session,
            grounded=True,
            citations=citations,
            rewritten_query=rewritten_query if rewritten_query != question else None,
            latency_ms=round(latency, 2),
        )
