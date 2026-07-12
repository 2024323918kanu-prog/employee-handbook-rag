import time
import traceback

from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.prompt_builder import PromptBuilder
from app.llm.ollama_client import OllamaClient


class ChatService:

    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = OllamaClient()

    def ask(self, question):

        timings = {}

        try:
            # -------------------------
            # Total Start
            # -------------------------
            total_start = time.perf_counter()

            # -------------------------
            # Retrieval
            # -------------------------
            start = time.perf_counter()

            retrieved = self.retriever.search(question)
            sources = retrieved["results"]
            timings["retrieval"] = round(
                time.perf_counter() - start,
                3
            )

            # -------------------------
            # Prompt Building
            # -------------------------
            start = time.perf_counter()

            prompt = PromptBuilder.build(
                sources,
                question
            )

            timings["prompt"] = round(
                time.perf_counter() - start,
                3
            )

            # -------------------------
            # LLM Generation
            # -------------------------
            start = time.perf_counter()

            answer = self.llm.generate(prompt)

            timings["llm"] = round(
                time.perf_counter() - start,
                6
            )

            # -------------------------
            # Total Time
            # -------------------------
            timings["total"] = round(
                time.perf_counter() - total_start,
                3
            )

            # -------------------------
            # Console Timing
            # -------------------------
            print("\n========== Pipeline Timing ==========")
            print(f"Retrieval : {timings['retrieval']} sec")
            print(f"Prompt    : {timings['prompt']} sec")
            print(f"LLM       : {timings['llm']} sec")
            print(f"Total     : {timings['total']} sec")
            print("====================================\n")
                       
            return {
                "answer": answer,
                "sources": sources,
                "debug": timings
            }

        except Exception as e:

            import traceback as tb
            import os

            error_text = tb.format_exc()

            print("!!!!!! EXCEPTION CAUGHT IN CHAT_SERVICE !!!!!!")
            print(error_text)

            log_path = os.path.join(os.getcwd(), "last_error.log")
            print("WRITING LOG TO:", log_path)

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(error_text)

            return {
                "answer": f"Error: {e}",
                "sources": [],
                "debug": timings
            }