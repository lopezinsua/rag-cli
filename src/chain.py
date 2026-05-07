from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .retriever import load_retriever

PROMPTS_DIR = Path("prompts")
MAX_HISTORY_TURNS = 6


def _load_system_prompt() -> str:
    return (PROMPTS_DIR / "qa.txt").read_text(encoding="utf-8").strip()


def _source_label(doc) -> str:
    return f"p.{doc.metadata.get('page', '?')}, chunk {doc.metadata.get('chunk', '?')}"


def chat_loop(pdf_path: Path) -> None:
    retriever = load_retriever(pdf_path.stem)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    system_prompt = _load_system_prompt()
    history: list[tuple[str, str]] = []

    print(f"Chateando con '{pdf_path.name}'. Escribe 'exit' para salir.\n")

    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "salir"):
            print("Hasta luego.")
            break

        docs = retriever.invoke(question)
        context = "\n\n".join(
            f"[{_source_label(d)}]\n{d.page_content}" for d in docs
        )
        sources = " | ".join(_source_label(d) for d in docs)

        messages: list = [SystemMessage(content=system_prompt)]
        for q, a in history[-MAX_HISTORY_TURNS:]:
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
        messages.append(HumanMessage(
            content=f"Contexto del documento:\n{context}\n\nPregunta: {question}"
        ))

        answer_parts: list[str] = []
        for chunk in llm.stream(messages):
            text = chunk.content
            print(text, end="", flush=True)
            answer_parts.append(text)

        answer = "".join(answer_parts)
        print(f"\n[Fuente: {sources}]\n")
        history.append((question, answer))
