
import time
import streamlit as st
from app.services.chat_service import ChatService




# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="HandbookGPT",
    page_icon="📘",
    layout="wide"
)

# ----------------------------------------------------
# LOAD CHATBOT (Cached)
# ----------------------------------------------------
@st.cache_resource

def load_chatbot():
    print("NEW CHAT SERVICE CREATED")
    return ChatService()

chatbot = load_chatbot()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.title("📘 HandbookGPT")

    st.success("🟢 System Ready")

    st.markdown("### 🤖 Model")
    st.write("Llama 3.2 3B (Ollama)")

    st.markdown("### 🔍 Retrieval")
    st.write("Hybrid RAG")

    st.markdown("### 🧠 Embeddings")
    st.write("all-MiniLM-L6-v2")

    st.markdown("### 💾 Vector DB")
    st.write("ChromaDB")

    st.markdown("### 📄 Document")
    st.write("Employee Handbook")

    st.markdown("### 📚 Indexed Chunks")
    st.write("165")
    

    st.markdown("### ⚡ Retrieval Pipeline")
    st.write("✅ Semantic Search")
    st.write("✅ BM25 Keyword Search")
    st.write("✅ CrossEncoder Reranker")

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------
st.title("📘  Employee Handbook Assistant")

st.caption(
    "AI-powered Employee Handbook Assistant using Hybrid Retrieval (Semantic + BM25 + CrossEncoder Reranking)"
)

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------
# WELCOME MESSAGE
# ----------------------------------------------------
if len(st.session_state.messages) == 0:

    st.info("""
### 👋 Welcome!

Ask anything about your employee handbook.

#### Example Questions

- Can I work from home?
- What is the dress code?
- How many annual leave days do I get?
- What are the working hours?
""")

# ----------------------------------------------------
# DISPLAY CHAT HISTORY
# ----------------------------------------------------
for message in st.session_state.messages:

    avatar = "👤" if message["role"] == "user" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            st.caption(
                 f"Generated in {message['response_time']:.2f} seconds"
            )

            # ---------------- Pipeline Timing ----------------
            if message.get("debug"):

                with st.expander("📊  Performance Metrics"):

                    debug = message["debug"]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Retrieval", f"{debug.get('retrieval', 0):.3f} s")
                        st.metric("Prompt", f"{debug.get('prompt', 0):.3f} s")

                    with col2:
                        st.metric("LLM", f"{debug.get('llm', 0):.3f} s")
                        st.metric("Total", f"{debug.get('total', 0):.3f} s")

            # ---------------- Sources ----------------
            with st.expander("📄 Sources"):

                for source in message["sources"]:

                    st.markdown(f"**📄 Page {source['page']}**")

                    text = source["text"]

                    if len(text) > 350:
                        text = text[:350] + "..."

                    st.info(text)

# ----------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------
question = st.chat_input("Ask a question about the handbook...")

if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):

        start = time.time()

     

        with st.spinner("🔍 Searching the handbook and generating an answer..."):
            response = chatbot.ask(question)

        response_time = time.time() - start

        

        try:
          
            
            st.subheader("Answer")
            st.markdown(response["answer"])

            st.caption(
                f"Generated in {response_time:.2f} seconds"
            )

            # ---------------- Pipeline Timing ----------------
            if response.get("debug"):

                with st.expander("📊 Pipeline Timing"):

                    debug = response["debug"]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Retrieval", f"{debug.get('retrieval', 0):.3f} s")
                        st.metric("Prompt", f"{debug.get('prompt', 0):.3f} s")

                    with col2:
                        st.metric("LLM", f"{debug.get('llm', 0):.3f} s")
                        st.metric("Total", f"{debug.get('total', 0):.3f} s")

            # ---------------- Sources ----------------
            with st.expander("📄 Sources"):

                for source in response["sources"]:

                    st.markdown(f"### 📄 Page {source['page']}")
                    st.info(source["text"][:350] + "...")

            # Save assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"],
                    "response_time": response_time,
                    "debug": response.get("debug", {})
                }
            )

        except Exception as e:

         

            st.error(f"❌ {e}")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.divider()

st.caption(
    "🚀 Built with Streamlit • Ollama • ChromaDB • Sentence Transformers • Hybrid Retrieval (Semantic + BM25 + crossEncoder Reranking)"
)