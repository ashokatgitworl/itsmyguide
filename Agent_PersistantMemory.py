import os, json, uuid
import chromadb
from chromadb.config import Settings
from datetime import datetime

import streamlit as st
from typing import Dict, Any, Annotated
from typing_extensions import TypedDict

from langchain_classic.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage

from tools import get_all_tools,get_db_collection
from llm import get_llm
from utils import load_yaml_config
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH


os.environ["TOKENIZERS_PARALLELISM"] = "false"
config = load_yaml_config(APP_CONFIG_FPATH)
llm = get_llm(config["llm_openai"])

USERNAME = "default_user"

def format_memory_for_prompt(memories):
    """
    Convert retrieved long-term memories into a system prompt block
    """
    if not memories:
        return None

    lines = ["🧠 Relevant long-term memory:"]
    for m in memories:
        lines.append(f"- {m}")

    return "\n".join(lines)

def store_memory(text: str):
    # collection = get_db_collection()
    collection = get_db_collection(collection_name="long_term_memory")

    collection.add(
        documents=[text],
        ids=[str(uuid.uuid4())],
        metadatas=[{
            "type": "long_term_memory",
            "user_id": USERNAME,
            "created_at": datetime.utcnow().isoformat()
        }]
    )

    print("🧠 Stored memory")
    print("📊 Count:", collection.count())

def retrieve_memory(query: str, top_k=5):
    # collection = get_db_collection()
    collection = get_db_collection(collection_name="long_term_memory")

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"user_id": USERNAME}
    )

    return results["documents"][0] if results["documents"] else []


# =================================================
# LANGGRAPH
# =================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

def llm_node(state: State):
    llm_with_tools = llm.bind_tools(get_all_tools())
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tools_node(state: State):
    registry = {t.name: t for t in get_all_tools()}
    last = state["messages"][-1]
    tool_messages = []

    if hasattr(last, "tool_calls") and last.tool_calls:
        for call in last.tool_calls:
            fn = registry.get(call["name"])
            if fn:
                result = fn.invoke(call["args"])
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=call["id"])
                )
    return {"messages": tool_messages}

def should_continue(state: State):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

def create_graph():
    graph = StateGraph(State)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile()

app = create_graph()
initial_state = {"messages": []}

# =================================================
# CONTEXT ENGINEERING
# =================================================
prompt_cfg = load_yaml_config(PROMPT_CONFIG_FPATH)["rag_assistant_prompt"]

SYSTEM_RULES = prompt_cfg["role"]
STYLE = prompt_cfg["style_or_tone"]
INSTRUCTIONS = prompt_cfg["instruction"]
CONSTRAINTS = prompt_cfg["output_constraints"]
FORMAT = prompt_cfg["output_format"]

MAX_RECENT_TURNS = 6

def build_context(history, user_prompt):
    context = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "system", "content": STYLE},
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "system", "content": CONSTRAINTS},
        {"role": "system", "content": FORMAT},
    ]
    context.extend(history[-MAX_RECENT_TURNS:])
    context.append({"role": "user", "content": user_prompt})
    return context


# =====================================================================temp 

# =================================================
# STREAMLIT CONFIGURATION
# =================================================
st.set_page_config(
    page_title="MY Guide – AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("## 🤖 MY Guide AI Assistant")
st.caption("Your intelligent conversational AI assistant")

# =================================================
# CHAT STORAGE
# =================================================
CHAT_DIR = "chat_data"
os.makedirs(CHAT_DIR, exist_ok=True)

def get_user_chat_file() -> str:
    return f"{CHAT_DIR}/{USERNAME}.json"

def load_chat_data() -> dict:
    if os.path.exists(get_user_chat_file()):
        with open(get_user_chat_file(), "r") as f:
            return json.load(f)
    return {
        "chats": {},
        "active_chat": None
    }

def save_chat_data(data: dict) -> None:
    with open(get_user_chat_file(), "w") as f:
        json.dump(data, f, indent=2)

if "data" not in st.session_state:
    st.session_state.data = load_chat_data()

if "rename_chat_id" not in st.session_state:
    st.session_state.rename_chat_id = None

data = st.session_state.data

# =================================================
# SIDEBAR – CHAT MANAGEMENT
# =================================================
def delete_chat(chat_id: str) -> None:
    if chat_id in data["chats"]:
        del data["chats"][chat_id]
        if data.get("active_chat") == chat_id:
            data["active_chat"] = None
        save_chat_data(data)

def rename_chat(chat_id: str, new_title: str) -> None:
    if chat_id in data["chats"] and new_title.strip():
        data["chats"][chat_id]["title"] = new_title.strip()
        save_chat_data(data)

with st.sidebar:
    st.markdown("### 💬 Conversations")

    if st.button("➕ New chat", use_container_width=True):
        data["active_chat"] = None
        save_chat_data(data)
        st.rerun()

    st.divider()

    for chat_id, chat in data["chats"].items():
        col_title, col_rename, col_delete = st.columns([0.70, 0.15, 0.15])

        # -------------------------------
        # CHAT TITLE (VIEW / EDIT MODE)
        # -------------------------------
        with col_title:
            if st.session_state.rename_chat_id == chat_id:
                with st.form(key=f"rename_form_{chat_id}"):
                    new_title = st.text_input(
                        "Rename chat",
                        value=chat["title"],
                        key=f"rename_input_{chat_id}",
                        label_visibility="collapsed"
                    )

                    submitted = st.form_submit_button(
                        "Save",
                        key=f"rename_submit_{chat_id}"
                    )

                    if submitted:
                        rename_chat(chat_id, new_title)
                        st.session_state.rename_chat_id = None
                        st.rerun()
            else:
                if st.button(
                    chat["title"],
                    key=f"chat_select_{chat_id}",
                    use_container_width=True
                ):
                    data["active_chat"] = chat_id
                    save_chat_data(data)
                    st.rerun()

        # -------------------------------
        # RENAME BUTTON
        # -------------------------------
        with col_rename:
            if st.button(
                "✏️",
                key=f"chat_rename_btn_{chat_id}"
            ):
                st.session_state.rename_chat_id = chat_id
                st.rerun()

        # -------------------------------
        # DELETE BUTTON
        # -------------------------------
        with col_delete:
            if st.button(
                "🗑️",
                key=f"chat_delete_{chat_id}"
            ):
                delete_chat(chat_id)
                st.rerun()

# =================================================
# ACTIVE CHAT
# =================================================
active_chat_id = data.get("active_chat")
active_chat = data["chats"].get(active_chat_id) if active_chat_id else None
messages = active_chat["messages"] if active_chat else []

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =================================================
# USER INPUT & LLM PROCESSING
# =================================================
if prompt := st.chat_input("Ask anything…"):

    # Create new chat if none exists
    if not active_chat:
        active_chat_id = str(uuid.uuid4())[:8]
        data["chats"][active_chat_id] = {
            "title": prompt[:32],
            "messages": []
        }
        data["active_chat"] = active_chat_id
        active_chat = data["chats"][active_chat_id]
        messages = active_chat["messages"]

    # User message
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Memory retrieval
    memories = retrieve_memory(prompt)
    memory_block = format_memory_for_prompt(memories)

    # Context construction
    context = build_context(messages[:-1], prompt)

    if memory_block:
        context.insert(0, {
            "role": "system",
            "content": memory_block
        })

    initial_state["messages"] = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else SystemMessage(content=m["content"])
        for m in context
    ]

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke(initial_state)
            response = result["messages"][-1].content
            st.markdown(response)

    messages.append({
        "role": "assistant",
        "content": response
    })

    # Memory persistence condition
    if "we use" in prompt.lower() or "we are using" in prompt.lower():
        store_memory(prompt)

    save_chat_data(data)
    st.rerun()
