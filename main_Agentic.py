import os, json, uuid
from langchain_groq import ChatGroq
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
# from ServerInteraction import get_db_collection, embed_documents
import streamlit as st
from typing import Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_classic.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from tools import get_all_tools
from langchain_core.messages import ToolMessage,HumanMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from llm import get_llm
from utils import load_yaml_config


# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"


config = load_yaml_config(APP_CONFIG_FPATH)
llm = get_llm(config["llm_openai"])


class State(TypedDict):
    messages: Annotated[list, add_messages]

def visualize_graph(graph: StateGraph, save_path: str):
    """Visualize the graph."""
    png = graph.get_graph().draw_mermaid_png()
    with open(save_path, "wb") as f:
        f.write(png)

def create_tool_registry() -> Dict[str, Any]:
    """Create a registry mapping tool names to their functions."""
    tools = get_all_tools()
    return {tool.name: tool for tool in tools}


def llm_node(state: State):
    """Node that handles LLM invocation."""
    # Get tools and create LLM with tools
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    # Invoke the LLM
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tools_node(state: State):
    """Node that handles tool execution."""
    tool_registry = create_tool_registry()

    # Get the last message (should be from LLM with tool calls)
    last_message = state["messages"][-1]

    tool_messages = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Execute all tool calls
        for tool_call in last_message.tool_calls:
            result = execute_tool_call(tool_call, tool_registry)
            # Create tool message
            tool_message = ToolMessage(
                content=str(result), tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)

    return {"messages": tool_messages}

def should_continue(state: State):
    """Determine whether to continue to tools or end."""
    last_message = state["messages"][-1]

    # If the last message has tool calls, go to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we're done
    return END


def execute_tool_call(tool_call: Dict[str, Any], tool_registry: Dict[str, Any]) -> Any:
    """Execute a single tool call and return the result."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name in tool_registry:
        tool_function = tool_registry[tool_name]
        result = tool_function.invoke(tool_args)
        print(f"🔧 Tool used: {tool_name} with args {tool_args} → Result: {result}")
        return result
    else:
        print(f"Unknown tool: {tool_name}")
        return f"Error: Tool '{tool_name}' not found"


def create_graph():
    """Create and configure the LangGraph workflow."""
    # Create the graph
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)

    # Set entry point
    graph.set_entry_point("llm")

    # Add conditional edges
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})

    # After tools, always go back to LLM
    graph.add_edge("tools", "llm")

    return graph.compile()


# =================================================
# CONTEXT ENGINEERING
# =================================================

prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
SYSTEM_RULES = prompt_config["rag_assistant_prompt"]["role"]
style_or_tone = prompt_config["rag_assistant_prompt"]["style_or_tone"] 
instructions = prompt_config["rag_assistant_prompt"]["instruction"]
output_constraints = prompt_config["rag_assistant_prompt"]["output_constraints"]
output_format = prompt_config["rag_assistant_prompt"]["output_format"]
 
MAX_RECENT_TURNS = 6

def summarize_conversation(messages):
    """
    Lightweight rolling summary.
    In production, replace with LLM-based summarization every N turns.
    """
    if len(messages) < 6:
        return ""

    user_topics = [
        m["content"] for m in messages
        if m["role"] == "user"
    ][-3:]

    return (
        "Recent discussion topics include: "
        + "; ".join(user_topics)
    )

def build_context_messages(chat_messages, user_prompt):
    """
    Builds a clean, ordered context for the LLM
    """

    summary = summarize_conversation(chat_messages)

    context = [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "system", "content": style_or_tone},
        {"role": "system", "content": instructions},
        {"role": "system", "content": output_constraints},
        {"role": "system", "content": output_format},
    ]

    if summary:
        context.append({
            "role": "system",
            "content": f"Conversation summary: {summary}"
        })

    # Only last N turns (short-term memory)
    recent = chat_messages[-MAX_RECENT_TURNS:]
    context.extend(recent)

    # Final user message
    context.append({
        "role": "user",
        "content": user_prompt
    })

    return context
# =================================================


if __name__ == "__main__":
    
    # Create the graph
    app = create_graph()
    
        # Display available tools
    tool_registry = create_tool_registry()
    print(f"Available tools: {', '.join(tool_registry.keys())}\n")
    
     # System message with dynamic tool information
    tool_descriptions = "\n".join(
        [f"- {name}: {tool.description}" for name, tool in tool_registry.items()]
    )
    system_content = f"""You are a helpful AI assistant. Remember the previous messages in this conversation. 

You have access to the following tools:
{tool_descriptions}

Use these tools when appropriate to help answer questions. if you don not find any usefull information then just reply that you dont have any information regarding that."""

    # Initialize conversation state
    initial_state = {"messages": [SystemMessage(content=system_content)]}

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="MY Guide – AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================
# HEADER
# =================================================
st.markdown(
    """
    <h2 style="margin-bottom:0;">🤖 MY Guide AI Assistant</h2>
    <p style="color:gray; margin-top:4px;">
    Data Governance • Cyber Security • Cloud • Generative AI
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =================================================
# FILE STORAGE
# =================================================
CHAT_DIR = "chat_data"
os.makedirs(CHAT_DIR, exist_ok=True)

USERNAME = "default_user"

def user_file():
    return f"{CHAT_DIR}/{USERNAME}.json"

def load_user_data():
    if os.path.exists(user_file()):
        with open(user_file()) as f:
            return json.load(f)
    return {"chats": {}, "active_chat": None}

def save_user_data(data):
    # Auto-delete empty chats
    empty = [cid for cid, c in data["chats"].items() if not c["messages"]]
    for cid in empty:
        del data["chats"][cid]
        if data["active_chat"] == cid:
            data["active_chat"] = None

    with open(user_file(), "w") as f:
        json.dump(data, f, indent=2)

# =================================================
# SESSION STATE
# =================================================
if "user_data" not in st.session_state:
    st.session_state.user_data = load_user_data()

data = st.session_state.user_data

# =================================================
# SIDEBAR – PROFESSIONAL NAV
# =================================================
with st.sidebar:
    st.markdown("### 💬 Conversations")

    if st.button("➕  New conversation", use_container_width=True):
        data["active_chat"] = None
        save_user_data(data)
        st.rerun()

    st.divider()

    chats_sorted = sorted(
        data["chats"].items(),
        key=lambda x: (not x[1].get("pinned", False))
    )

    for cid, c in chats_sorted:
        col1, col2 = st.columns([7, 1])

        with col1:
            if st.button(
                f"{'📌 ' if c.get('pinned') else ''}{c['title']}",
                key=f"chat_{cid}",
                use_container_width=True
            ):
                data["active_chat"] = cid
                save_user_data(data)
                st.rerun()

        with col2:
            with st.popover("⋮"):
                if st.button(
                    "📌 Unpin" if c.get("pinned") else "📌 Pin",
                    key=f"pin_{cid}",
                    use_container_width=True
                ):
                    c["pinned"] = not c.get("pinned", False)
                    save_user_data(data)
                    st.rerun()

                if st.button(
                    "🗑️ Delete",
                    key=f"del_{cid}",
                    use_container_width=True
                ):
                    del data["chats"][cid]
                    if data["active_chat"] == cid:
                        data["active_chat"] = None
                    save_user_data(data)
                    st.rerun()

    st.divider()

    st.markdown(
        """
        **Platform Stack**
        - Groq (LLM Inference)
        - LangGraph (Agent Flow)
        - LangSmith (Tracing)
        - ChromaDB (Vector Store)
        - Streamlit (UI)
        """,
    )

# =================================================
# ACTIVE CHAT
# =================================================
chat_id = data.get("active_chat")

if chat_id and chat_id in data["chats"]:
    chat = data["chats"][chat_id]
    messages = chat["messages"]
else:
    chat = None
    messages = []

# =================================================
# RENAME (CLEAN UX)
# =================================================
if chat:
    with st.expander("✏ Rename conversation", expanded=False):
        new_title = st.text_input(
            "Conversation title",
            chat["title"]
        )
        if new_title and new_title != chat["title"]:
            chat["title"] = new_title[:32]
            save_user_data(data)
            st.success("Title updated")

# =================================================
# CHAT DISPLAY
# =================================================
if chat:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    st.info("💡 Start by asking a question to begin a new conversation.")

if prompt := st.chat_input("Ask your question…"):

    # Lazy chat creation
    if not chat:
        chat_id = str(uuid.uuid4())[:8]
        data["chats"][chat_id] = {
            "title": prompt[:32],
            "messages": [],
            "pinned": False
        }
        data["active_chat"] = chat_id
        chat = data["chats"][chat_id]
        messages = chat["messages"]

    # Save user message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 👉 CONTEXT ENGINEERING HERE
    context_messages = build_context_messages(
        messages[:-1],  # exclude current prompt
        prompt
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Reset LangGraph state cleanly
            initial_state["messages"] = []

            # Inject engineered context
            for m in context_messages:
                if m["role"] == "user":
                    initial_state["messages"].append(
                        HumanMessage(content=m["content"])
                    )
                else:
                    initial_state["messages"].append(
                        SystemMessage(content=m["content"])
                    )

            result = app.invoke(initial_state)
            # print("Result contrext --> ", result)
            response = result["messages"][-1].content

        st.markdown(response)

    messages.append({"role": "assistant", "content": response})
    save_user_data(data)
    st.rerun()
