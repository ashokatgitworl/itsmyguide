import os
# import logging
from langchain_groq import ChatGroq
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from ServerInteraction import get_db_collection, embed_documents
import streamlit as st
from typing import Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_classic.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from tools import get_all_tools
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from llm import get_llm
from utils import load_yaml_config


# To avoid tokenizer parallelism warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"


config = load_yaml_config(APP_CONFIG_FPATH)
llm = get_llm(config["llm_anthropic"])

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
    
      # Page configuration
    st.set_page_config(
        page_title="ASK ME AI Chatbot Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main title and description
    st.title("🤖 MY Guide AI Chatbot Assistant")
    st.markdown("Ask me about the Data goverance, Cyber Security,CLoud computing , Genrative AI  and I'll help you with intelligent responses!")
  
    
 # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

     # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # response = get_ai_responseSLIT(prompt, llm)
           # Add user message to state
                initial_state["messages"].append(HumanMessage(content=prompt))

                            # Run the graph
                result = app.invoke(initial_state)
                # Update state with results
                initial_state["messages"] = result["messages"]
                last_message = result["messages"][-1]
                response = last_message.content
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

   # Sidebar for additional features
    with st.sidebar:
        st.markdown("### About This App")
        st.markdown("This AI chatbot is powered by:")
        st.markdown("- **Groq** for fast LLM inference")
        st.markdown("- **LangChain** for AI orchestration")
        st.markdown("- **LangSmith** for Monitoring & Evaluation")
        st.markdown("- **Railway** for vector databse hosting")
        st.markdown("- **ChromaDB** for vector database management")
        st.markdown("- **Streamlit** for the beautiful UI")

        st.markdown("---")
        st.markdown("### Chat Controls")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### Stats")
        st.metric("Messages in Chat", len(st.session_state.messages))

        if st.session_state.messages:
            user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            st.metric("Questions Asked", user_messages)

# Enable below code for debugging
# def main():

#     print("LangGraph Chatbot with Custom Tools")
#     print("Type 'exit' or 'quit' to end the session.")

#     # Create the graph
#     app = create_graph()

#     # Display available tools
#     tool_registry = create_tool_registry()
#     print(f"Available tools: {', '.join(tool_registry.keys())}\n")

#     # System message with dynamic tool information
#     tool_descriptions = "\n".join(
#         [f"- {name}: {tool.description}" for name, tool in tool_registry.items()]
#     )
#     system_content = f"""You are a helpful AI assistant. Remember the previous messages in this conversation. 

# You have access to the following tools:
# {tool_descriptions}

# Use these tools when appropriate to help answer questions. if you don not find any usefull information then just reply that you dont have any information regarding that."""

#     # Initialize conversation state
#     initial_state = {"messages": [SystemMessage(content=system_content)]}

#     try:
#         while True:
#             user_input = input("You: ")
#             if user_input.strip().lower() in {"exit", "quit"}:
#                 print("👋 Goodbye!")
#                 break

#             # validate_input(user_input)
#             # Add user message to state
#             initial_state["messages"].append(HumanMessage(content=user_input))

#             # Run the graph
#             result = app.invoke(initial_state)

#             # Update state with results
#             initial_state["messages"] = result["messages"]

#             # Display the final response
#             last_message = result["messages"][-1]
#             if hasattr(last_message, "content") and last_message.content:
#                 print(f"Bot: {last_message.content}\n")

#     except KeyboardInterrupt:
#         print("\n👋 Session terminated.")


# if __name__ == "__main__":
#     main()