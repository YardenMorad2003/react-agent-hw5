import os
import streamlit as st

from langchain_openai import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- API key handling ---
# On Streamlit Cloud, keys live in st.secrets (set in the app settings UI).
# For local runs, fall back to environment variables.
def _get_key(name: str) -> str:
    # st.secrets raises if no secrets.toml exists locally, so guard it.
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, "")

OPENAI_API_KEY = _get_key("OPENAI_API_KEY")
SERPAPI_API_KEY = _get_key("SERPAPI_API_KEY")

if not OPENAI_API_KEY or not SERPAPI_API_KEY:
    st.error(
        "Missing API keys. Set OPENAI_API_KEY and SERPAPI_API_KEY in "
        "Streamlit secrets (cloud) or as environment variables (local)."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

# --- Build the agent (cached so it only initializes once per session) ---

@st.cache_resource
def build_agent():
    llm = OpenAI(temperature=0)

    # Search tool
    search_tools = load_tools(["serpapi"])

    # Compare tool
    compare_prompt = PromptTemplate(
        input_variables=["items", "category"],
        template=(
            "Compare the following {category}: {items}.\n\n"
            "Provide a structured comparison covering:\n"
            "- Key features of each\n"
            "- Strengths and weaknesses\n"
            "- A brief overall recommendation\n\n"
            "Keep the comparison concise and factual."
        ),
    )
    compare_chain = LLMChain(llm=llm, prompt=compare_prompt)

    def compare_items(query: str) -> str:
        try:
            parts = [p.strip() for p in query.split(",") if p.strip()]
            if len(parts) < 3:
                return ("Error: Compare tool needs at least two items and a category. "
                        "Format: 'item1, item2, category'.")
            items = parts[:-1]
            category = parts[-1]
            return compare_chain.run(items=", ".join(items), category=category)
        except Exception as e:
            return f"Error running comparison: {str(e)}"

    compare_tool = Tool(
        name="Compare",
        func=compare_items,
        description=(
            "Use this tool to compare two or more items in the same category. "
            "Input MUST be a single comma-separated string where the last element "
            "is the category, e.g., 'iPhone 15 Pro, Galaxy S24 Ultra, smartphones'. "
            "Use this AFTER Search."
        ),
    )

    # Analyze tool
    analyze_prompt = PromptTemplate(
        input_variables=["results", "query"],
        template=(
            "You are analyzing information to answer a user's question.\n\n"
            "User's original question: {query}\n\n"
            "Information gathered so far:\n{results}\n\n"
            "Provide a concise analysis that:\n"
            "- Extracts the key facts relevant to the question\n"
            "- Summarizes main insights in 3-5 sentences\n"
            "- Highlights trade-offs or conclusions\n\n"
            "Be concise and factual."
        ),
    )
    analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt)

    def analyze_results(input_str: str) -> str:
        try:
            if "||" in input_str:
                query, results = input_str.split("||", 1)
                query, results = query.strip(), results.strip()
            else:
                query, results = "the user's question", input_str.strip()
            if not results:
                return "Error: no results provided to analyze."
            return analyze_chain.run(query=query, results=results)
        except Exception as e:
            return f"Error running analysis: {str(e)}"

    analyze_tool = Tool(
        name="Analyze",
        func=analyze_results,
        description=(
            "Use this tool to analyze and summarize search results or comparisons. "
            "Input MUST be formatted as: 'original_user_question || content_to_analyze'. "
            "Use this LAST, after Search and/or Compare."
        ),
    )

    tools = [search_tools[0], compare_tool, analyze_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True,
    )
    return agent

agent = build_agent()

# --- Streamlit UI ---

st.set_page_config(page_title="ReAct Agent", page_icon="🤖", layout="wide")
st.title("🤖 ReAct Agent with Multiple Tools")
st.caption("Search · Compare · Analyze — powered by LangChain + OpenAI + SerpAPI")
st.caption("DS-UA 301 HW5 Q1 · Yarden Morad")

with st.expander("ℹ️ How this works"):
    st.markdown(
        "This agent uses the ReAct (Yao et al., 2022) framework. Given a query, it "
        "reasons step by step about which tool to call. Three tools are available: "
        "**Search** (SerpAPI), **Compare** (LLM comparison of items in a category), "
        "and **Analyze** (LLM synthesis of intermediate results). The full reasoning "
        "trace is shown below the final answer."
    )

query = st.text_area(
    "Enter your query:",
    placeholder="e.g., What are the top 3 smartphones in 2023, and how do they compare on camera and battery?",
    height=100,
)

if st.button("Run ReAct Agent", type="primary"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Agent is reasoning..."):
            try:
                response = agent({"input": query})
                final_answer = response["output"]
                intermediate_steps = response.get("intermediate_steps", [])

                st.subheader("✅ Final Answer")
                st.success(final_answer)

                st.subheader("🧠 Reasoning Trace")
                if not intermediate_steps:
                    st.info("No intermediate steps recorded.")
                else:
                    for i, (action, observation) in enumerate(intermediate_steps, start=1):
                        with st.expander(f"Step {i}: {action.tool}", expanded=True):
                            st.markdown("**Thought + Action:**")
                            st.code(action.log.strip(), language="text")
                            st.markdown("**Observation:**")
                            st.code(str(observation).strip(), language="text")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("DS-UA 301 HW5 Q1 · ReAct Agent Implementation")
