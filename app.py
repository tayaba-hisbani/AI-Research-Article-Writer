"""
app.py
------
AI Research & Article Writer

Streamlit frontend for the CrewAI research and article-writing system.
"""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from agent_crew import AgentCrewManager


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Research & Article Writer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.7rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sub-header {
            font-size: 1.15rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        .info-card {
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 1rem;
        }

        .topic-example {
            padding: 0.7rem 1rem;
            border-radius: 8px;
            background-color: rgba(128, 128, 128, 0.08);
            margin-bottom: 0.5rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "article" not in st.session_state:
    st.session_state.article = ""

if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    st.markdown(
        """
        Enter your Groq API key below if you are not using
        a local `.env` file.
        """
    )

    sidebar_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Your API key is used only for this application session.",
    )

    st.divider()

    st.subheader("Advanced Options")

    research_depth = st.selectbox(
        "Research Depth",
        options=[
            "Basic",
            "Detailed",
            "Deep",
        ],
        index=1,
        help="Controls how extensively the researcher should investigate the topic.",
    )

    article_length = st.selectbox(
        "Article Length",
        options=[
            "Short",
            "Medium",
            "Long",
        ],
        index=1,
        help="Controls the approximate size of the final article.",
    )

    st.divider()

    st.caption(
        "AI Research & Article Writer\n"
        "Streamlit + CrewAI + Groq + DuckDuckGo"
    )


# ---------------------------------------------------------
# API key resolution
# ---------------------------------------------------------

env_api_key = os.getenv("GROQ_API_KEY", "").strip()

api_key = sidebar_api_key.strip() if sidebar_api_key.strip() else env_api_key


# ---------------------------------------------------------
# Main header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-header">🔎 AI Research & Article Writer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-header">
    Research a topic using web search and generate a structured,
    professional Markdown article with AI.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Information cards
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
        <strong>🔎 Research</strong><br>
        Searches the web for relevant facts and sources.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
        <strong>🤖 AI Writing</strong><br>
        Converts research into a structured article.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
        <strong>📥 Export</strong><br>
        Download the finished article as Markdown.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Topic input
# ---------------------------------------------------------

st.subheader("Research Topic")

topic = st.text_input(
    "What would you like to research?",
    placeholder="Example: Generative AI in Business",
    help="Enter a clear topic that you want the AI to research and write about.",
)


# ---------------------------------------------------------
# Suggestions
# ---------------------------------------------------------

st.markdown("**Sample topics:**")

suggestions = [
    "Generative AI in Business",
    "AI Agents in Education",
    "Future of Data Analytics",
    "Cybersecurity for Small Businesses",
    "AI in Healthcare",
]

suggestion_columns = st.columns(len(suggestions))

for column, suggestion in zip(suggestion_columns, suggestions):
    with column:
        if st.button(
            suggestion,
            key=f"suggestion_{suggestion}",
        ):
            st.session_state.selected_topic = suggestion
            st.rerun()


if "selected_topic" in st.session_state:
    if st.session_state.selected_topic:
        topic = st.session_state.selected_topic

        st.info(
            f"Selected topic: **{topic}**"
        )


# ---------------------------------------------------------
# Generate button
# ---------------------------------------------------------

st.write("")

generate_button = st.button(
    "🚀 Generate Article",
    type="primary",
)


# ---------------------------------------------------------
# Article generation
# ---------------------------------------------------------

if generate_button:

    # Validate topic
    if not topic or not topic.strip():
        st.warning(
            "Please enter a research topic before generating an article."
        )
        st.stop()

    # Validate API key
    if not api_key:
        st.error(
            "Groq API key is missing. Add it in the sidebar or configure "
            "GROQ_API_KEY in your environment."
        )
        st.stop()

    clean_topic = topic.strip()

    st.session_state.last_topic = clean_topic

    try:

        # Status container
        status = st.status(
            "Starting AI research workflow...",
            expanded=True,
        )

        with status:

            st.write("🔐 Validating Groq API configuration...")

            manager = AgentCrewManager(
                api_key=api_key,
                model="openai/gpt-oss-120b",
                temperature=0.2,
            )

            st.write("🔎 Researcher agent is searching for relevant information...")

            st.write(
                f"📚 Research depth: **{research_depth}**"
            )

            st.write(
                f"📝 Article length: **{article_length}**"
            )

            st.write("✍️ Writer agent is preparing the article...")

            article = manager.generate_article(
                topic=clean_topic,
                depth=research_depth,
                article_length=article_length,
            )

            st.session_state.article = article

            st.write("✅ Article generation completed.")

        status.update(
            label="Article generated successfully!",
            state="complete",
            expanded=False,
        )

    except Exception as error:

        st.error(
            "The article could not be generated."
        )

        with st.expander("Technical error details"):
            st.code(str(error))

        st.stop()


# ---------------------------------------------------------
# Display final article
# ---------------------------------------------------------

if st.session_state.article:

    st.divider()

    st.subheader("📄 Generated Article")

    st.markdown(
        st.session_state.article
    )

    st.divider()

    # Download filename
    safe_topic = (
        st.session_state.last_topic
        .lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{safe_topic[:50]}_article_{timestamp}.md"
    )

    st.download_button(
        label="📥 Download Article as Markdown",
        data=st.session_state.article,
        file_name=filename,
        mime="text/markdown",
    )

    st.caption(
        "Tip: You can open the downloaded `.md` file in "
        "VS Code, GitHub, Obsidian, Typora, or another Markdown editor."
    )
