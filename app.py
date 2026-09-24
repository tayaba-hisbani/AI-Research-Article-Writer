import os
import streamlit as st
from dotenv import load_dotenv

from agent_crew import AgentCrewManager


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research & Article Writer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.15rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        .feature-card {
            padding: 1.2rem;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background-color: #ffffff;
            min-height: 150px;
        }

        .feature-title {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .feature-text {
            color: #6b7280;
            line-height: 1.6;
        }

        .success-box {
            padding: 0.8rem;
            border-radius: 8px;
            background-color: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GET GROQ API KEY
# ============================================================

def get_groq_api_key():
    """
    Priority:
    1. Streamlit Cloud Secrets
    2. Local .env file
    3. Manual sidebar input
    """

    # --------------------------------------------------------
    # 1. Streamlit Secrets
    # --------------------------------------------------------

    try:
        if "GROQ_API_KEY" in st.secrets:
            secret_key = st.secrets["GROQ_API_KEY"]

            if secret_key:
                return str(secret_key).strip()
    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Local .env
    # --------------------------------------------------------

    env_key = os.getenv("GROQ_API_KEY")

    if env_key:
        return env_key.strip()

    # --------------------------------------------------------
    # 3. No key found
    # --------------------------------------------------------

    return None


api_key = get_groq_api_key()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    if api_key:

        st.success("✓ Groq API key loaded")

        st.caption(
            "Your API key is being loaded from "
            "Streamlit Secrets or the local environment."
        )

    else:

        st.warning("Groq API key not found.")

        manual_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq API key if you are not using Streamlit Secrets.",
        )

        if manual_key:
            api_key = manual_key.strip()

    st.divider()

    # --------------------------------------------------------
    # ADVANCED OPTIONS
    # --------------------------------------------------------

    st.subheader("Advanced Options")

    research_depth = st.selectbox(
        "Research Depth",
        options=[
            "Quick",
            "Standard",
            "Detailed",
            "Deep Research",
        ],
        index=2,
        help="Controls how much research the AI should perform.",
    )

    article_length = st.selectbox(
        "Article Length",
        options=[
            "Short",
            "Medium",
            "Long",
            "Very Long",
        ],
        index=1,
        help="Controls the approximate length of the generated article.",
    )

    st.divider()

    st.caption(
        "AI Research & Article Writer\n"
        "Powered by Streamlit, CrewAI, Groq and web search."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔎 AI Research & Article Writer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Research a topic using web search and generate a structured,
    professional Markdown article with AI.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">🔎 Research</div>
            <div class="feature-text">
                Searches the web for relevant facts, information
                and sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">🤖 AI Writing</div>
            <div class="feature-text">
                Converts research into a structured,
                professional article.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">📥 Export</div>
            <div class="feature-text">
                Download the finished article as a Markdown file.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")
st.write("")


# ============================================================
# RESEARCH TOPIC
# ============================================================

st.header("Research Topic")

st.write("What would you like to research?")

topic = st.text_area(
    "Enter your research topic",
    placeholder=(
        "Example: The impact of artificial intelligence "
        "on business education"
    ),
    height=120,
    label_visibility="collapsed",
)


# ============================================================
# TOPIC SUGGESTIONS
# ============================================================

st.caption("💡 Topic suggestions")

suggestion_col1, suggestion_col2, suggestion_col3 = st.columns(3)

with suggestion_col1:

    if st.button(
        "AI in Business",
        use_container_width=True,
    ):
        st.session_state["topic"] = (
            "The impact of artificial intelligence on modern business"
        )

with suggestion_col2:

    if st.button(
        "Future of Remote Work",
        use_container_width=True,
    ):
        st.session_state["topic"] = (
            "The future of remote work and its impact on businesses"
        )

with suggestion_col3:

    if st.button(
        "AI in Education",
        use_container_width=True,
    ):
        st.session_state["topic"] = (
            "The impact of artificial intelligence on education"
        )


# Use suggested topic if selected
if "topic" in st.session_state and st.session_state["topic"]:

    topic = st.session_state["topic"]

    st.info(
        f"Selected topic: **{topic}**"
    )


st.write("")


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🚀 Generate Article",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ARTICLE GENERATION
# ============================================================

if generate_button:

    # --------------------------------------------------------
    # Validate API key
    # --------------------------------------------------------

    if not api_key:

        st.error(
            "Groq API key is missing. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

        st.stop()

    # --------------------------------------------------------
    # Validate topic
    # --------------------------------------------------------

    if not topic or not topic.strip():

        st.warning(
            "Please enter a research topic first."
        )

        st.stop()

    topic = topic.strip()

    # --------------------------------------------------------
    # Generate article
    # --------------------------------------------------------

    try:

        with st.status(
            "🔎 Researching your topic...",
            expanded=True,
        ) as status:

            st.write("Searching for relevant information...")
            st.write("Analyzing the research...")
            st.write("Preparing the article...")

            # ------------------------------------------------
            # Create Crew Manager
            # ------------------------------------------------

            manager = AgentCrewManager(
                api_key=api_key
            )

            # ------------------------------------------------
            # Generate article
            # ------------------------------------------------

            if hasattr(manager, "generate_article"):

                result = manager.generate_article(
                    topic=topic,
                    depth=research_depth,
                    length=article_length,
                )

            elif hasattr(manager, "run"):

                result = manager.run(
                    topic=topic,
                    depth=research_depth,
                    length=article_length,
                )

            else:

                raise AttributeError(
                    "AgentCrewManager does not contain "
                    "generate_article() or run()."
                )

            status.update(
                label="✅ Article generated successfully!",
                state="complete",
                expanded=False,
            )

        # ----------------------------------------------------
        # Convert result to text
        # ----------------------------------------------------

        if hasattr(result, "raw"):

            article = result.raw

        else:

            article = str(result)

        if not article.strip():

            st.error(
                "The AI returned an empty article."
            )

            st.stop()

        # ----------------------------------------------------
        # Save article in session state
        # ----------------------------------------------------

        st.session_state["generated_article"] = article
        st.session_state["generated_topic"] = topic

    except Exception as e:

        st.error(
            "Something went wrong while generating the article."
        )

        st.exception(e)


# ============================================================
# DISPLAY GENERATED ARTICLE
# ============================================================

if "generated_article" in st.session_state:

    st.divider()

    st.header("📄 Generated Article")

    article = st.session_state["generated_article"]

    # --------------------------------------------------------
    # Article preview
    # --------------------------------------------------------

    st.markdown(article)

    st.divider()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    download_filename = "AI_Research_Article.md"

    st.download_button(
        label="📥 Download Article as Markdown",
        data=article,
        file_name=download_filename,
        mime="text/markdown",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.write("")

st.caption(
    "AI Research & Article Writer • Built with Streamlit, "
    "CrewAI, Groq and DuckDuckGo"
)
