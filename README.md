# 🔎 AI Research & Article Writer

An AI-powered research and article-writing web application built with:

- Streamlit
- CrewAI
- Groq
- `openai/gpt-oss-120b`
- DuckDuckGo Search
- LangChain Groq
- Python

The application uses a sequential multi-agent workflow:

```text
User Topic
    ↓
Senior Technical Researcher
    ↓
DuckDuckGo Web Search
    ↓
Research Findings
    ↓
Technical Article Writer
    ↓
Markdown Article
    ↓
Download
