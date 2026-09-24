"""
agent_crew.py
-------------
Backend for the AI Research & Article Writer.

Architecture:

User Topic
    ↓
Researcher Agent
    ↓
Custom DuckDuckGo CrewAI Tool
    ↓
Research Findings
    ↓
Writer Agent
    ↓
Final Markdown Article
"""

import os
from typing import Optional

from crewai import Agent, Crew, Process, Task, tool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS


# ============================================================
# DuckDuckGo Search Tool
# ============================================================

@tool("DuckDuckGo Web Search")
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Use this tool when you need current information, facts,
    sources, official documentation, research papers, or
    recent developments about a topic.
    """

    if not query or not query.strip():
        return "Search query cannot be empty."

    try:
        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                query=query.strip(),
                max_results=8,
                safesearch="moderate",
            )

            for item in search_results:
                title = item.get("title", "No title")
                url = item.get("href", "")
                body = item.get("body", "")

                results.append(
                    f"TITLE: {title}\n"
                    f"URL: {url}\n"
                    f"SUMMARY: {body}\n"
                )

        if not results:
            return (
                "No search results were found. "
                "Try a different or more specific search query."
            )

        return "\n---\n".join(results)

    except Exception as exc:
        return (
            "DuckDuckGo search failed.\n"
            f"Error: {str(exc)}"
        )


# ============================================================
# Agent Crew Manager
# ============================================================

class AgentCrewManager:
    """
    Manages the complete research and article-writing workflow.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
    ):
        """
        Initialize the CrewAI workflow.
        """

        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to Streamlit Secrets, your .env file, "
                "or enter it in the sidebar."
            )

        self.model_name = model
        self.temperature = temperature

        # ----------------------------------------------------
        # Groq LLM
        # ----------------------------------------------------

        self.llm = ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key,
            temperature=self.temperature,
        )

    # ========================================================
    # Researcher Agent
    # ========================================================

    def create_researcher(
        self,
        depth: str = "Detailed",
    ) -> Agent:
        """
        Create the Senior Technical Researcher agent.
        """

        depth_instructions = {
            "Basic": (
                "Focus on the most important facts and "
                "provide a concise research summary."
            ),
            "Detailed": (
                "Conduct thorough research covering definitions, "
                "important developments, practical examples, "
                "and reliable sources."
            ),
            "Deep": (
                "Conduct extensive research. Cross-check important "
                "claims, look for primary sources where possible, "
                "identify recent developments, and distinguish "
                "established facts from claims or opinions."
            ),
        }

        research_instruction = depth_instructions.get(
            depth,
            depth_instructions["Detailed"],
        )

        return Agent(
            role="Senior Technical Researcher",

            goal=(
                "Research the requested topic accurately and "
                "provide well-organized factual material for "
                "a professional article writer."
            ),

            backstory=(
                "You are an experienced technical researcher "
                "specializing in online research. You prioritize "
                "official documentation, primary sources, "
                "academic research, government sources, "
                "reputable organizations, and credible "
                "industry publications."
            ),

            tools=[
                duckduckgo_search
            ],

            llm=self.llm,

            verbose=False,

            allow_delegation=False,
        )

    # ========================================================
    # Writer Agent
    # ========================================================

    def create_writer(self) -> Agent:
        """
        Create the Technical Article Writer agent.
        """

        return Agent(
            role="Technical Article Writer & Formatting Specialist",

            goal=(
                "Transform research findings into an accurate, "
                "useful, professional and readable Markdown article."
            ),

            backstory=(
                "You are an experienced technical writer who "
                "turns complex research into clear educational "
                "content. You never intentionally invent facts, "
                "statistics, citations, quotes or URLs."
            ),

            llm=self.llm,

            verbose=False,

            allow_delegation=False,
        )

    # ========================================================
    # Research Task
    # ========================================================

    def create_research_task(
        self,
        topic: str,
        depth: str = "Detailed",
    ) -> Task:
        """
        Create the research task.
        """

        return Task(
            description=f"""
Research the following topic:

TOPIC:
{topic}

RESEARCH DEPTH:
{depth}

{{
RESEARCH INSTRUCTIONS
}}

{self._research_depth_instruction(depth)}

You must:

1. Clearly define the topic.

2. Identify the most important concepts.

3. Find factual information.

4. Find recent developments when relevant.

5. Find real-world applications.

6. Search for primary sources whenever possible.

7. Prefer:
   - Official websites
   - Government sources
   - Academic sources
   - Research papers
   - Official documentation
   - Reputable organizations
   - Credible industry publications

8. Identify useful statistics only when supported.

9. Record source names.

10. Record source URLs whenever available.

11. Do not invent sources.

12. Do not invent URLs.

13. Clearly identify uncertainty or conflicting information.

14. Organize the findings so the writer can directly use them.

The output should be a structured research report,
NOT a finished article.
""",

            expected_output="""
A structured research report containing:

- Topic definition
- Key concepts
- Important facts
- Recent developments
- Real-world applications
- Examples
- Statistics where verified
- Primary/reliable sources
- Source URLs
- Notes about uncertainty
""",
        )

    # ========================================================
    # Research Depth Helper
    # ========================================================

    @staticmethod
    def _research_depth_instruction(
        depth: str,
    ) -> str:

        instructions = {
            "Basic": """
Focus on the most important information.
Avoid unnecessary details.
""",

            "Detailed": """
Provide a thorough research report covering
the major concepts, facts, applications,
examples and sources.
""",

            "Deep": """
Perform extensive research.
Cross-check important claims,
look for primary sources,
look for recent developments,
and identify disagreements or uncertainty.
""",
        }

        return instructions.get(
            depth,
            instructions["Detailed"],
        )

    # ========================================================
    # Writing Task
    # ========================================================

    def create_writing_task(
        self,
        topic: str,
        research_task: Task,
        article_length: str = "Medium",
    ) -> Task:
        """
        Create the article-writing task.
        """

        length_instructions = {
            "Short": (
                "Write approximately 700–1,000 words."
            ),

            "Medium": (
                "Write approximately 1,200–1,800 words."
            ),

            "Long": (
                "Write approximately 2,000–3,000 words."
            ),
        }

        length_instruction = length_instructions.get(
            article_length,
            length_instructions["Medium"],
        )

        return Task(
            description=f"""
Write a professional Markdown article about:

TOPIC:
{topic}

ARTICLE LENGTH:
{length_instruction}

Use the research findings from the previous
Researcher Agent as the factual foundation.

The final article MUST contain:

# Title

## Introduction

Explain what the topic is and why it matters.

## Core Concepts

Explain the major concepts clearly.

## Real-World Use Cases

Explain practical applications and examples.

## Key Takeaways

Provide concise bullet points.

## Summary Table

Create a useful Markdown table.

## Conclusion

Summarize the major findings.

## References

List the relevant sources and URLs supplied
by the research.

FORMATTING RULES:

1. Return clean Markdown.

2. Use headings and subheadings.

3. Use bullet points where useful.

4. Use Markdown tables.

5. Keep the writing professional and readable.

6. Explain technical terminology when necessary.

7. Avoid unnecessary repetition.

8. Do not mention CrewAI.

9. Do not mention internal prompts.

10. Do not invent citations.

11. Do not invent URLs.

12. Do not invent statistics.

13. Do not invent quotes.

14. Keep factual claims aligned with the research.

The output must be ready to save as a Markdown file.
""",

            expected_output="""
A complete Markdown article containing:

- Title
- Introduction
- Core Concepts
- Real-World Use Cases
- Key Takeaways
- Summary Table
- Conclusion
- References
""",

            context=[
                research_task
            ],
        )

    # ========================================================
    # Main Execution
    # ========================================================

    def generate_article(
        self,
        topic: str,
        depth: str = "Detailed",
        article_length: str = "Medium",
    ) -> str:
        """
        Run:

        Researcher
            ↓
        Writer
            ↓
        Final Article
        """

        if not topic or not topic.strip():
            raise ValueError(
                "Please provide a research topic."
            )

        # ----------------------------------------------------
        # Create agents
        # ----------------------------------------------------

        researcher = self.create_researcher(
            depth=depth
        )

        writer = self.create_writer()

        # ----------------------------------------------------
        # Create tasks
        # ----------------------------------------------------

        research_task = self.create_research_task(
            topic=topic.strip(),
            depth=depth,
        )

        writing_task = self.create_writing_task(
            topic=topic.strip(),
            research_task=research_task,
            article_length=article_length,
        )

        # Assign agents
        research_task.agent = researcher
        writing_task.agent = writer

        # ----------------------------------------------------
        # Create Crew
        # ----------------------------------------------------

        crew = Crew(
            agents=[
                researcher,
                writer,
            ],

            tasks=[
                research_task,
                writing_task,
            ],

            process=Process.sequential,

            verbose=False,
        )

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        result = crew.kickoff()

        return str(result)
```
