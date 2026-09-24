```python
"""
AI Research & Article Writer
Backend module

Workflow:
User Topic
    |
    v
Researcher Agent
    |
    v
DuckDuckGo Search
    |
    v
Research Findings
    |
    v
Writer Agent
    |
    v
Final Markdown Article
"""

import os

from crewai import Agent, Crew, Process, Task
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS


# ============================================================
# DuckDuckGo Search
# ============================================================

def search_duckduckgo(query):
    """
    Search DuckDuckGo and return search results as text.
    """

    if not query:
        return "No search query was provided."

    try:
        results = []

        with DDGS() as search:

            search_results = search.text(
                query=query,
                max_results=8
            )

            for result in search_results:

                title = result.get("title", "")
                url = result.get("href", "")
                body = result.get("body", "")

                results.append(
                    "Title: "
                    + title
                    + "\nURL: "
                    + url
                    + "\nSummary: "
                    + body
                )

        if not results:
            return "No search results were found."

        return "\n\n---\n\n".join(results)

    except Exception as error:

        return (
            "DuckDuckGo search failed.\n"
            + str(error)
        )


# ============================================================
# Agent Crew Manager
# ============================================================

class AgentCrewManager:
    """
    Manages the CrewAI research and writing workflow.
    """

    def __init__(
        self,
        api_key=None,
        model="openai/gpt-oss-120b",
        temperature=0.2
    ):

        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        if not self.api_key:

            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add your API key to Streamlit Secrets "
                "or enter it in the sidebar."
            )

        self.model = model
        self.temperature = temperature

        self.llm = ChatGroq(
            model=self.model,
            groq_api_key=self.api_key,
            temperature=self.temperature
        )

    # ========================================================
    # Researcher
    # ========================================================

    def create_researcher(self):

        researcher = Agent(

            role="Senior Technical Researcher",

            goal=(
                "Research the requested topic and provide "
                "accurate, useful and well-organized information."
            ),

            backstory=(
                "You are an experienced technical researcher. "
                "You investigate topics carefully and prioritize "
                "reliable sources, official documentation, "
                "academic research and reputable publications."
            ),

            llm=self.llm,

            verbose=False,

            allow_delegation=False
        )

        return researcher

    # ========================================================
    # Writer
    # ========================================================

    def create_writer(self):

        writer = Agent(

            role="Technical Article Writer",

            goal=(
                "Transform research findings into a clear, "
                "professional and well-structured Markdown article."
            ),

            backstory=(
                "You are an experienced technical writer. "
                "You turn complex research into easy-to-understand "
                "professional articles."
            ),

            llm=self.llm,

            verbose=False,

            allow_delegation=False
        )

        return writer

    # ========================================================
    # Research Task
    # ========================================================

    def create_research_task(
        self,
        topic,
        depth
    ):

        if depth == "Basic":

            depth_instruction = (
                "Focus on the most important facts "
                "and keep the research concise."
            )

        elif depth == "Deep":

            depth_instruction = (
                "Perform extensive research. Look for "
                "primary sources, recent developments, "
                "real-world examples and conflicting information."
            )

        else:

            depth_instruction = (
                "Conduct thorough research covering "
                "definitions, concepts, facts, examples, "
                "applications and reliable sources."
            )

        description = f"""
Research the following topic:

{topic}

Research depth:

{depth}

{depth_instruction}

Your research should cover:

1. Topic definition
2. Important concepts
3. Important facts
4. Recent developments when relevant
5. Real-world applications
6. Practical examples
7. Reliable sources
8. Primary sources where possible
9. Source names
10. Source URLs

Important rules:

- Do not invent facts.
- Do not invent statistics.
- Do not invent sources.
- Do not invent URLs.
- Clearly identify uncertain information.

Use web search when additional information is required.

Return a structured research report.
Do not write the final article.
"""

        task = Task(

            description=description,

            expected_output=(
                "A structured research report containing "
                "the topic definition, key concepts, facts, "
                "applications, examples and reliable sources."
            )
        )

        return task

    # ========================================================
    # Writing Task
    # ========================================================

    def create_writing_task(
        self,
        topic,
        research_task,
        article_length
    ):

        if article_length == "Short":

            length = (
                "Approximately 700 to 1000 words."
            )

        elif article_length == "Long":

            length = (
                "Approximately 2000 to 3000 words."
            )

        else:

            length = (
                "Approximately 1200 to 1800 words."
            )

        description = f"""
Write a professional Markdown article about:

{topic}

Article length:

{length}

Use the research report from the Researcher Agent
as the factual foundation.

The article MUST contain:

# Title

## Introduction

Explain the topic and why it matters.

## Core Concepts

Explain the major concepts clearly.

## Real-World Use Cases

Explain practical applications and examples.

## Key Takeaways

Provide important points as bullet points.

## Summary Table

Create a useful Markdown table.

## Conclusion

Summarize the major findings.

## References

List the sources supplied by the research.

Rules:

1. Return clean Markdown.
2. Use headings and subheadings.
3. Use bullet points where useful.
4. Include a Markdown table.
5. Keep the writing professional.
6. Explain technical terminology.
7. Avoid unnecessary repetition.
8. Do not mention CrewAI.
9. Do not mention internal prompts.
10. Do not invent citations.
11. Do not invent URLs.
12. Do not invent statistics.
13. Do not invent quotes.
14. Base factual claims on the research.

The final result must be ready to save as a Markdown file.
"""

        task = Task(

            description=description,

            expected_output=(
                "A complete Markdown article containing "
                "a title, introduction, core concepts, "
                "real-world use cases, key takeaways, "
                "summary table, conclusion and references."
            ),

            context=[
                research_task
            ]
        )

        return task

    # ========================================================
    # Generate Article
    # ========================================================

    def generate_article(
        self,
        topic,
        depth="Detailed",
        article_length="Medium"
    ):

        if not topic or not topic.strip():

            raise ValueError(
                "Please provide a research topic."
            )

        researcher = self.create_researcher()

        writer = self.create_writer()

        research_task = self.create_research_task(
            topic=topic.strip(),
            depth=depth
        )

        writing_task = self.create_writing_task(
            topic=topic.strip(),
            research_task=research_task,
            article_length=article_length
        )

        research_task.agent = researcher

        writing_task.agent = writer

        crew = Crew(

            agents=[
                researcher,
                writer
            ],

            tasks=[
                research_task,
                writing_task
            ],

            process=Process.sequential,

            verbose=False
        )

        result = crew.kickoff()

        return str(result)
```
