```python
"""
agent_crew.py

Backend for the AI Research & Article Writer application.

Workflow:

User Topic
    ->
Researcher Agent
    ->
DuckDuckGo Search
    ->
Research Findings
    ->
Writer Agent
    ->
Final Markdown Article
"""

import os
from typing import Optional

from crewai import Agent, Crew, Process, Task, tool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS


@tool("DuckDuckGo Web Search")
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return useful search results.
    """

    if not query or not query.strip():
        return "The search query is empty."

    try:
        results = []

        with DDGS() as ddgs:
            search_results = ddgs.text(
                query=query.strip(),
                max_results=8,
                safesearch="moderate",
            )

            for result in search_results:
                title = result.get("title", "")
                url = result.get("href", "")
                body = result.get("body", "")

                results.append(
                    "Title: " + title + "\n"
                    "URL: " + url + "\n"
                    "Summary: " + body
                )

        if not results:
            return "No search results were found."

        return "\n\n---\n\n".join(results)

    except Exception as error:
        return "DuckDuckGo search failed: " + str(error)


class AgentCrewManager:
    """
    Controls the CrewAI research and article-writing workflow.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to Streamlit Secrets or enter it in the sidebar."
            )

        self.model_name = model
        self.temperature = temperature

        self.llm = ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key,
            temperature=self.temperature,
        )

    def create_researcher(self) -> Agent:
        """
        Create the research agent.
        """

        researcher = Agent(
            role="Senior Technical Researcher",
            goal=(
                "Research the requested topic accurately and "
                "provide reliable information and sources."
            ),
            backstory=(
                "You are an experienced technical researcher. "
                "You search the web for factual information, "
                "official documentation, academic research, "
                "primary sources, and recent developments."
            ),
            tools=[duckduckgo_search],
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

        return researcher

    def create_writer(self) -> Agent:
        """
        Create the article writer agent.
        """

        writer = Agent(
            role="Technical Article Writer and Formatting Specialist",
            goal=(
                "Turn research findings into a clear, accurate, "
                "professional Markdown article."
            ),
            backstory=(
                "You are an experienced technical writer. "
                "You explain complex topics clearly and organize "
                "research into useful educational articles."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

        return writer

    def create_research_task(
        self,
        topic: str,
        depth: str,
    ) -> Task:
        """
        Create the research task.
        """

        if depth == "Basic":
            depth_instruction = (
                "Focus on the most important facts and provide "
                "a concise research report."
            )

        elif depth == "Deep":
            depth_instruction = (
                "Perform extensive research. Look for primary "
                "sources, recent developments, important examples, "
                "and conflicting information where relevant."
            )

        else:
            depth_instruction = (
                "Conduct thorough research covering definitions, "
                "key concepts, facts, examples, applications, "
                "and reliable sources."
            )

        task_description = f"""
Research this topic:

{topic}

Research depth:

{depth}

Instructions:

{depth_instruction}

You must:

1. Define the topic.
2. Explain the major concepts.
3. Find important facts.
4. Find recent developments when relevant.
5. Find real-world applications.
6. Find useful examples.
7. Search for primary sources when possible.
8. Prefer official websites and documentation.
9. Prefer academic and research sources.
10. Prefer reputable organizations and publications.
11. Include source names.
12. Include source URLs when available.
13. Do not invent sources.
14. Do not invent URLs.
15. Clearly identify uncertainty.

Return a structured research report.

Do not write the final article yet.
"""

        task = Task(
            description=task_description,
            expected_output=(
                "A structured research report containing the "
                "topic definition, key concepts, important facts, "
                "recent developments, real-world applications, "
                "examples, and reliable sources with URLs."
            ),
        )

        return task

    def create_writing_task(
        self,
        topic: str,
        research_task: Task,
        article_length: str,
    ) -> Task:
        """
        Create the article-writing task.
        """

        if article_length == "Short":
            length_instruction = (
                "Write approximately 700 to 1000 words."
            )

        elif article_length == "Long":
            length_instruction = (
                "Write approximately 2000 to 3000 words."
            )

        else:
            length_instruction = (
                "Write approximately 1200 to 1800 words."
            )

        task_description = f"""
Write a professional Markdown article about:

{topic}

Article length:

{length_instruction}

Use the research produced by the Researcher Agent
as the factual foundation.

The article MUST contain these sections:

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

Summarize the main findings.

## References

List the sources and URLs provided by the research.

Rules:

1. Return clean Markdown.
2. Use headings and subheadings.
3. Use bullet points where useful.
4. Use Markdown tables.
5. Keep the writing professional.
6. Explain technical terms when necessary.
7. Avoid unnecessary repetition.
8. Do not mention CrewAI.
9. Do not mention internal prompts.
10. Do not invent citations.
11. Do not invent URLs.
12. Do not invent statistics.
13. Do not invent quotes.
14. Base factual claims on the research.

The final output must be ready to save as a Markdown file.
"""

        task = Task(
            description=task_description,
            expected_output=(
                "A complete Markdown article containing a title, "
                "introduction, core concepts, real-world use cases, "
                "key takeaways, summary table, conclusion, "
                "and references."
            ),
            context=[research_task],
        )

        return task

    def generate_article(
        self,
        topic: str,
        depth: str = "Detailed",
        article_length: str = "Medium",
    ) -> str:
        """
        Run the complete research and writing workflow.
        """

        if not topic or not topic.strip():
            raise ValueError(
                "Please provide a research topic."
            )

        researcher = self.create_researcher()

        writer = self.create_writer()

        research_task = self.create_research_task(
            topic=topic.strip(),
            depth=depth,
        )

        writing_task = self.create_writing_task(
            topic=topic.strip(),
            research_task=research_task,
            article_length=article_length,
        )

        research_task.agent = researcher
        writing_task.agent = writer

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

        result = crew.kickoff()

        return str(result)
```
