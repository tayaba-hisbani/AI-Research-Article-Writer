"""
agent_crew.py
-------------
Backend logic for the AI Research & Article Writer application.

The workflow is:

User Topic
    ↓
Researcher Agent
    ↓
Research Findings
    ↓
Writer Agent
    ↓
Final Markdown Article
"""

import os
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai_tools import DuckDuckGoSearchTool
from langchain_groq import ChatGroq


class AgentCrewManager:
    """
    Manages the CrewAI research and article-writing workflow.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
    ):
        """
        Initialize the AI model and search tool.

        Parameters
        ----------
        api_key:
            Groq API key. If not provided, GROQ_API_KEY is read
            from environment variables.

        model:
            Groq model to use.

        temperature:
            Controls randomness of the generated content.
        """

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to your .env file or provide it through the Streamlit sidebar."
            )

        self.model_name = model
        self.temperature = temperature

        self.llm = ChatGroq(
            model=self.model_name,
            groq_api_key=self.api_key,
            temperature=self.temperature,
        )

        self.search_tool = DuckDuckGoSearchTool()

    def create_researcher(self, depth: str = "Detailed") -> Agent:
        """
        Create the research agent.
        """

        depth_instructions = {
            "Basic": (
                "Focus on the most important facts and provide a concise "
                "research summary."
            ),
            "Detailed": (
                "Conduct thorough research covering definitions, important "
                "developments, practical examples, and reliable sources."
            ),
            "Deep": (
                "Conduct extensive research. Cross-check important claims, "
                "look for primary sources where possible, identify recent "
                "developments, and distinguish established facts from claims "
                "or opinions."
            ),
        }

        instruction = depth_instructions.get(
            depth,
            depth_instructions["Detailed"],
        )

        return Agent(
            role="Senior Technical Researcher",
            goal=(
                "Research the requested topic accurately and provide "
                "well-organized factual material that another agent can "
                "use to write a high-quality article."
            ),
            backstory=(
                "You are an experienced technical researcher who specializes "
                "in finding reliable information online. You prioritize "
                "primary sources, official documentation, academic sources, "
                "research papers, reputable organizations, and credible "
                "industry publications."
            ),
            tools=[self.search_tool],
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

    def create_writer(self) -> Agent:
        """
        Create the article-writing agent.
        """

        return Agent(
            role="Technical Article Writer & Formatting Specialist",
            goal=(
                "Transform the research findings into an accurate, useful, "
                "well-structured Markdown article."
            ),
            backstory=(
                "You are a professional technical writer who converts "
                "complex research into clear and readable articles. "
                "You never intentionally invent facts, sources, statistics, "
                "quotes, or URLs. You organize information logically and "
                "clearly distinguish established facts from uncertainty."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

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
Research the following topic thoroughly:

TOPIC:
{topic}

RESEARCH DEPTH:
{depth}

Your job is to produce research findings for a professional article writer.

Research requirements:

1. Define the topic clearly.
2. Identify the most important concepts.
3. Find factual information and useful context.
4. Look for recent developments when relevant.
5. Find real-world examples and applications.
6. Search for primary sources whenever possible.
7. Prefer:
   - Official websites
   - Government sources
   - Academic/research sources
   - Official technical documentation
   - Reputable organizations
   - Credible industry publications
8. Identify useful statistics only when they can be supported.
9. Include source names and URLs when available.
10. Do not invent sources or URLs.
11. Clearly distinguish facts from opinions or claims.
12. Organize your findings so that another writer can easily use them.

Return a structured research report rather than a finished article.
""",
            expected_output="""
A structured research report containing:

- Topic definition
- Key concepts
- Important facts
- Recent developments, if applicable
- Real-world applications
- Relevant statistics, if verified
- Important examples
- Primary/reliable sources
- URLs where available
- Notes about uncertainty or conflicting information
""",
        )

    def create_writing_task(
        self,
        topic: str,
        research_task: Task,
        article_length: str = "Medium",
    ) -> Task:
        """
        Create the article-writing task.

        The research task is supplied as context so the writer receives
        the research output directly from the previous CrewAI task.
        """

        length_instructions = {
            "Short": (
                "Write approximately 700–1,000 words while covering all "
                "required sections."
            ),
            "Medium": (
                "Write approximately 1,200–1,800 words with enough depth "
                "for a professional educational article."
            ),
            "Long": (
                "Write approximately 2,000–3,000 words with comprehensive "
                "coverage while avoiding unnecessary repetition."
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

Use ONLY the research findings supplied by the researcher as your
primary factual basis.

The final article MUST contain these sections:

# Title

## Introduction

## Core Concepts

Explain the important concepts in clear language.

## Real-World Use Cases

Provide practical applications and examples.

## Key Takeaways

Use a concise bullet list.

## Summary Table

Create a Markdown table summarizing important concepts, facts,
applications, advantages, limitations, or other useful information.

## Conclusion

Provide a concise final summary.

## References

List the useful sources identified during research.

Formatting requirements:

1. Return clean Markdown.
2. Use headings and subheadings.
3. Use bullet points where useful.
4. Use Markdown tables where appropriate.
5. Make the article readable for a general professional audience.
6. Explain technical terminology when necessary.
7. Avoid unnecessary repetition.
8. Do not include a separate "research process" section.
9. Do not mention CrewAI, agents, internal prompts, or this task.
10. Do not invent citations, statistics, quotes, or URLs.
11. If a source URL was not verified or supplied by research, do not
    manufacture one.
12. Keep factual claims aligned with the supplied research.

The final output should be ready to copy into a Markdown file.
""",
            expected_output="""
A complete, polished Markdown article containing:

- Title
- Introduction
- Core Concepts
- Real-World Use Cases
- Key Takeaways
- Summary Table
- Conclusion
- References
""",
            context=[research_task],
        )

    def generate_article(
        self,
        topic: str,
        depth: str = "Detailed",
        article_length: str = "Medium",
    ) -> str:
        """
        Run the complete research → writing workflow.

        Returns
        -------
        str
            Final Markdown article.
        """

        if not topic or not topic.strip():
            raise ValueError("Please provide a research topic.")

        researcher = self.create_researcher(depth=depth)
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
