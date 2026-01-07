"""
Prompt Templates for Omni-Cortex

Reusable templates for framework selection, reasoning, and code generation.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# Framework selection prompt
FRAMEWORK_SELECTION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an AI routing system that selects the optimal reasoning framework.

Available frameworks: {frameworks}

Task characteristics:
- Type: {task_type}
- Complexity: {complexity}
- Has code: {has_code}

Previous frameworks used: {framework_history}"""),
    ("human", "{query}")
])


# General reasoning prompt
REASONING_TEMPLATE = PromptTemplate(
    input_variables=["framework", "query", "context", "chat_history"],
    template="""You are using the {framework} reasoning framework.

Previous conversation:
{chat_history}

Current task: {query}

Context:
{context}

Apply the {framework} methodology to solve this problem."""
)


# Code generation prompt
CODE_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are a code generation expert using {framework}.

Requirements:
- Write clean, well-commented code
- Follow best practices
- Handle edge cases
- Include error handling"""),
    ("human", """Task: {query}

Existing code:
{code_context}

Generate improved code:""")
])
