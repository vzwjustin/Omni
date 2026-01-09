# Omni-Cortex Product Overview

Omni-Cortex is an MCP (Model Context Protocol) server that provides 62+ AI reasoning frameworks for code-focused tasks. It acts as a "thinking framework library" that enhances IDE agents with structured reasoning patterns.

## Core Concept

- **Pass-through design**: Server doesn't call LLMs itself - returns optimized prompts for your IDE's LLM to execute
- **Framework orchestration**: Each framework implements real multi-turn reasoning flows (not just prompt templates)
- **Token efficient**: ~60-80 tokens per framework call vs 150+ for verbose prompts
- **IDE agnostic**: Works with any MCP-compatible IDE (Claude Desktop, Cursor, Windsurf, etc.)

## Key Features

- 62 thinking frameworks across 7 categories (Strategy, Search, Iterative, Code, Context, Fast, Verification)
- Smart framework selection via HyperRouter with vibe-based matching
- Optional RAG capabilities with ChromaDB for code search and memory
- LangGraph workflow orchestration with LangChain memory integration
- Docker-based deployment with automated setup

## Target Use Cases

- Complex debugging and root cause analysis
- Architecture and system design decisions
- Code quality improvement and security review
- Multi-step problem solving and planning
- Learning from failures and iterative improvement