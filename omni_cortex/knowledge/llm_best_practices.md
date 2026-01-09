# LLM Debugging & Reasoning Knowledge Base
# Pre-seeded content for Omni-Cortex

## Prompt Engineering Best Practices

When crafting prompts for LLMs:
- Be specific and explicit about what you want
- Use structured output formats (JSON, XML, Markdown)
- Provide examples (few-shot prompting) for complex tasks
- Break complex tasks into smaller steps (chain-of-thought)
- Use system prompts to set context and constraints
- Include negative examples to show what NOT to do
- Test with edge cases and adversarial inputs
- Use temperature 0 for deterministic outputs, higher for creativity

## Debugging LLM Responses

When LLM outputs are incorrect or unexpected:
1. **Check the prompt**: Is it ambiguous? Missing context?
2. **Verify input format**: Malformed JSON/data can confuse models
3. **Token limits**: Response may be truncated - check max_tokens
4. **Temperature**: Too high causes randomness, too low lacks creativity
5. **Model mismatch**: Different models have different capabilities
6. **Context window**: Earlier context may be "forgotten" in long conversations
7. **Hallucinations**: Add grounding/retrieval for factual queries
8. **Instruction following**: Some models need explicit "You MUST..." phrasing

## Chain-of-Thought Reasoning

Chain-of-thought (CoT) improves accuracy on reasoning tasks:
- Ask the model to "think step by step"
- Break problems into sub-problems
- Have the model show its work before the final answer
- Use phrases like "Let's work through this carefully"
- For math: require intermediate calculations
- For code: require pseudocode before implementation
- Validate each step before proceeding

## Common LLM Failure Modes

Understanding common failures helps debug faster:
- **Sycophancy**: Model agrees with user even when wrong - ask for honest assessment
- **Positional bias**: Earlier/later items in lists get different attention
- **Format lock-in**: Model copies format from examples too rigidly
- **Refusal cascade**: One refusal triggers more refusals
- **Context contamination**: Previous conversation affects current response
- **Capability overhang**: Model can solve but needs the right prompt
- **Token boundary issues**: Words split across tokens cause problems

## Reasoning Framework Selection

Choose frameworks based on problem type:
- **Chain-of-Thought**: Step-by-step reasoning for math/logic
- **Self-Consistency**: Generate multiple answers, pick majority
- **Tree-of-Thoughts**: Explore multiple reasoning paths
- **ReAct**: Interleave reasoning and actions for tool use
- **Reflexion**: Self-critique and retry on failures
- **Debate**: Multiple "agents" argue different positions
- **Mixture of Experts**: Route to specialists for domains

## Code Generation Debugging

When LLM-generated code fails:
1. Check for syntax errors (missing imports, wrong indentation)
2. Verify function signatures match documentation
3. Test with minimal examples first
4. Look for hallucinated APIs that don't exist
5. Check version compatibility (libraries change)
6. Validate edge cases (empty lists, null values)
7. Review error handling (try/except coverage)
8. Run type checking (mypy) on generated code

## Context Management

Managing context effectively:
- Summarize long conversations to preserve important info
- Use RAG to retrieve relevant context dynamically
- Structure context clearly (SYSTEM, USER, ASSISTANT)
- Remove irrelevant context to reduce noise
- Front-load important information (primacy effect)
- Repeat key instructions at the end (recency effect)

## Multi-Turn Conversation Best Practices

For multi-turn dialogues:
- Maintain consistent persona/system prompt
- Track state across turns explicitly
- Summarize decisions made so far
- Handle topic switches gracefully
- Detect and recover from misunderstandings
- Use conversation IDs for thread management
- Implement graceful degradation on errors

## Error Recovery Strategies

When things go wrong:
- Retry with clearer instructions
- Break the task into smaller pieces
- Provide more examples
- Try a different model
- Add explicit constraints
- Use a critic to validate outputs
- Implement circuit breakers for cascading failures

## Prompt Injection Defense

Protecting against prompt attacks:
- Validate and sanitize user inputs
- Use separate system and user contexts
- Implement output filtering
- Avoid executing LLM-generated code directly
- Use structured outputs to limit free-form responses
- Monitor for anomalous patterns
- Rate limit requests

## Performance Optimization

Making LLM apps faster:
- Use streaming for long responses
- Cache common queries
- Batch similar requests
- Use smaller models for simple tasks
- Implement progressive disclosure
- Parallelize independent calls
- Use async/await for concurrency

## Testing LLM Applications

Quality assurance strategies:
- Create golden test sets with expected outputs
- Use semantic similarity for fuzzy matching
- Test edge cases and adversarial inputs
- Monitor for regression on model updates
- A/B test prompt variations
- Track metrics: latency, accuracy, cost
- Implement shadow testing for new models

## Structured Output Patterns

Getting reliable structured data:
- Request JSON with a schema example
- Use XML tags for clear boundaries
- Implement output parsers with fallbacks
- Validate against schemas (Pydantic, JSON Schema)
- Handle partial/malformed responses gracefully
- Use constrained decoding when available

## Tool/Function Calling Best Practices

When LLMs use tools:
- Provide clear tool descriptions
- Include examples of when to use each tool
- Validate tool arguments before execution
- Handle tool failures gracefully
- Limit available tools to reduce confusion
- Log all tool calls for debugging
- Implement tool result summarization

## Memory and State Management

For stateful applications:
- Use vector stores for semantic memory
- Implement explicit state objects
- Serialize/deserialize cleanly between turns
- Handle state corruption gracefully
- Implement memory consolidation for long sessions
- Use checksums for integrity verification

## Cost Optimization

Managing API costs:
- Use caching for repeated queries
- Choose appropriate model size
- Truncate context to minimum needed
- Batch process when possible
- Use cheaper models for classification/routing
- Monitor and alert on spend
- Implement usage quotas per user

## Model Selection Guidelines

Choosing the right model:
- GPT-4/Claude 3 Opus: Complex reasoning, long context
- GPT-3.5/Claude 3 Haiku: Simple tasks, speed priority
- Gemini Flash: Cost-effective, good at structured output
- Specialized models: Code (Codex), embeddings, vision
- Local models: Privacy requirements, offline use
- Fine-tuned models: Domain-specific tasks
