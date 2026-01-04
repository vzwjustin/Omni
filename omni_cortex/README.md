# Omni-Cortex

40 thinking frameworks for your IDE. Just describe what you want - the router picks the right framework.

## Install

Tell your IDE:
> "Install the omni-cortex MCP server"

Or run:
```bash
curl -fsSL https://raw.githubusercontent.com/vzwjustin/thinking-frameworks/initial-release/omni_cortex/install.sh | bash
```

Or manually pull from Docker Hub and add to your MCP config:
```bash
docker pull vzwjustin/omni-cortex:latest
```

```json
{
  "mcpServers": {
    "omni-cortex": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "vzwjustin/omni-cortex:latest"]
    }
  }
}
```

Requires Docker. That's it.

## Use

Just talk naturally:

| Say this | Gets you |
|----------|----------|
| "wtf is wrong with this" | `active_inference` - hypothesis testing |
| "clean this up" | `graph_of_thoughts` - structured refactoring |
| "make it faster" | `tree_of_thoughts` - optimization paths |
| "pros and cons" | `multi_agent_debate` - argue both sides |
| "quick fix" | `system1` - fast response |
| "major rewrite" | `everything_of_thought` - full exploration |
| "is this secure" | `chain_of_verification` - security audit |

Or call directly: `think_active_inference`, `think_tree_of_thoughts`, etc.

## 40 Frameworks

| Category | Frameworks |
|----------|------------|
| **Strategy** | reason_flux, self_discover, buffer_of_thoughts, coala, least_to_most, comparative_arch, plan_and_solve |
| **Search** | mcts_rstar, tree_of_thoughts, graph_of_thoughts, everything_of_thought |
| **Iterative** | active_inference, multi_agent_debate, adaptive_injection, re2, rubber_duck, react, reflexion, self_refine |
| **Code** | program_of_thoughts, chain_of_verification, critic, chain_of_code, self_debugging, tdd_prompting, reverse_cot, alphacodium, codechain, evol_instruct, llmloop, procoder, recode |
| **Context** | chain_of_note, step_back, analogical, red_team, state_machine, chain_of_thought |
| **Fast** | skeleton_of_thought, system1 |

## How it Works

Your IDE calls Omni-Cortex → gets a structured prompt → IDE's LLM executes it.

Pass-through architecture. No API keys needed.

## Docs

- [MCP_SETUP.md](MCP_SETUP.md) - Manual IDE configs
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
- [setup.sh](setup.sh) - Dev setup (clone + build)

## License

MIT
