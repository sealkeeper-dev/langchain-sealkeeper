# langchain-sealkeeper

Check an AI agent or MCP server on [SealKeeper](https://sealkeeper.run) from LangChain, before your agent calls it.

SealKeeper issues the SEAL, a signed credential. This package checks it on your machine with the [`sealkeeper`](https://pypi.org/project/sealkeeper/) library, and every decision is the library's `Gate`.

A SEAL comes in two kinds. An Agent SEAL is for an AI agent, such as a remote A2A agent. A Service SEAL is for a service, such as an MCP server.

## Install

```sh
pip install langchain-sealkeeper
```

Python 3.10 or later. It depends on `sealkeeper`, `langchain-core` and `langchain`. The MCP example also needs `langchain-mcp-adapters`.

In each example, `model` is any LangChain chat model that calls tools.

To gate an MCP server or an A2A agent by its URL, copy the URL from its page on SealKeeper, under Use and verify. Only that exact URL finds it. A trailing slash or a query string finds nothing, and the gate blocks the call as `not_found`.

## Look up

`SealKeeperLookupTool` lets the model look a target up.

```python
from langchain.agents import create_agent
from langchain_sealkeeper import SealKeeperLookupTool

agent = create_agent(model, tools=[SealKeeperLookupTool()])
question = "Check aibenchmarkindex.com/ai-benchmark-index on SealKeeper."
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
```

The tool is `sealkeeper_lookup`, with one argument, `target`. A target is a handle such as `aibenchmarkindex.com/ai-benchmark-index`, a SealKeeper page URL, an agent id, a `did:key`, an A2A card URL or an MCP server URL.

The model reads one line, such as `aibenchmarkindex.com/ai-benchmark-index: verified agent SEAL, level bronze, Trust Score 200.` or `alice/helper: no verified SEAL (not_found).`

The `ToolMessage` artifact is the lookup as a dict of JSON values, so a checkpointer keeps it as it is.

```python
lookup = next(m for m in result["messages"] if m.type == "tool").artifact
if lookup["verified"]:
    seal = lookup["seal"]
    print(seal["kind"], seal["level"], seal["trust"])
else:
    print(lookup["code"])
```

`verified` says whether the SEAL verified, and `seal` holds it when it did. Without one, `code` says why, as in the `sealkeeper` library's `Lookup`. A broken SEAL reads `verified` false and no level, kind or Trust Score.

A lookup that fails, such as SealKeeper not answering in time, answers an error `ToolMessage` with the reason.

## Gate an MCP server

`SealGateInterceptor` checks an MCP server's SEAL before each call to one of its tools. The example is `stoop-mcp-server`, which holds a bronze Service SEAL.

```python
import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_sealkeeper import SealGateInterceptor

connections = {
    "stoop": {
        "transport": "streamable_http",
        "url": "https://stoop-mcp-server.stoop.workers.dev/mcp",
    },
}


async def main():
    gate = SealGateInterceptor(connections, ["stoop"])
    client = MultiServerMCPClient(connections, tool_interceptors=[gate])
    tools = await client.get_tools()
    agent = create_agent(model, tools=tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Find a plumber in Miami."}]}
    )
    print(result["messages"][-1].content)


asyncio.run(main())
```

Pass it the same `connections` as the client. It reads each server's `url` from them at every call, as a service. Load the tools with `client.get_tools()`, or with `load_mcp_tools(session, server_name=...)` and the connection's name, since a tool loaded without its server's name is not gated.

- A pass calls the tool.
- A block answers an error `ToolMessage` with the code and the reason, such as `SealKeeper blocked this call: not_found. SealKeeper knows no agent or service by that target`. The tool is not called and the agent's run goes on. The message has no `name`, and outside a LangGraph `ToolNode` its `tool_call_id` is empty.
- A server you do not name passes untouched.
- A stdio server has no URL to check, so naming one raises `ValueError`.

## Gate a remote agent

`SealGateMiddleware` checks a target's SEAL before each call to a tool that calls it. The example is AI Benchmark Index, an A2A agent that holds a bronze Agent SEAL. The tool sends an A2A `message/send` to the endpoint its card names in `url`.

```python
import uuid

import httpx
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_sealkeeper import SealGateMiddleware, SealTarget

CARD_URL = "https://aibenchmarkindex.com/.well-known/agent-card.json"
ENDPOINT = "https://aibenchmarkindex.com/a2a"


@tool
def ask_benchmark_index(question: str) -> str:
    """Ask AI Benchmark Index, a remote A2A agent, to find AI benchmarks."""
    message = {
        "kind": "message",
        "messageId": str(uuid.uuid4()),
        "role": "user",
        "parts": [{"kind": "text", "text": question}],
    }
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {"message": message},
    }
    answer = httpx.post(ENDPOINT, json=request, timeout=30)
    return answer.text


gate = SealGateMiddleware({"ask_benchmark_index": SealTarget("agent", CARD_URL)})
agent = create_agent(model, tools=[ask_benchmark_index], middleware=[gate])
question = "List benchmarks in the Coding category."
result = agent.invoke({"messages": [{"role": "user", "content": question}]})
```

Each pinned tool name maps to a `SealTarget(kind, ref)`. `kind` is `"agent"` or `"service"`, and `ref` is any target the lookup takes. Pin it to the card URL, so the SEAL must name the card you read the endpoint from.

A block answers an error `ToolMessage` with the code and the reason, the tool is not called and the run goes on. A tool you do not pin passes untouched. It works with `invoke` and `ainvoke`.

## The rules

Both gates decide with a `sealkeeper.Gate` and its default rules. A pass needs a current, valid SEAL of the kind at bronze or above, and for a card or MCP server URL the SEAL must name that URL.

`allow_unsealed=True` on either gate also lets through a target SealKeeper does not know, one it is still checking and one with no SEAL while it is dormant. A withheld or broken SEAL, a delisted target and SealKeeper unavailable still block.

```python
gate = SealGateInterceptor(connections, ["stoop"], allow_unsealed=True)
```

The block codes are in the [`sealkeeper` README](https://pypi.org/project/sealkeeper/). Each decision writes one INFO line to the `sealkeeper` logger, never the SEAL, a full URL or the tool's arguments.

The guide at [sealkeeper.run/docs/langchain](https://sealkeeper.run/docs/langchain) walks through each example.

## License

MIT
