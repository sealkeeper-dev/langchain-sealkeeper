# langchain-sealkeeper

Check an AI agent or MCP server on [SealKeeper](https://sealkeeper.run) from LangChain, before your agent calls it.

A SEAL is SealKeeper's signed credential for an agent or a service. This package wires the [`sealkeeper`](https://pypi.org/project/sealkeeper/) library into LangChain. Every SEAL is verified on your machine and every decision is the library's `Gate`.

## Install

```sh
pip install langchain-sealkeeper
```

Python 3.10 or later. It depends on `sealkeeper`, `langchain-core` and `langchain`.

## Look up

`SealKeeperLookupTool` lets the model look a target up.

```python
from langchain.agents import create_agent
from langchain_sealkeeper import SealKeeperLookupTool

agent = create_agent(model, tools=[SealKeeperLookupTool()])
```

The tool is `sealkeeper_lookup`, with one argument, `target`. A target is a handle such as `alice/helper`, a SealKeeper page URL, an agent id, a `did:key`, an A2A card URL or an MCP server URL.

The model reads one line, such as `alice/helper: verified agent SEAL, level bronze, Trust Score 412.` or `alice/helper: no verified SEAL (broken, bad_signature).`

The `ToolMessage` artifact is the lookup as a dict of JSON values, so a checkpointer keeps it as it is. `verified` says whether the SEAL verified, and `seal` holds it when it did. Without one, `code` says why, as in the `sealkeeper` library's `Lookup`. A broken SEAL reads `verified` false and no level, kind or Trust Score.

A lookup that fails, such as SealKeeper not answering in time, answers an error `ToolMessage` with the reason.

## Gate an MCP server

`SealGateInterceptor` checks an MCP server's SEAL before each call to one of its tools. It needs `langchain-mcp-adapters`.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_sealkeeper import SealGateInterceptor

connections = {
    "weather": {"transport": "streamable_http", "url": "https://mcp.example.com/mcp"},
    "notes": {"transport": "stdio", "command": "notes-server", "args": []},
}
gate = SealGateInterceptor(connections, ["weather"])
client = MultiServerMCPClient(connections, tool_interceptors=[gate])
tools = await client.get_tools()
```

Pass it the same `connections` as the client. It reads each server's `url` from them at every call, as a service. Load the tools with `client.get_tools()`, or with `load_mcp_tools(session, server_name=...)` and the connection's name, since a tool loaded without its server's name is not gated.

- A pass calls the tool.
- A block answers an error `ToolMessage` with the code and the reason, such as `SealKeeper blocked this call: not_found. SealKeeper knows no agent or service by that target`. The tool is not called and the agent's run goes on. The message has no `name`, and outside a LangGraph `ToolNode` its `tool_call_id` is empty.
- A server you do not name passes untouched.
- A stdio server has no URL to check, so naming one raises `ValueError`.

## Gate a remote agent

`SealGateMiddleware` checks a target's SEAL before each call to a tool that calls it, such as a tool that sends a task to a remote A2A agent.

```python
from langchain.agents import create_agent
from langchain_sealkeeper import SealGateMiddleware, SealTarget

gate = SealGateMiddleware({
    "ask_helper": SealTarget("agent", "alice/helper"),
    "fetch_report": SealTarget("service", "https://mcp.example.com/mcp"),
})
agent = create_agent(model, tools=[ask_helper, fetch_report, search], middleware=[gate])
```

Each pinned tool name maps to a `SealTarget(kind, ref)`. `kind` is `"agent"` or `"service"`, and `ref` is any target the lookup takes. A block answers an error `ToolMessage` with the code and the reason, the tool is not called and the run goes on. A tool you do not pin passes untouched. It works with `invoke` and `ainvoke`.

## The rules

Both gates decide with a `sealkeeper.Gate` and its default rules. A pass needs a current, valid SEAL of the kind at bronze or above, and for a card or MCP server URL the SEAL must name that URL.

`allow_unsealed=True` on either gate also lets through a target SealKeeper does not know, one it is still checking and one with no SEAL while it is dormant. A withheld or broken SEAL, a delisted target and SealKeeper unavailable still block.

```python
gate = SealGateInterceptor(connections, ["weather"], allow_unsealed=True)
```

The block codes are in the [`sealkeeper` README](https://pypi.org/project/sealkeeper/). Each decision writes one INFO line to the `sealkeeper` logger, never the SEAL, a full URL or the tool's arguments.

## License

MIT
