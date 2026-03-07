#!/usr/bin/env python3
"""
Mnemosyne MCP Stdio Proxy

Bridges VS Code's stdio MCP client to the remote Mnemosyne HTTP server.
Uses the official MCP library for proper protocol handling.

Usage:
    python mnemosyne_proxy.py

Environment:
    MNEMOSYNE_URL  - HTTP endpoint (default: http://localhost:8010/mcp)
"""
import asyncio
import json
import os
import httpx
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

MNEMOSYNE_URL = os.environ.get("MNEMOSYNE_URL", "http://localhost:8010/mcp")
TIMEOUT = 30.0

server = Server("mnemosyne")


async def call_remote_tool(tool_name: str, arguments: dict) -> dict:
    """Forward a tool call to the remote Mnemosyne server."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            MNEMOSYNE_URL,
            json=request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            raise Exception(result["error"].get("message", "Unknown error"))

        return result.get("result", {})


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Mnemosyne tools."""
    return [
        Tool(
            name="mnemosyne_bootstrap",
            description="Return startup context with pinned and recent memory items.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit_pinned": {"type": "integer", "default": 8},
                    "limit_recent": {"type": "integer", "default": 10},
                    "workspace_hint": {"type": "string"},
                    "mode": {"type": "string", "enum": ["thin", "hybrid", "full"]},
                    "max_tokens": {"type": "integer"},
                    "max_items": {"type": "integer"},
                    "include_sessions": {"type": "boolean"},
                },
            },
        ),
        Tool(
            name="mnemosyne_write",
            description="Store a memory item (deduplicates by kind+title).",
            inputSchema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "tags_json": {"type": "string", "default": "[]"},
                    "pinned": {"type": "boolean", "default": False},
                    "content_compact": {"type": "string"},
                    "workspace_hint": {"type": "string"},
                    "importance": {"type": "integer"},
                    "source": {"type": "string"},
                },
                "required": ["kind", "title", "content"],
            },
        ),
        Tool(
            name="mnemosyne_read",
            description="Read a single memory item by id (on-demand expansion).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "prefer": {"type": "string", "enum": ["full", "compact"]},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="mnemosyne_search",
            description="Search memory using keyword, semantic, or hybrid retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 8},
                    "prefer": {"type": "string", "enum": ["compact", "full"]},
                    "snippet_chars": {"type": "integer"},
                    "method": {
                        "type": "string",
                        "enum": ["keyword", "semantic", "hybrid"],
                        "description": "Search method (default: hybrid)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="mnemosyne_commit_session",
            description="Commit session summary at end of coding session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace_hint": {"type": "string"},
                    "summary": {"type": "string"},
                    "decisions_json": {"type": "string", "default": "[]"},
                    "next_steps_json": {"type": "string", "default": "[]"},
                },
                "required": ["workspace_hint", "summary"],
            },
        ),
        Tool(
            name="mnemosyne_last_session",
            description="Get most recent session logs for a workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace_hint": {"type": "string", "default": "global"},
                    "limit": {"type": "integer", "default": 3},
                },
            },
        ),
        Tool(
            name="mnemosyne_ingest",
            description="Ingest a document: chunks, embeds, and stores for RAG retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "source": {"type": "string"},
                    "mime_type": {"type": "string"},
                    "workspace_hint": {"type": "string"},
                    "chunk_size": {"type": "integer"},
                    "chunk_overlap": {"type": "integer"},
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="mnemosyne_ask",
            description="RAG question answering: retrieves context and generates an answer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "workspace_hint": {"type": "string"},
                    "max_context_items": {"type": "integer"},
                    "method": {"type": "string", "enum": ["keyword", "semantic", "hybrid"]},
                    "include_chunks": {"type": "boolean"},
                    "rerank": {"type": "boolean"},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="mnemosyne_backfill_embeddings",
            description="Backfill embeddings for memory items that don't have them yet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls by forwarding to the remote Mnemosyne server."""
    try:
        result = await call_remote_tool(name, arguments)
        if isinstance(result, (dict, list)):
            text = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            text = str(result)
        return [TextContent(type="text", text=text)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error calling {name}: {str(e)}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
