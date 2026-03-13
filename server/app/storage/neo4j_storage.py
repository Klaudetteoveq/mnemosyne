"""
Neo4j knowledge graph storage backend for Mnemosyne.

Graph Schema:
  (:MemoryItem {id, kind, title, content, content_compact, created_at, updated_at,
                pinned, importance, workspace_hint, source})
    -[:TAGGED_WITH]-> (:Tag {name})
    -[:DECIDED_IN]-> (:Session)
    -[:RELATES_TO]-> (:MemoryItem)

  (:Session {id, workspace_hint, summary, created_at})
    -[:FOLLOWS]-> (:Session)
    -[:IN_WORKSPACE]-> (:Workspace {name})
    -[:HAS_DECISION]-> (decision:string)
    -[:HAS_NEXT_STEP]-> (next_step:string)
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from .base import MemoryStorage, RequestContext, BootstrapMode, ContentPrefer

logger = logging.getLogger(__name__)

# RAG configuration
EMBED_ON_WRITE = os.environ.get("MNEMOSYNE_EMBED_ON_WRITE", "0").strip() in ("1", "true", "yes")
RRF_K = 60  # Reciprocal Rank Fusion constant

VALID_KINDS = {"answer", "decision", "pattern", "command", "note"}

# --- Context pollution mitigation: ranking constants ---
KIND_WEIGHTS: dict[str, float] = {
    "decision": 1.4,
    "pattern": 1.3,
    "command": 1.2,
    "answer": 1.1,
    "note": 0.7,
}
RECENCY_HALF_LIFE_DAYS = 14.0
WORKSPACE_MATCH_BOOST = 1.2
WORKSPACE_MISMATCH_PENALTY = 0.8
# Max chars for auto-generated compact content
AUTO_COMPACT_MAX_CHARS = 200
# Kinds eligible for full content in "hybrid" mode (if short enough)
HYBRID_FULL_KINDS = {"command", "pattern"}
HYBRID_FULL_MAX_CHARS = 300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auto_compact(content: str, max_chars: int = AUTO_COMPACT_MAX_CHARS) -> str:
    """Generate a compact snippet from full content.

    Deterministic heuristic: take first ``max_chars`` characters, break at
    the last sentence boundary (period/newline) if possible, append "…".
    """
    content = (content or "").strip()
    if len(content) <= max_chars:
        return content
    truncated = content[:max_chars]
    # Try to break at a sentence boundary
    for sep in ("\n", ". ", "! ", "? "):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            truncated = truncated[: idx + len(sep)].rstrip()
            break
    return truncated + "…"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return math.ceil(len(text) / 4) if text else 0


def _recency_weight(updated_at: str, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """Half-life decay weight based on age."""
    try:
        updated = datetime.fromisoformat(updated_at)
        age = datetime.now(timezone.utc) - updated
        age_days = max(age.total_seconds() / 86400, 0)
        return 0.5 ** (age_days / half_life_days)
    except (ValueError, TypeError):
        return 0.5  # fallback for unparseable dates


def _score_item(item: dict, workspace_hint: str = "global") -> float:
    """Score a memory item for ranking. Higher = more relevant."""
    kind = item.get("kind", "note")
    w_kind = KIND_WEIGHTS.get(kind, 0.7)
    w_recency = _recency_weight(item.get("updated_at", ""))
    importance = item.get("importance", 50) or 50
    item_workspace = item.get("workspace_hint") or ""
    if workspace_hint and workspace_hint != "global" and item_workspace:
        w_workspace = WORKSPACE_MATCH_BOOST if item_workspace == workspace_hint else WORKSPACE_MISMATCH_PENALTY
    else:
        w_workspace = 1.0
    return w_kind * w_recency * (0.5 + importance / 100) * w_workspace


def _select_content_for_mode(
    item: dict,
    mode: BootstrapMode,
) -> str:
    """Pick the right content string based on bootstrap mode."""
    content_compact = item.get("content_compact") or ""
    content_full = item.get("content") or ""
    if mode == "full":
        return content_full
    if mode == "hybrid":
        kind = item.get("kind", "note")
        if kind in HYBRID_FULL_KINDS and len(content_full) <= HYBRID_FULL_MAX_CHARS:
            return content_full
        return content_compact or _auto_compact(content_full)
    # thin
    return content_compact or _auto_compact(content_full)


def _render_item_thin(item: dict) -> str:
    """Render a memory item in thin format for bootstrap."""
    kind = item.get("kind", "note")
    title = item.get("title", "")
    content = item.get("content", "")
    tags = item.get("tags", "[]")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, TypeError):
            tags = []
    updated = item.get("updated_at", "")
    tag_str = ",".join(tags) if tags else ""
    lines = [f"[{kind}] {title}"]
    if content:
        # Indent content bullets
        for line in content.split("\n")[:3]:
            line = line.strip()
            if line:
                lines.append(f"  {line}")
    meta_parts = []
    if tag_str:
        meta_parts.append(f"tags: {tag_str}")
    if updated:
        meta_parts.append(f"updated: {updated[:19]}")
    if meta_parts:
        lines.append(f"  {' | '.join(meta_parts)}")
    return "\n".join(lines)


class Neo4jStorage(MemoryStorage):
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "mnemosyne",
        database: str = "neo4j",
        multi_tenant: bool | None = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: AsyncDriver | None = None
        # Feature flag for multi-tenancy; default from env `MNEMOSYNE_MULTI_TENANT`
        if multi_tenant is None:
            env_val = os.environ.get("MNEMOSYNE_MULTI_TENANT", "0").strip()
            self._multi_tenant = env_val in ("1", "true", "True", "yes")
        else:
            self._multi_tenant = bool(multi_tenant)

    async def initialize(self) -> None:
        """Connect to Neo4j and create indexes/constraints."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

        # Verify connectivity
        async with self._driver.session(database=self.database) as session:
            await session.run("RETURN 1")

        # Create constraints and indexes
        async with self._driver.session(database=self.database) as session:
            # Unique constraint on MemoryItem kind+title for dedup
            await session.run(
                "CREATE INDEX memory_item_kind_title IF NOT EXISTS "
                "FOR (m:MemoryItem) ON (m.kind, m.title)"
            )
            # Index for pinned lookups
            await session.run(
                "CREATE INDEX memory_item_pinned IF NOT EXISTS "
                "FOR (m:MemoryItem) ON (m.pinned)"
            )
            # Index for updated_at ordering
            await session.run(
                "CREATE INDEX memory_item_updated IF NOT EXISTS "
                "FOR (m:MemoryItem) ON (m.updated_at)"
            )
            # Fulltext index for search (includes content_compact)
            try:
                await session.run(
                    "CREATE FULLTEXT INDEX memory_fulltext IF NOT EXISTS "
                    "FOR (m:MemoryItem) ON EACH [m.title, m.content, m.content_compact]"
                )
            except Exception as e:
                # Fulltext index might already exist with different config
                logger.warning("Fulltext index creation: %s", e)

            # Index for workspace_hint scoping
            await session.run(
                "CREATE INDEX memory_item_workspace IF NOT EXISTS "
                "FOR (m:MemoryItem) ON (m.workspace_hint)"
            )

            # Tag uniqueness
            await session.run(
                "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS "
                "FOR (t:Tag) REQUIRE t.name IS UNIQUE"
            )
            # Workspace uniqueness
            await session.run(
                "CREATE CONSTRAINT workspace_name_unique IF NOT EXISTS "
                "FOR (w:Workspace) REQUIRE w.name IS UNIQUE"
            )
            # Space id uniqueness (for multi-tenancy)
            await session.run(
                "CREATE CONSTRAINT space_id_unique IF NOT EXISTS "
                "FOR (s:Space) REQUIRE s.id IS UNIQUE"
            )
            # Session index
            await session.run(
                "CREATE INDEX session_created IF NOT EXISTS "
                "FOR (s:Session) ON (s.created_at)"
            )
            await session.run(
                "CREATE INDEX session_workspace IF NOT EXISTS "
                "FOR (s:Session) ON (s.workspace_hint)"
            )
            await session.run(
                "CREATE INDEX session_space IF NOT EXISTS "
                "FOR (s:Session) ON (s.space_id)"
            )

            # Compound index to enforce per-space dedup by (kind, title)
            await session.run(
                "CREATE INDEX memory_item_space_kind_title IF NOT EXISTS "
                "FOR (m:MemoryItem) ON (m.space_id, m.kind, m.title)"
            )

            # --- RAG: Vector indexes ---
            try:
                await session.run(
                    """
                    CREATE VECTOR INDEX memory_embedding IF NOT EXISTS
                    FOR (m:MemoryItem)
                    ON (m.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                    """
                )
            except Exception as e:
                logger.warning("Memory vector index creation: %s", e)

            try:
                await session.run(
                    """
                    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                    FOR (c:Chunk)
                    ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                    """
                )
            except Exception as e:
                logger.warning("Chunk vector index creation: %s", e)

            # Document indexes
            await session.run(
                "CREATE INDEX document_workspace IF NOT EXISTS "
                "FOR (d:Document) ON (d.workspace_hint)"
            )
            await session.run(
                "CREATE INDEX document_ingested IF NOT EXISTS "
                "FOR (d:Document) ON (d.ingested_at)"
            )

        logger.info("Neo4j storage initialized at %s", self.uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def write_memory(
        self,
        kind: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        pinned: bool = False,
        content_compact: str | None = None,
        workspace_hint: str | None = None,
        importance: int | None = None,
        source: str | None = None,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        kind = (kind or "").strip().lower()
        if kind not in VALID_KINDS:
            kind = "note"
        title = title.strip()
        content = content.strip()
        tags = tags or []
        now = _now()

        # Auto-generate compact content if not provided
        if content_compact is None:
            content_compact = _auto_compact(content)
        else:
            content_compact = content_compact.strip()

        # Normalize importance (0-100, default 50)
        if importance is None:
            importance = 50
        importance = max(0, min(100, importance))

        # Normalize source
        source = (source or "agent").strip()

        # Normalize workspace_hint
        workspace_hint = (workspace_hint or "").strip() or None

        async with self._driver.session(database=self.database) as session:
            if self._multi_tenant:
                space_id, _ = self._derive_space_and_allowed(context)
                # Ensure space exists and upsert memory within space scope
                result = await session.run(
                    """
                    MERGE (s:Space {id: $space_id})
                    MERGE (m:MemoryItem {space_id: $space_id, kind: $kind, title: $title})
                    ON CREATE SET
                        m.content = $content,
                        m.content_compact = $content_compact,
                        m.created_at = $now,
                        m.updated_at = $now,
                        m.pinned = $pinned,
                        m.importance = $importance,
                        m.workspace_hint = $workspace_hint,
                        m.source = $source
                    ON MATCH SET
                        m.content = $content,
                        m.content_compact = $content_compact,
                        m.updated_at = $now,
                        m.pinned = $pinned,
                        m.importance = $importance,
                        m.workspace_hint = $workspace_hint,
                        m.source = $source
                    WITH s, m,
                         CASE WHEN m.created_at = $now THEN 'created' ELSE 'updated' END AS action
                    MERGE (s)-[:CONTAINS]->(m)
                    RETURN elementId(m) AS id, action
                    """,
                    space_id=space_id,
                    kind=kind,
                    title=title,
                    content=content,
                    content_compact=content_compact,
                    now=now,
                    pinned=pinned,
                    importance=importance,
                    workspace_hint=workspace_hint,
                    source=source,
                )
            else:
                # Legacy single-tenant behavior
                result = await session.run(
                    """
                    MERGE (m:MemoryItem {kind: $kind, title: $title})
                    ON CREATE SET
                        m.content = $content,
                        m.content_compact = $content_compact,
                        m.created_at = $now,
                        m.updated_at = $now,
                        m.pinned = $pinned,
                        m.importance = $importance,
                        m.workspace_hint = $workspace_hint,
                        m.source = $source
                    ON MATCH SET
                        m.content = $content,
                        m.content_compact = $content_compact,
                        m.updated_at = $now,
                        m.pinned = $pinned,
                        m.importance = $importance,
                        m.workspace_hint = $workspace_hint,
                        m.source = $source
                    WITH m,
                         CASE WHEN m.created_at = $now THEN 'created' ELSE 'updated' END AS action
                    RETURN elementId(m) AS id, action
                    """,
                    kind=kind,
                    title=title,
                    content=content,
                    content_compact=content_compact,
                    now=now,
                    pinned=pinned,
                    importance=importance,
                    workspace_hint=workspace_hint,
                    source=source,
                )
            record = await result.single()
            item_id = record["id"]
            action = record["action"]

            # Remove old tag relationships and create new ones
            if self._multi_tenant:
                await session.run(
                    "MATCH (m:MemoryItem {space_id: $space_id, kind: $kind, title: $title})-[r:TAGGED_WITH]->() DELETE r",
                    space_id=space_id,
                    kind=kind,
                    title=title,
                )
            else:
                await session.run(
                    "MATCH (m:MemoryItem {kind: $kind, title: $title})-[r:TAGGED_WITH]->() DELETE r",
                    kind=kind,
                    title=title,
                )

            for tag_name in tags:
                tag_name = tag_name.strip()
                if tag_name:
                    if self._multi_tenant:
                        await session.run(
                            """
                            MATCH (m:MemoryItem {space_id: $space_id, kind: $kind, title: $title})
                            MERGE (t:Tag {name: $tag})
                            MERGE (m)-[:TAGGED_WITH]->(t)
                            """,
                            space_id=space_id,
                            kind=kind,
                            title=title,
                            tag=tag_name,
                        )
                    else:
                        await session.run(
                            """
                            MATCH (m:MemoryItem {kind: $kind, title: $title})
                            MERGE (t:Tag {name: $tag})
                            MERGE (m)-[:TAGGED_WITH]->(t)
                            """,
                            kind=kind,
                            title=title,
                            tag=tag_name,
                        )

            # Embed the item asynchronously (best-effort)
            await self._embed_item(str(item_id), title, content_compact)

            return {"ok": True, "action": action, "id": str(item_id)}

    async def _embed_item(self, item_id: str, title: str, content_compact: str | None) -> None:
        """Compute and store embedding for a memory item (best-effort)."""
        if not EMBED_ON_WRITE:
            return
        try:
            from embedding import get_embedding, embedding_text_for_memory
            text = embedding_text_for_memory(title, content_compact)
            embedding = await get_embedding(text)
            if embedding:
                await self.set_embedding(item_id, embedding)
        except Exception as e:
            logger.warning("Failed to embed item %s: %s", item_id, e)

    async def search_memory(
        self,
        query: str,
        limit: int = 8,
        prefer: ContentPrefer = "full",
        snippet_chars: int = 400,
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        limit = max(1, min(limit, 25))

        async with self._driver.session(database=self.database) as session:
            spaces: list[str] | None = None
            if self._multi_tenant:
                _, allowed = self._derive_space_and_allowed(context)
                spaces = allowed
            # Use fulltext index for search
            try:
                if self._multi_tenant:
                    result = await session.run(
                        """
                        CALL db.index.fulltext.queryNodes('memory_fulltext', $search_text)
                        YIELD node, score
                        WHERE node.space_id IN $spaces
                        OPTIONAL MATCH (node)-[:TAGGED_WITH]->(t:Tag)
                        WITH node, score, collect(t.name) AS tags
                        RETURN
                            elementId(node) AS id,
                            node.kind AS kind,
                            node.title AS title,
                            node.content AS content,
                            node.content_compact AS content_compact,
                            tags,
                            node.pinned AS pinned,
                            node.updated_at AS updated_at,
                            node.importance AS importance,
                            node.workspace_hint AS workspace_hint,
                            score
                        ORDER BY score DESC
                        LIMIT $lim
                        """,
                        search_text=query,
                        lim=limit,
                        spaces=spaces,
                    )
                else:
                    result = await session.run(
                        """
                        CALL db.index.fulltext.queryNodes('memory_fulltext', $search_text)
                        YIELD node, score
                        OPTIONAL MATCH (node)-[:TAGGED_WITH]->(t:Tag)
                        WITH node, score, collect(t.name) AS tags
                        RETURN
                            elementId(node) AS id,
                            node.kind AS kind,
                            node.title AS title,
                            node.content AS content,
                            node.content_compact AS content_compact,
                            tags,
                            node.pinned AS pinned,
                            node.updated_at AS updated_at,
                            node.importance AS importance,
                            node.workspace_hint AS workspace_hint,
                            score
                        ORDER BY score DESC
                        LIMIT $lim
                        """,
                        search_text=query,
                        lim=limit,
                    )
                records = [record.data() async for record in result]
                return self._format_search_results(records, prefer, snippet_chars)
            except Exception as e:
                logger.warning(
                    "Fulltext search failed, falling back to CONTAINS: %s", e
                )
                # Fallback: simple CONTAINS match
                if self._multi_tenant:
                    result = await session.run(
                        """
                        MATCH (m:MemoryItem)
                        WHERE (toLower(m.title) CONTAINS toLower($search_text)
                           OR toLower(m.content) CONTAINS toLower($search_text))
                          AND m.space_id IN $spaces
                        OPTIONAL MATCH (m)-[:TAGGED_WITH]->(t:Tag)
                        WITH m, collect(t.name) AS tags
                        RETURN
                            elementId(m) AS id,
                            m.kind AS kind,
                            m.title AS title,
                            m.content AS content,
                            m.content_compact AS content_compact,
                            tags,
                            m.pinned AS pinned,
                            m.updated_at AS updated_at,
                            m.importance AS importance,
                            m.workspace_hint AS workspace_hint
                        ORDER BY m.updated_at DESC
                        LIMIT $lim
                        """,
                        search_text=query,
                        lim=limit,
                        spaces=spaces,
                    )
                else:
                    result = await session.run(
                        """
                        MATCH (m:MemoryItem)
                        WHERE toLower(m.title) CONTAINS toLower($search_text)
                           OR toLower(m.content) CONTAINS toLower($search_text)
                        OPTIONAL MATCH (m)-[:TAGGED_WITH]->(t:Tag)
                        WITH m, collect(t.name) AS tags
                        RETURN
                            elementId(m) AS id,
                            m.kind AS kind,
                            m.title AS title,
                            m.content AS content,
                            m.content_compact AS content_compact,
                            tags,
                            m.pinned AS pinned,
                            m.updated_at AS updated_at,
                            m.importance AS importance,
                            m.workspace_hint AS workspace_hint
                        ORDER BY m.updated_at DESC
                        LIMIT $lim
                        """,
                        search_text=query,
                        lim=limit,
                    )
                records = [record.data() async for record in result]
                return self._format_search_results(records, prefer, snippet_chars)

    async def bootstrap(
        self,
        limit_pinned: int = 8,
        limit_recent: int = 10,
        workspace_hint: str = "global",
        mode: BootstrapMode = "full",
        max_tokens: int = 0,
        max_items: int = 15,
        include_sessions: bool = False,
        include_index: bool = False,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        limit_pinned = max(0, min(limit_pinned, 25))
        limit_recent = max(0, min(limit_recent, 50))
        max_items = max(1, min(max_items, 50))
        workspace_hint = (workspace_hint or "global").strip()

        async with self._driver.session(database=self.database) as session:
            spaces: list[str] | None = None
            if self._multi_tenant:
                _, allowed = self._derive_space_and_allowed(context)
                spaces = allowed

            # --- Fetch pinned items ---
            pinned_query_fields = """
                        elementId(m) AS id,
                        m.kind AS kind,
                        m.title AS title,
                        m.content AS content,
                        m.content_compact AS content_compact,
                        tags,
                        m.updated_at AS updated_at,
                        m.importance AS importance,
                        m.workspace_hint AS workspace_hint
            """
            if self._multi_tenant:
                pinned_result = await session.run(
                    f"""
                    MATCH (m:MemoryItem {{pinned: true}})
                    WHERE m.space_id IN $spaces
                    OPTIONAL MATCH (m)-[:TAGGED_WITH]->(t:Tag)
                    WITH m, collect(t.name) AS tags
                    RETURN {pinned_query_fields}
                    ORDER BY m.updated_at DESC
                    LIMIT $limit
                    """,
                    limit=limit_pinned,
                    spaces=spaces,
                )
            else:
                pinned_result = await session.run(
                    f"""
                    MATCH (m:MemoryItem {{pinned: true}})
                    OPTIONAL MATCH (m)-[:TAGGED_WITH]->(t:Tag)
                    WITH m, collect(t.name) AS tags
                    RETURN {pinned_query_fields}
                    ORDER BY m.updated_at DESC
                    LIMIT $limit
                    """,
                    limit=limit_pinned,
                )
            pinned_raw = [r.data() async for r in pinned_result]

            # --- Fetch recent items (over-fetch for ranking) ---
            fetch_limit = max(limit_recent * 3, max_items * 2)
            if self._multi_tenant:
                recent_result = await session.run(
                    f"""
                    MATCH (m:MemoryItem)
                    WHERE m.space_id IN $spaces
                    OPTIONAL MATCH (m)-[:TAGGED_WITH]->(t:Tag)
                    WITH m, collect(t.name) AS tags
                    RETURN {pinned_query_fields}
                    ORDER BY m.updated_at DESC
                    LIMIT $limit
                    """,
                    limit=fetch_limit,
                    spaces=spaces,
                )
            else:
                recent_result = await session.run(
                    f"""
                    MATCH (m:MemoryItem)
                    OPTIONAL MATCH (m)-[:TAGGED_WITH]->(t:Tag)
                    WITH m, collect(t.name) AS tags
                    RETURN {pinned_query_fields}
                    ORDER BY m.updated_at DESC
                    LIMIT $limit
                    """,
                    limit=fetch_limit,
                )
            recent_raw = [r.data() async for r in recent_result]

            # --- Fetch last session (if requested) ---
            last_session_data = None
            if include_sessions:
                session_records = await self.last_session(
                    workspace_hint=workspace_hint, limit=1, context=context
                )
                if session_records:
                    last_session_data = session_records[0]

            # --- Rank & budget (Python-side) ---
            pinned_ids = {p["id"] for p in pinned_raw}
            # Remove pinned from recent candidates
            recent_candidates = [r for r in recent_raw if r["id"] not in pinned_ids]

            # Score and sort recent candidates
            for item in recent_candidates:
                item["_score"] = _score_item(item, workspace_hint)
            recent_candidates.sort(key=lambda x: x["_score"], reverse=True)

            # Apply budgeting
            budget = max_tokens * 4 if max_tokens > 0 else float("inf")  # chars
            used = 0
            pinned_out = []
            recent_out = []

            # Pinned items always included (they're pinned for a reason!) — but shaped
            for item in pinned_raw:
                content_text = _select_content_for_mode(item, mode)
                cost = len(content_text) + len(item.get("title", ""))
                formatted = self._format_bootstrap_item(item, content_text)
                pinned_out.append(formatted)
                used += cost
                if len(pinned_out) >= max_items:
                    break

            # Fill recent with budget
            remaining_slots = max_items - len(pinned_out)
            for item in recent_candidates:
                if remaining_slots <= 0:
                    break
                content_text = _select_content_for_mode(item, mode)
                cost = len(content_text) + len(item.get("title", ""))
                if max_tokens > 0 and used + cost > budget:
                    continue  # skip this item, try smaller ones
                formatted = self._format_bootstrap_item(item, content_text)
                recent_out.append(formatted)
                used += cost
                remaining_slots -= 1

            result = {"pinned": pinned_out, "recent": recent_out}
            if include_sessions:
                result["last_session"] = last_session_data

        # Generate knowledge index outside the main session block
        if include_index:
            index_data = await self.generate_knowledge_index(
                workspace_hint=workspace_hint,
                max_tokens=200,  # keep index compact within bootstrap
                context=context,
            )
            result["knowledge_index"] = index_data.get("index", "")

        return result

    def _format_bootstrap_item(self, raw: dict, content_text: str) -> dict:
        """Format a raw Neo4j record into a bootstrap response item."""
        tags = raw.get("tags", [])
        if isinstance(tags, list):
            tags = json.dumps(tags)
        has_full = bool(raw.get("content") and raw.get("content") != content_text)
        return {
            "id": raw["id"],
            "kind": raw.get("kind", "note"),
            "title": raw.get("title", ""),
            "content": content_text,
            "tags": tags,
            "updated_at": raw.get("updated_at", ""),
            "has_full": has_full,
        }

    def _format_search_results(
        self, records: list[dict], prefer: ContentPrefer, snippet_chars: int
    ) -> list[dict[str, Any]]:
        """Format search results with content preference."""
        results = []
        for r in records:
            content_full = r.get("content") or ""
            content_compact = r.get("content_compact") or ""
            has_full = bool(content_full)

            if prefer == "compact":
                if content_compact:
                    content = content_compact
                else:
                    content = _auto_compact(content_full, max_chars=snippet_chars)
            else:
                content = content_full

            results.append({
                "id": r["id"],
                "kind": r["kind"],
                "title": r["title"],
                "content": content,
                "tags": json.dumps(r.get("tags", [])),
                "pinned": 1 if r.get("pinned") else 0,
                "updated_at": r.get("updated_at", ""),
                "has_full": has_full,
            })
        return results

    async def read_memory(
        self,
        item_id: str,
        prefer: ContentPrefer = "full",
        context: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        """Read a single memory item by its Neo4j element id."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (m:MemoryItem)
                WHERE elementId(m) = $item_id
                OPTIONAL MATCH (m)-[:TAGGED_WITH]->(t:Tag)
                WITH m, collect(t.name) AS tags
                RETURN
                    elementId(m) AS id,
                    m.kind AS kind,
                    m.title AS title,
                    m.content AS content,
                    m.content_compact AS content_compact,
                    tags,
                    m.pinned AS pinned,
                    m.updated_at AS updated_at,
                    m.created_at AS created_at,
                    m.importance AS importance,
                    m.workspace_hint AS workspace_hint,
                    m.source AS source
                """,
                item_id=item_id,
            )
            record = await result.single()
            if record is None:
                return None

            r = record.data()
            content_full = r.get("content") or ""
            content_compact = r.get("content_compact") or ""

            if prefer == "compact":
                content = content_compact or _auto_compact(content_full)
            else:
                content = content_full

            return {
                "id": r["id"],
                "kind": r["kind"],
                "title": r["title"],
                "content": content,
                "content_compact": content_compact,
                "content_full": content_full,
                "tags": json.dumps(r.get("tags", [])),
                "pinned": 1 if r.get("pinned") else 0,
                "updated_at": r.get("updated_at", ""),
                "created_at": r.get("created_at", ""),
                "importance": r.get("importance", 50),
                "workspace_hint": r.get("workspace_hint", ""),
                "source": r.get("source", ""),
            }

    async def commit_session(
        self,
        workspace_hint: str,
        summary: str,
        decisions: list[str] | None = None,
        next_steps: list[str] | None = None,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        workspace_hint = (workspace_hint or "global").strip()
        summary = (summary or "").strip()
        decisions = decisions or []
        next_steps = next_steps or []
        now = _now()

        async with self._driver.session(database=self.database) as session:
            if self._multi_tenant:
                space_id, _ = self._derive_space_and_allowed(context)
                # Create session node linked to workspace and space
                await session.run(
                    """
                    MERGE (w:Workspace {name: $workspace})
                    MERGE (sp:Space {id: $space_id})
                    CREATE (s:Session {
                        workspace_hint: $workspace,
                        summary: $summary,
                        decisions: $decisions,
                        next_steps: $next_steps,
                        created_at: $now,
                        space_id: $space_id
                    })
                    CREATE (s)-[:IN_WORKSPACE]->(w)
                    CREATE (s)-[:IN_SPACE]->(sp)
                    WITH s, w
                    OPTIONAL MATCH (prev:Session)-[:IN_WORKSPACE]->(w)
                    WHERE prev <> s AND prev.space_id = $space_id
                    WITH s, prev
                    ORDER BY prev.created_at DESC
                    LIMIT 1
                    FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
                        CREATE (s)-[:FOLLOWS]->(prev)
                    )
                    """,
                    workspace=workspace_hint,
                    summary=summary,
                    decisions=json.dumps(decisions),
                    next_steps=json.dumps(next_steps),
                    now=now,
                    space_id=space_id,
                )
            else:
                # Legacy single-tenant behavior
                await session.run(
                    """
                    MERGE (w:Workspace {name: $workspace})
                    CREATE (s:Session {
                        workspace_hint: $workspace,
                        summary: $summary,
                        decisions: $decisions,
                        next_steps: $next_steps,
                        created_at: $now
                    })
                    CREATE (s)-[:IN_WORKSPACE]->(w)
                    WITH s, w
                    OPTIONAL MATCH (prev:Session)-[:IN_WORKSPACE]->(w)
                    WHERE prev <> s
                    WITH s, prev
                    ORDER BY prev.created_at DESC
                    LIMIT 1
                    FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
                        CREATE (s)-[:FOLLOWS]->(prev)
                    )
                    """,
                    workspace=workspace_hint,
                    summary=summary,
                    decisions=json.dumps(decisions),
                    next_steps=json.dumps(next_steps),
                    now=now,
                )

            return {"ok": True}

    async def last_session(
        self,
        workspace_hint: str = "global",
        limit: int = 3,
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        workspace_hint = (workspace_hint or "global").strip()
        limit = max(1, min(limit, 10))

        async with self._driver.session(database=self.database) as session:
            if self._multi_tenant:
                _, allowed = self._derive_space_and_allowed(context)
                result = await session.run(
                    """
                    MATCH (s:Session {workspace_hint: $workspace})
                    WHERE s.space_id IN $spaces
                    RETURN
                        elementId(s) AS id,
                        s.created_at AS created_at,
                        s.workspace_hint AS workspace_hint,
                        s.summary AS summary,
                        s.decisions AS decisions,
                        s.next_steps AS next_steps
                    ORDER BY s.created_at DESC
                    LIMIT $limit
                    """,
                    workspace=workspace_hint,
                    limit=limit,
                    spaces=allowed,
                )
            else:
                result = await session.run(
                    """
                    MATCH (s:Session {workspace_hint: $workspace})
                    RETURN
                        elementId(s) AS id,
                        s.created_at AS created_at,
                        s.workspace_hint AS workspace_hint,
                        s.summary AS summary,
                        s.decisions AS decisions,
                        s.next_steps AS next_steps
                    ORDER BY s.created_at DESC
                    LIMIT $limit
                    """,
                    workspace=workspace_hint,
                    limit=limit,
                )
            records = [record.data() async for record in result]
            return [
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "workspace_hint": r["workspace_hint"],
                    "summary": r["summary"],
                    "decisions": (
                        json.loads(r["decisions"])
                        if isinstance(r["decisions"], str)
                        else r["decisions"]
                    ),
                    "next_steps": (
                        json.loads(r["next_steps"])
                        if isinstance(r["next_steps"], str)
                        else r["next_steps"]
                    ),
                }
                for r in records
            ]

    def _derive_space_and_allowed(
        self, context: RequestContext | None
    ) -> tuple[str, list[str]]:
        ctx = context or {}
        user_id = (ctx.get("user_id") or "").strip()
        space_id = (ctx.get("space_id") or "").strip()
        if not space_id:
            space_id = f"personal:{user_id}" if user_id else "global"
        allowed = ctx.get("allowed_spaces")
        if not isinstance(allowed, list) or not allowed:
            allowed = [space_id]
        return space_id, allowed

    # ----------------------------------------------------------------
    # RAG: Vector search
    # ----------------------------------------------------------------

    async def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 8,
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        async with self._driver.session(database=self.database) as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.vector.queryNodes('memory_embedding', $k, $embedding)
                    YIELD node, score
                    OPTIONAL MATCH (node)-[:TAGGED_WITH]->(t:Tag)
                    WITH node, score, collect(t.name) AS tags
                    RETURN
                        elementId(node) AS id,
                        node.kind AS kind,
                        node.title AS title,
                        node.content AS content,
                        node.content_compact AS content_compact,
                        tags,
                        node.pinned AS pinned,
                        node.updated_at AS updated_at,
                        node.importance AS importance,
                        node.workspace_hint AS workspace_hint,
                        score
                    ORDER BY score DESC
                    LIMIT $lim
                    """,
                    k=limit,
                    embedding=query_embedding,
                    lim=limit,
                )
                records = [record.data() async for record in result]
                return records
            except Exception as e:
                logger.warning("Vector search failed: %s", e)
                return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        limit: int = 8,
        prefer: ContentPrefer = "full",
        snippet_chars: int = 400,
        method: str = "hybrid",
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: keyword + vector with reciprocal rank fusion."""
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(limit, 25))

        # Keyword search
        keyword_results = []
        if method in ("keyword", "hybrid"):
            keyword_results = await self.search_memory(
                query, limit=limit * 2, prefer="full",
                snippet_chars=snippet_chars, context=context,
            )

        # Vector search
        vector_results = []
        if method in ("semantic", "hybrid") and query_embedding:
            raw = await self.vector_search(
                query_embedding, limit=limit * 2, context=context,
            )
            vector_results = self._format_search_results(raw, "full", snippet_chars)

        # If only one method, return directly
        if method == "keyword":
            return keyword_results[:limit]
        if method == "semantic":
            return vector_results[:limit]

        # Reciprocal Rank Fusion
        if not vector_results:
            # No embeddings available, fall back to keyword
            return keyword_results[:limit]

        fused = self._reciprocal_rank_fusion(
            [keyword_results, vector_results], k=RRF_K,
        )
        # Format based on preference
        results = []
        for item in fused[:limit]:
            content_full = item.get("content") or ""
            content_compact = item.get("content_compact") or ""
            has_full = bool(content_full)
            if prefer == "compact":
                content = content_compact or _auto_compact(content_full, max_chars=snippet_chars)
            else:
                content = content_full
            results.append({
                "id": item["id"],
                "kind": item.get("kind", "note"),
                "title": item.get("title", ""),
                "content": content,
                "tags": item.get("tags", "[]"),
                "pinned": 1 if item.get("pinned") else 0,
                "updated_at": item.get("updated_at", ""),
                "has_full": has_full,
                "rrf_score": item.get("_rrf_score", 0),
            })
        return results

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict]],
        k: int = 60,
    ) -> list[dict]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i)) for each list where the item appears.
        """
        scored: dict[str, dict] = {}
        for result_list in result_lists:
            for rank, item in enumerate(result_list):
                item_id = item.get("id", "")
                if not item_id:
                    continue
                if item_id not in scored:
                    scored[item_id] = {**item, "_rrf_score": 0.0}
                scored[item_id]["_rrf_score"] += 1.0 / (k + rank)
        return sorted(scored.values(), key=lambda x: x["_rrf_score"], reverse=True)

    # ----------------------------------------------------------------
    # RAG: Graph-augmented retrieval
    # ----------------------------------------------------------------

    async def graph_expand(
        self,
        item_ids: list[str],
        max_hops: int = 1,
        limit: int = 10,
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        if not item_ids:
            return []
        max_hops = max(1, min(max_hops, 3))
        limit = max(1, min(limit, 50))

        async with self._driver.session(database=self.database) as session:
            try:
                # Neo4j doesn't allow parameters in variable-length path patterns,
                # so we cap hops to a safe literal range.
                hop_clause = "1..2" if max_hops >= 2 else "1..1"
                query = f"""
                    UNWIND $ids AS seedId
                    MATCH (seed:MemoryItem) WHERE elementId(seed) = seedId
                    CALL {{
                        WITH seed
                        // Traverse RELATES_TO
                        OPTIONAL MATCH (seed)-[:RELATES_TO*{hop_clause}]-(related:MemoryItem)
                        WHERE related <> seed
                        RETURN related AS neighbor
                        UNION
                        WITH seed
                        // Traverse shared tags
                        OPTIONAL MATCH (seed)-[:TAGGED_WITH]->(t:Tag)<-[:TAGGED_WITH]-(related:MemoryItem)
                        WHERE related <> seed
                        RETURN related AS neighbor
                        UNION
                        WITH seed
                        // Traverse shared sessions
                        OPTIONAL MATCH (seed)-[:DECIDED_IN]->(s:Session)<-[:DECIDED_IN]-(related:MemoryItem)
                        WHERE related <> seed
                        RETURN related AS neighbor
                    }}
                    WITH DISTINCT neighbor
                    WHERE neighbor IS NOT NULL
                      AND NOT elementId(neighbor) IN $ids
                    OPTIONAL MATCH (neighbor)-[:TAGGED_WITH]->(t:Tag)
                    WITH neighbor, collect(t.name) AS tags
                    RETURN
                        elementId(neighbor) AS id,
                        neighbor.kind AS kind,
                        neighbor.title AS title,
                        neighbor.content AS content,
                        neighbor.content_compact AS content_compact,
                        tags,
                        neighbor.pinned AS pinned,
                        neighbor.updated_at AS updated_at,
                        neighbor.importance AS importance,
                        neighbor.workspace_hint AS workspace_hint
                    LIMIT $lim
                """
                result = await session.run(
                    query,
                    ids=item_ids,
                    lim=limit,
                )
                return [record.data() async for record in result]
            except Exception as e:
                logger.warning("Graph expansion failed: %s", e)
                return []

    # ----------------------------------------------------------------
    # RAG: Document ingestion
    # ----------------------------------------------------------------

    async def ingest_document(
        self,
        title: str,
        chunks: list[dict],
        source: str | None = None,
        mime_type: str | None = None,
        workspace_hint: str | None = None,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        title = (title or "").strip()
        if not title or not chunks:
            return {"ok": False, "error": "Title and chunks are required"}

        now = _now()
        workspace_hint = (workspace_hint or "").strip() or None
        source = (source or "").strip() or None
        mime_type = (mime_type or "text/plain").strip()

        async with self._driver.session(database=self.database) as session:
            # Create Document node
            doc_result = await session.run(
                """
                CREATE (d:Document {
                    title: $title,
                    source: $source,
                    mime_type: $mime_type,
                    ingested_at: $now,
                    workspace_hint: $workspace_hint,
                    chunk_count: $chunk_count
                })
                WITH d
                OPTIONAL MATCH (w:Workspace {name: $workspace})
                FOREACH (_ IN CASE WHEN w IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (d)-[:IN_WORKSPACE]->(w)
                )
                RETURN elementId(d) AS id
                """,
                title=title,
                source=source,
                mime_type=mime_type,
                now=now,
                workspace_hint=workspace_hint,
                chunk_count=len(chunks),
                workspace=workspace_hint or "",
            )
            doc_record = await doc_result.single()
            doc_id = doc_record["id"]

            # Create Chunk nodes with embeddings
            for chunk in chunks:
                chunk_content = chunk.get("content", "")
                chunk_embedding = chunk.get("embedding")
                chunk_position = chunk.get("position", 0)
                chunk_tokens = chunk.get("token_count", 0)

                if chunk_embedding:
                    await session.run(
                        """
                        MATCH (d:Document) WHERE elementId(d) = $doc_id
                        CREATE (c:Chunk {
                            content: $content,
                            embedding: $embedding,
                            position: $position,
                            token_count: $token_count,
                            created_at: $now
                        })
                        CREATE (d)-[:HAS_CHUNK]->(c)
                        """,
                        doc_id=doc_id,
                        content=chunk_content,
                        embedding=chunk_embedding,
                        position=chunk_position,
                        token_count=chunk_tokens,
                        now=now,
                    )
                else:
                    await session.run(
                        """
                        MATCH (d:Document) WHERE elementId(d) = $doc_id
                        CREATE (c:Chunk {
                            content: $content,
                            position: $position,
                            token_count: $token_count,
                            created_at: $now
                        })
                        CREATE (d)-[:HAS_CHUNK]->(c)
                        """,
                        doc_id=doc_id,
                        content=chunk_content,
                        position=chunk_position,
                        token_count=chunk_tokens,
                        now=now,
                    )

            return {"ok": True, "document_id": str(doc_id), "chunk_count": len(chunks)}

    async def search_chunks(
        self,
        query_embedding: list[float],
        limit: int = 8,
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        async with self._driver.session(database=self.database) as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
                    YIELD node, score
                    MATCH (d:Document)-[:HAS_CHUNK]->(node)
                    RETURN
                        elementId(node) AS chunk_id,
                        node.content AS content,
                        node.position AS position,
                        node.token_count AS token_count,
                        score,
                        elementId(d) AS document_id,
                        d.title AS document_title,
                        d.source AS document_source,
                        d.workspace_hint AS workspace_hint
                    ORDER BY score DESC
                    LIMIT $lim
                    """,
                    k=limit,
                    embedding=query_embedding,
                    lim=limit,
                )
                return [record.data() async for record in result]
            except Exception as e:
                logger.warning("Chunk vector search failed: %s", e)
                return []

    # ----------------------------------------------------------------
    # RAG: Backfill
    # ----------------------------------------------------------------

    async def get_items_without_embeddings(
        self,
        limit: int = 100,
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (m:MemoryItem)
                WHERE m.embedding IS NULL
                RETURN
                    elementId(m) AS id,
                    m.title AS title,
                    m.content_compact AS content_compact,
                    m.content AS content
                ORDER BY m.updated_at DESC
                LIMIT $lim
                """,
                lim=limit,
            )
            return [record.data() async for record in result]

    async def set_embedding(
        self,
        item_id: str,
        embedding: list[float],
    ) -> bool:
        try:
            async with self._driver.session(database=self.database) as session:
                await session.run(
                    """
                    MATCH (m:MemoryItem)
                    WHERE elementId(m) = $item_id
                    SET m.embedding = $embedding
                    """,
                    item_id=item_id,
                    embedding=embedding,
                )
                return True
        except Exception as e:
            logger.warning("Failed to set embedding for %s: %s", item_id, e)
            return False

    # ----------------------------------------------------------------
    # Knowledge index (compressed structural map)
    # ----------------------------------------------------------------

    async def generate_knowledge_index(
        self,
        workspace_hint: str = "global",
        max_tokens: int = 800,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        workspace_hint = (workspace_hint or "global").strip()
        max_chars = max_tokens * 4

        async with self._driver.session(database=self.database) as session:
            spaces: list[str] | None = None
            if self._multi_tenant:
                _, allowed = self._derive_space_and_allowed(context)
                spaces = allowed

            # 1. Counts per kind
            if self._multi_tenant:
                kind_result = await session.run(
                    """
                    MATCH (m:MemoryItem)
                    WHERE m.space_id IN $spaces
                    RETURN m.kind AS kind, count(m) AS cnt
                    ORDER BY cnt DESC
                    """,
                    spaces=spaces,
                )
            else:
                kind_result = await session.run(
                    "MATCH (m:MemoryItem) RETURN m.kind AS kind, count(m) AS cnt ORDER BY cnt DESC"
                )
            kind_counts = {r["kind"]: r["cnt"] async for r in kind_result}

            # 2. Top tags with counts
            if self._multi_tenant:
                tag_result = await session.run(
                    """
                    MATCH (m:MemoryItem)-[:TAGGED_WITH]->(t:Tag)
                    WHERE m.space_id IN $spaces
                    RETURN t.name AS tag, count(m) AS cnt
                    ORDER BY cnt DESC
                    LIMIT 30
                    """,
                    spaces=spaces,
                )
            else:
                tag_result = await session.run(
                    """
                    MATCH (m:MemoryItem)-[:TAGGED_WITH]->(t:Tag)
                    RETURN t.name AS tag, count(m) AS cnt
                    ORDER BY cnt DESC
                    LIMIT 30
                    """
                )
            tag_counts = [(r["tag"], r["cnt"]) async for r in tag_result]

            # 3. Workspaces with item counts
            if self._multi_tenant:
                ws_result = await session.run(
                    """
                    MATCH (m:MemoryItem)
                    WHERE m.space_id IN $spaces AND m.workspace_hint IS NOT NULL
                    RETURN m.workspace_hint AS ws, count(m) AS cnt
                    ORDER BY cnt DESC
                    LIMIT 20
                    """,
                    spaces=spaces,
                )
            else:
                ws_result = await session.run(
                    """
                    MATCH (m:MemoryItem)
                    WHERE m.workspace_hint IS NOT NULL
                    RETURN m.workspace_hint AS ws, count(m) AS cnt
                    ORDER BY cnt DESC
                    LIMIT 20
                    """
                )
            workspace_counts = [(r["ws"], r["cnt"]) async for r in ws_result]

            # 4. Pinned item titles (always important)
            if self._multi_tenant:
                pinned_result = await session.run(
                    """
                    MATCH (m:MemoryItem {pinned: true})
                    WHERE m.space_id IN $spaces
                    RETURN m.kind AS kind, m.title AS title
                    ORDER BY m.updated_at DESC
                    LIMIT 15
                    """,
                    spaces=spaces,
                )
            else:
                pinned_result = await session.run(
                    """
                    MATCH (m:MemoryItem {pinned: true})
                    RETURN m.kind AS kind, m.title AS title
                    ORDER BY m.updated_at DESC
                    LIMIT 15
                    """
                )
            pinned_titles = [(r["kind"], r["title"]) async for r in pinned_result]

            # 5. Recent high-value items (decisions + patterns)
            if self._multi_tenant:
                recent_result = await session.run(
                    """
                    MATCH (m:MemoryItem)
                    WHERE m.space_id IN $spaces AND m.kind IN ['decision', 'pattern']
                    RETURN m.kind AS kind, m.title AS title, m.workspace_hint AS ws
                    ORDER BY m.updated_at DESC
                    LIMIT 15
                    """,
                    spaces=spaces,
                )
            else:
                recent_result = await session.run(
                    """
                    MATCH (m:MemoryItem)
                    WHERE m.kind IN ['decision', 'pattern']
                    RETURN m.kind AS kind, m.title AS title, m.workspace_hint AS ws
                    ORDER BY m.updated_at DESC
                    LIMIT 15
                    """
                )
            recent_hv = [(r["kind"], r["title"], r.get("ws") or "") async for r in recent_result]

            # 6. Document count
            if self._multi_tenant:
                doc_result = await session.run(
                    """
                    MATCH (d:Document)
                    WHERE d.workspace_hint IS NULL OR d.workspace_hint IN
                          [x IN $spaces | x]
                    RETURN count(d) AS cnt
                    """,
                    spaces=spaces,
                )
            else:
                doc_result = await session.run(
                    "MATCH (d:Document) RETURN count(d) AS cnt"
                )
            doc_record = await doc_result.single()
            doc_count = doc_record["cnt"] if doc_record else 0

        # --- Build compressed index markdown ---
        total_items = sum(kind_counts.values())
        lines: list[str] = []
        lines.append("# Mnemosyne Knowledge Index")
        lines.append(f"Total: {total_items} memories, {doc_count} documents")
        lines.append("")

        # Kinds overview
        if kind_counts:
            lines.append("## By Type")
            for kind, cnt in kind_counts.items():
                lines.append(f"- **{kind}**: {cnt}")
            lines.append("")

        # Pinned items
        if pinned_titles:
            lines.append("## Pinned (Always Relevant)")
            for kind, title in pinned_titles:
                lines.append(f"- [{kind}] {title}")
            lines.append("")

        # Recent decisions & patterns
        if recent_hv:
            lines.append("## Recent Decisions & Patterns")
            for kind, title, ws in recent_hv:
                ws_tag = f" ({ws})" if ws else ""
                lines.append(f"- [{kind}] {title}{ws_tag}")
            lines.append("")

        # Tag clusters
        if tag_counts:
            lines.append("## Topics (by tag)")
            for tag, cnt in tag_counts:
                lines.append(f"- **{tag}** ({cnt})")
            lines.append("")

        # Workspaces
        if workspace_counts:
            lines.append("## Workspaces")
            for ws, cnt in workspace_counts:
                lines.append(f"- **{ws}**: {cnt} items")
            lines.append("")

        index_text = "\n".join(lines)

        # Truncate if over budget
        if len(index_text) > max_chars:
            index_text = index_text[:max_chars].rsplit("\n", 1)[0] + "\n…"

        return {
            "index": index_text,
            "token_estimate": _estimate_tokens(index_text),
            "stats": {
                "total_items": total_items,
                "total_documents": doc_count,
                "kinds": kind_counts,
                "tag_count": len(tag_counts),
                "workspace_count": len(workspace_counts),
                "pinned_count": len(pinned_titles),
            },
        }

    # ----------------------------------------------------------------
    # Auto-context (pre-message injection)
    # ----------------------------------------------------------------

    async def auto_context(
        self,
        message: str,
        limit: int = 5,
        min_score: float = 0.3,
        context: RequestContext | None = None,
    ) -> list[dict[str, Any]]:
        message = (message or "").strip()
        if not message or len(message) < 3:
            return []

        limit = max(1, min(limit, 10))

        # Try vector search first (best quality), fall back to keyword
        results = []
        try:
            from embedding import get_embedding
            embedding = await get_embedding(message[:500])
            if embedding:
                raw = await self.vector_search(embedding, limit=limit * 2, context=context)
                for item in raw:
                    score = item.get("score", 0)
                    if score >= min_score:
                        results.append({
                            "text": _auto_compact(
                                (item.get("title", "") + ": " + (item.get("content_compact") or item.get("content") or "")).strip(),
                                max_chars=300,
                            ),
                            "score": round(score, 3),
                            "id": item.get("id", ""),
                            "kind": item.get("kind", "note"),
                        })
                return results[:limit]
        except Exception as e:
            logger.debug("Auto-context vector search unavailable: %s", e)

        # Fallback: keyword search
        keyword_results = await self.search_memory(
            message[:200], limit=limit, prefer="compact",
            snippet_chars=200, context=context,
        )
        for item in keyword_results:
            results.append({
                "text": _auto_compact(
                    (item.get("title", "") + ": " + (item.get("content") or "")).strip(),
                    max_chars=300,
                ),
                "score": 1.0,  # keyword search doesn't provide similarity scores
                "id": item.get("id", ""),
                "kind": item.get("kind", "note"),
            })
        return results[:limit]
