# CyberBrain Dependencies

**English** | [中文](QUICKSTART.md)

CyberBrain depends on three external database services and a set of local SQLite files at runtime. All connection settings live in `.env` (setting names match `app/config/settings.py` one to one; see `.env.example`).

| Database | Required | Purpose | Settings | Default address |
|---|---|---|---|---|
| **Qdrant** | Yes | Vector storage for long-term memory: the Brief, Episodic and Knowledge collections | `QDRANT_URL` | `http://localhost:6333` |
| **Redis** | Yes | Persona, subject context, relationships, working memory (PFC) | `REDIS_URL` | `redis://localhost:6379/0` |
| **MongoDB** | Optional | Agent life-event timeline (what was heard and said, thoughts, to-dos, mood changes, etc.) | `MONGODB_URI`, `MONGODB_DB_NAME`, `MONGODB_COLLECTION_LIFE_EVENT` | — |
| **SQLite** (local files) | Created automatically | mem0's memory change history | `MEM0_BRIEF_HISTORY_DB`, `MEM0_EPISODIC_HISTORY_DB` | `./data/*.db` |

## Qdrant

The service creates the following collections on startup; no manual setup is needed:

| Collection | Contents | How it is written |
|---|---|---|
| `brief_memories` | Brief memories (stable facts and beliefs about "me") | mem0 writes the dense vector, then a `bm25` sparse vector is added |
| `episodic_memories` | Episodic memories (experiences, events) | Same as above |
| `knowledge_memories` | Knowledge memories (expertise) | Written to Qdrant directly, dense vector only |

All three collections use `QDRANT_DIMENSION` dimensions (1024 by default) and cosine distance. The two collections created by mem0 come with a sparse vector slot named `bm25`; the service checks it on startup and enables IDF on it.

Notes:

- **Qdrant must support server-side BM25 inference.** Brief retrieval generates sparse vectors on the server with the `qdrant/bm25` model (Chinese text needs the `multilingual` tokenizer). Development used Qdrant 1.19.2.
- **The vector dimension must match the embedding model**: `QDRANT_DIMENSION` and `EMBEDDING_DIMENSION` both default to 1024. Once a collection is created with a given dimension, changing the dimension or switching embedding models makes existing data unsearchable; the collection has to be cleared and rebuilt.
- The service adds `localhost` to `NO_PROXY` so that a local HTTP proxy does not intercept connections to Qdrant.

## Redis

No keys need to be created in advance; all data is written per agent at runtime. The main keys are (full details in [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md#storage-layout)):

| Key | Type | Contents |
|---|---|---|
| `persona:{agent_id}:system1` / `:system2` / `:speaking_manner` | String | The three parts of the persona |
| `subject_context:{agent_id}` | String (JSON, expires after 30 days) | Situation, overall state, thoughts, to-dos |
| `agent:{agent_id}:relationships` | Hash | Relationships |
| `wm:{agent_id}:stm` / `wm:{agent_id}:stm:compress` | List | Working memory raw records / compressed summaries (the braces around `{agent_id}` are literal; they act as a Redis Cluster hash tag) |

## MongoDB

- Stores life events. The database defaults to `cyberbrain` and the collection to `agent_life_event`. Indexes are created automatically on startup.
- **The service still runs if MongoDB is unreachable**; it just stops recording life events, and `GET /admin/timeline/{agent_id}` returns nothing.

## External services besides the databases

The service also calls the following model APIs, configured in `.env` as well:

- **Volcano Engine Ark**: `ARK_BASE_URL`, `ARK_API_KEY`, `ARK_SYSTEM_ONE_URL`, and three model names: `DOUBAO_MODEL_PRO` / `DOUBAO_MODEL_LITE` / `DOUBAO_MODEL_CHARACTER`
- **Alibaba Cloud Model Studio (DashScope)**: `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`. The same key is also used for the `qwen3-rerank` reranking model

A full setup guide is not provided yet. See [README.en.md](README.en.md) for the design and [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md) for implementation details.
