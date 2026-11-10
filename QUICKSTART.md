# CyberBrain 依赖的数据库

[English](QUICKSTART.en.md) | **中文**

CyberBrain 运行时依赖三个外部数据库服务和一组本地 SQLite 文件。连接地址都在 `.env` 中配置（配置项名与 `app/config/settings.py` 一一对应，可参考 `.env.example`）。

| 数据库 | 是否必需 | 用途 | 配置项 | 默认地址 |
|---|---|---|---|---|
| **Qdrant** | 必需 | 长期记忆的向量存储：Brief、Episodic、Knowledge 三个集合 | `QDRANT_URL` | `http://localhost:6333` |
| **Redis** | 必需 | 人格画像、主体性上下文、人际关系、前额叶工作记忆（PFC） | `REDIS_URL` | `redis://localhost:6379/0` |
| **MongoDB** | 可选 | Agent 生命事件时间线（听到/说过的话、念头、待办、情绪变化等） | `MONGODB_URI`、`MONGODB_DB_NAME`、`MONGODB_COLLECTION_LIFE_EVENT` | — |
| **SQLite**（本地文件） | 自动创建 | mem0 的记忆变更历史 | `MEM0_BRIEF_HISTORY_DB`、`MEM0_EPISODIC_HISTORY_DB` | `./data/*.db` |

## Qdrant

服务启动时会自动创建以下集合，无需手动建表：

| 集合 | 内容 | 写入方式 |
|---|---|---|
| `brief_memories` | Brief 记忆（关于"我"的稳定事实、信条） | mem0 写入稠密向量，再补写 `bm25` 稀疏向量 |
| `episodic_memories` | Episodic 记忆（经历、事件） | 同上 |
| `knowledge_memories` | Knowledge 记忆（专业知识） | 直接写 Qdrant，只有稠密向量 |

三个集合的向量维度都是 `QDRANT_DIMENSION`（默认 1024），距离都是余弦。mem0 建的两个集合自带名为 `bm25` 的稀疏向量槽位，服务启动时会检查并给它开启 IDF。

注意事项：

- **需要支持服务端 BM25 推理的 Qdrant 版本**。Brief 检索用 `qdrant/bm25` 模型在服务端生成稀疏向量（中文需要 `multilingual` 分词器）。开发环境使用的是 Qdrant 1.19.2。
- **向量维度要和 embedding 模型一致**：`QDRANT_DIMENSION` 与 `EMBEDDING_DIMENSION` 默认都是 1024。集合一旦按某个维度建好，再改维度或换 embedding 模型，已有数据就无法正常检索，需要清空重建。
- 服务会把 `localhost` 加入 `NO_PROXY`，避免本机的 HTTP 代理拦截到 Qdrant 的连接。

## Redis

不需要预先建任何 key，所有数据都在运行中按 agent 写入。主要 key 如下（完整说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#存储布局)）：

| Key | 类型 | 内容 |
|---|---|---|
| `persona:{agent_id}:system1` / `:system2` / `:speaking_manner` | String | 人格画像三段 |
| `subject_context:{agent_id}` | String（JSON，30 天过期） | 处境、整体状态、念头、待办 |
| `agent:{agent_id}:relationships` | Hash | 人际关系 |
| `wm:{agent_id}:stm` / `wm:{agent_id}:stm:compress` | List | 工作记忆原文 / 压缩摘要（`{agent_id}` 两侧的花括号是字面量，用作 Redis Cluster 的 hash tag） |

## MongoDB

- 用于记录生命事件，库名默认 `cyberbrain`，集合名默认 `agent_life_event`。启动时会自动创建索引。
- **连不上时服务仍能正常运行**，只是不再记录生命事件，`GET /admin/timeline/{agent_id}` 也查不到数据。

## 除数据库外还需要的外部服务

数据库之外，服务还会调用以下模型接口，同样在 `.env` 中配置：

- **火山引擎方舟（Ark）**：`ARK_BASE_URL`、`ARK_API_KEY`、`ARK_SYSTEM_ONE_URL` 和三个模型名 `DOUBAO_MODEL_PRO` / `DOUBAO_MODEL_LITE` / `DOUBAO_MODEL_CHARACTER`
- **阿里云百炼（DashScope）**：`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`。同一个 key 也用于 `qwen3-rerank` 重排模型

完整的启动流程暂不提供。项目设计见 [README.md](README.md)，实现细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
