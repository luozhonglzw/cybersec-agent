"""向量存储后端(默认后端之外的真实实现)。

分层意图
--------

`app/rag/**` 是**离线检索底座**:纯数据契约、确定性切块、离线嵌入基线、
内存存储、完整性校验与确定性 top-k。它有一条冻结的结构断言 —— 不 import
任何网络模块、只有 `corpus.py` 碰文件系统、不 import 授权层。

本包放**依赖外部存储/SDK**的实现。方向是单向的:本包 import `app.rag`,
`app.rag` 不认识本包。于是"检索底座不会自己发请求"这条证明不需要因为
引入 `qdrant-client` 而被削弱。

当前只有 Qdrant Local(`qdrant_local`)。它是**单进程、本地目录、排他锁**
的嵌入式模式,**不是** Qdrant 服务端,也不提供任何分布式或高可用保证。
"""
from app.vectorstore.qdrant_local import (
    COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    INDEX_FORMAT,
    POINT_ID_NAMESPACE,
    QdrantLocalVectorStore,
    build_qdrant_index,
    build_qdrant_retriever,
    point_id_of,
)

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_INDEX_DIR",
    "INDEX_FORMAT",
    "POINT_ID_NAMESPACE",
    "QdrantLocalVectorStore",
    "build_qdrant_index",
    "build_qdrant_retriever",
    "point_id_of",
]
