# -*- coding: utf-8 -*-
"""
阶段 3：

用户 Query -> Embedding -> Chroma 检索

支持：
- 指定 domain
- 指定 subdomain
- 不指定 domain 时跨所有 rag_* collection 检索并统一排序
"""

from __future__ import annotations

import argparse
import unicodedata
from typing import List, Dict, Any

import chromadb
import torch

from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    QUERY_INSTRUCTION,
    COLLECTION_PREFIX,
)


# ============================================================
# 1. Embedding 模型
# ============================================================

def load_embedding_model() -> SentenceTransformer:
    """
    自动选择设备：
    - 有 CUDA -> GPU
    - 无 CUDA -> CPU
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )


# ============================================================
# 2. Chroma 客户端
# ============================================================

def get_client():
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# ============================================================
# 3. 获取所有 RAG Collection
# ============================================================

def list_rag_collections(client) -> List[str]:
    collections = client.list_collections()

    names = []

    for item in collections:
        name = item if isinstance(item, str) else item.name

        if name.startswith(COLLECTION_PREFIX):
            names.append(name)

    return sorted(names)


# ============================================================
# 4. Metadata Filter
# ============================================================

def build_where(subdomain: str | None):
    """构造 Chroma metadata filter。当前支持 subdomain 过滤。"""
    if subdomain:
        return {"subdomain": subdomain}
    return None


# ============================================================
# 5. 查询单个 Collection
# ============================================================

def query_collection(
    collection,
    query_embedding,
    top_k: int,
    where,
) -> List[Dict[str, Any]]:

    kwargs = {
        "query_embeddings": [
            query_embedding.tolist()
        ],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:
        kwargs["where"] = where

    result = collection.query(
        **kwargs
    )

    documents = result.get(
        "documents",
        [[]]
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]]
    )[0]

    distances = result.get(
        "distances",
        [[]]
    )[0]

    items = []

    for doc, meta, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        items.append(
            {
                "document": doc,
                "metadata": meta,

                "distance": float(
                    distance
                ),

                "similarity": (
                    1.0 - float(distance)
                ),
            }
        )

    return items


# ============================================================
# 6. 终端中文宽度处理
# ============================================================

def display_width(text: str) -> int:
    """
    计算字符串在终端中的实际显示宽度。

    中文通常占 2 个字符宽度，
    英文和数字通常占 1 个字符宽度。
    """

    width = 0

    for ch in str(text):
        if unicodedata.east_asian_width(
            ch
        ) in ("W", "F", "A"):
            width += 2
        else:
            width += 1

    return width


def pad(
    text,
    width: int,
) -> str:
    """
    按终端实际显示宽度补齐空格。
    """

    text = "" if text is None else str(text)

    current_width = display_width(
        text
    )

    if current_width >= width:
        return text

    return (
        text
        + " " * (
            width - current_width
        )
    )


# ============================================================
# 7. RAG 检索
# ============================================================

def run(
    query: str,
    top_k: int,
    domain: str | None,
    subdomain: str | None,
):

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    client = get_client()

    model = load_embedding_model()

    # BGE Query 侧添加检索指令
    query_text = (
        QUERY_INSTRUCTION
        + query.strip()
    )

    query_embedding = model.encode(
        [query_text],

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    )[0]


    # --------------------------------------------------------
    # 确定需要检索哪些 Collection
    # --------------------------------------------------------

    if domain:
        collection_names = [
            COLLECTION_PREFIX
            + domain.lower()
        ]

    else:
        collection_names = (
            list_rag_collections(
                client
            )
        )


    if not collection_names:
        raise RuntimeError(
            "没有找到向量库 collection，"
            "请先运行 python ingest.py\n"
            f"Chroma path: {CHROMA_DIR}"
        )


    # --------------------------------------------------------
    # Metadata Filter
    # --------------------------------------------------------

    where = build_where(subdomain)


    # --------------------------------------------------------
    # 跨 Collection 检索
    # --------------------------------------------------------

    # 每个 Collection 多取一些结果，
    # 最后再跨 Collection 统一排序。

    per_collection_k = max(
        top_k,
        top_k * 2,
    )

    all_results: List[
        Dict[str, Any]
    ] = []


    for collection_name in collection_names:

        try:
            collection = (
                client.get_collection(
                    name=collection_name,
                    embedding_function=None,
                )
            )

        except Exception:
            continue


        items = query_collection(
            collection=collection,

            query_embedding=(
                query_embedding
            ),

            top_k=(
                per_collection_k
            ),

            where=where,
        )


        for item in items:
            item[
                "collection"
            ] = collection_name


        all_results.extend(
            items
        )


    # --------------------------------------------------------
    # 全局排序
    # --------------------------------------------------------

    # cosine distance 越小，
    # 表示语义越相似。

    all_results.sort(
        key=lambda x: x["distance"]
    )

    final = all_results[:top_k]


    # --------------------------------------------------------
    # 输出 Query
    # --------------------------------------------------------

    print(
        f"\nQuery: {query}"
    )

    print(
        "=" * 100
    )


    if not final:
        print(
            "没有检索到结果。"
        )
        return


    # ========================================================
    # 终端两列展示
    # ========================================================

    LEFT_WIDTH = 42


    for i, item in enumerate(
        final,
        start=1,
    ):

        meta = item["metadata"]


        print(
            f"\n[{i}] "
            f"similarity="
            f"{item['similarity']:.4f}"
        )


        # ----------------------------------------------------
        # 第一行：
        # Collection | Domain
        # ----------------------------------------------------

        left = (
            f"collection="
            f"{item['collection']}"
        )

        right = (
            f"domain="
            f"{meta.get('domain', '')}"
        )

        print(
            f"{pad(left, LEFT_WIDTH)}"
            f" | "
            f"{right}"
        )


        # ----------------------------------------------------
        # 第二行：
        # Subdomain | Document Type
        # ----------------------------------------------------

        left = (
            f"subdomain="
            f"{meta.get('subdomain', '')}"
        )

        right = (
            f"document_type="
            f"{meta.get('document_type', '')}"
        )

        print(
            f"{pad(left, LEFT_WIDTH)}"
            f" | "
            f"{right}"
        )


        # ----------------------------------------------------
        # 第三行：
        # Source | Content Type
        # ----------------------------------------------------

        left = (
            f"source="
            f"{meta.get('source_file', '')}"
        )

        right = (
            f"content_type="
            f"{meta.get('content_type', '')}"
        )

        print(
            f"{pad(left, LEFT_WIDTH)}"
            f" | "
            f"{right}"
        )


        # ----------------------------------------------------
        # 第四行：
        # Section
        # ----------------------------------------------------

        left = "section"

        right = meta.get(
            "section_path",
            "",
        )

        print(
            f"{pad(left, LEFT_WIDTH)}"
            f" | "
            f"{right}"
        )


        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if meta.get(
            "page"
        ) is not None:

            left = "page"

            right = str(
                meta.get(
                    "page"
                )
            )

            print(
                f"{pad(left, LEFT_WIDTH)}"
                f" | "
                f"{right}"
            )


        # ----------------------------------------------------
        # Excel / CSV
        # ----------------------------------------------------

        if meta.get(
            "sheet_name"
        ):

            left = (
                f"sheet="
                f"{meta.get('sheet_name')}"
            )

            right = (
                f"rows="
                f"{meta.get('row_start')}"
                f"-"
                f"{meta.get('row_end')}"
            )

            print(
                f"{pad(left, LEFT_WIDTH)}"
                f" | "
                f"{right}"
            )


        # ----------------------------------------------------
        # PowerPoint
        # ----------------------------------------------------

        if meta.get(
            "slide_number"
        ) is not None:

            left = "slide"

            right = str(
                meta.get(
                    "slide_number"
                )
            )

            print(
                f"{pad(left, LEFT_WIDTH)}"
                f" | "
                f"{right}"
            )


        # ----------------------------------------------------
        # Audio / Video
        # ----------------------------------------------------

        if meta.get(
            "media_start_sec"
        ) is not None:

            left = "time"

            right = (
                f"{meta.get('media_start_sec')}"
                f"-"
                f"{meta.get('media_end_sec')}"
                f"s"
            )

            print(
                f"{pad(left, LEFT_WIDTH)}"
                f" | "
                f"{right}"
            )


        # ----------------------------------------------------
        # 正文
        # ----------------------------------------------------

        print(
            "-" * 100
        )

        print(
            item[
                "document"
            ][:1600]
        )


# ============================================================
# 8. CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="RAG 向量检索"
    )


    parser.add_argument(
        "query",
        type=str,
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--domain",
        type=str,
        default=None,
    )


    parser.add_argument(
        "--subdomain",
        type=str,
        default=None,
    )


    args = parser.parse_args()


    run(
        query=args.query,

        top_k=args.top_k,

        domain=args.domain,

        subdomain=args.subdomain,
    )


if __name__ == "__main__":
    main()