"""
語意去重模組（Semantic Deduplication）
功能：找出資料庫中語意相似的文章，標記為重複，只保留最新和最舊的

執行頻率：每 14 天執行一次（每兩週）
處理範圍：最近 14 天內的 RSS 新聞文章
"""

import os
import logging
import numpy as np
import psycopg2
import sys
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

load_dotenv()

logger = logging.getLogger(__name__)

# 設定區
DEDUP_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.15"))
DAYS_TO_CHECK   = int(os.getenv("DAYS_TO_CHECK_DUPLICATE", "7"))   

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION     = os.getenv("GCP_REGION", "us-central1")

MODEL_NAME = "gemini-embedding-001"
CHUNK_SIZE = 500

# 1. 資料庫相關操作
def get_db_connection():
    """建立資料庫連線"""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "ai_knowledge"),
        user=os.getenv("POSTGRES_USER", "crawler_user"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def fetch_recent_news_articles(conn) -> list:
    """
    從資料庫取出最近 N 天的「新聞類」文章。

    條件：
      - content_type = 'rss'（只處理新聞，不處理 arXiv）
      - governance_status = 'pending'
      - fetched_at 在最近 DAYS_TO_CHECK 天內
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                a.id,
                a.title,
                a.content,
                a.published_at,
                s.name AS source_name,
                s.source_tier
            FROM articles a
            JOIN sources s ON a.source_id = s.id
            WHERE
                a.content_type = 'rss'
                AND a.governance_status = 'pending'
                AND a.fetched_at >= NOW() - INTERVAL '%s days'
            ORDER BY a.published_at DESC
        """, (DAYS_TO_CHECK,))

        rows = cur.fetchall()
        columns = ["id", "title", "content", "published_at",
                   "source_name", "source_tier"]
        return [dict(zip(columns, row)) for row in rows]


def mark_as_duplicate(conn, article_ids: list):
    """
    把被標記為重複的文章，更新 governance_status 為 'duplicate'。
    原始資料不刪除，只標記狀態，保留完整記錄。
    """
    if not article_ids:
        return

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE articles
            SET governance_status = 'duplicate'
            WHERE id = ANY(%s)
        """, (article_ids,))
        conn.commit()

    logger.info(f"   標記 {len(article_ids)} 篇為重複")


# 2. Chunking
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """把文字切成固定大小的 chunk"""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_first_chunk(article: dict) -> str:
    """取得文章的 first chunk，用來做 embedding"""
    title   = article.get("title", "") or ""
    content = article.get("content", "") or ""

    full_text = f"{title}\n{content}".strip()
    chunks = chunk_text(full_text)

    if not chunks:
        return title

    return chunks[0]


# 3. Vertex AI Embedding
def init_vertex_ai():
    """初始化 Vertex AI SDK"""
    if not GCP_PROJECT_ID:
        raise ValueError("缺少 GCP_PROJECT_ID，請在 .env 設定")

    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    logger.info(f"   Vertex AI 初始化完成（project={GCP_PROJECT_ID}, region={GCP_REGION}）")


def embed_articles(articles: list) -> np.ndarray:
    """
    用 Vertex AI gemini-embedding-001 把所有文章轉成向量矩陣。

    輸入：N 篇文章
    輸出：N × 3072 的矩陣
    """
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)

    texts = [get_first_chunk(a) for a in articles]
    logger.info(f"   對 {len(texts)} 篇文章做向量化（model={MODEL_NAME}）...")

    BATCH_SIZE = 250
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = [
            TextEmbeddingInput(text=t, task_type="SEMANTIC_SIMILARITY")
            for t in batch
        ]
        response = model.get_embeddings(inputs)
        for embedding in response:
            all_embeddings.append(embedding.values)

        logger.info(f"批次 {i // BATCH_SIZE + 1}：完成 {min(i + BATCH_SIZE, len(texts))}/{len(texts)} 篇")

    embeddings = np.array(all_embeddings)
    logger.info(f"向量化完成，維度：{embeddings.shape}")
    return embeddings


# 4. Clustering 聚類分析
def cluster_articles(embeddings: np.ndarray) -> np.ndarray:
    """
    用 Average Linkage + Fcluster 把相似的文章分成群。

    回傳：每篇文章的 cluster 編號陣列
    """
    similarity_matrix = cosine_similarity(embeddings)
    distance_matrix   = 1 - similarity_matrix
    distance_matrix   = np.clip(distance_matrix, 0, 1)
    np.fill_diagonal(distance_matrix, 0)

    condensed = squareform(distance_matrix)
    Z         = linkage(condensed, method='average')
    labels    = fcluster(Z, t=DEDUP_THRESHOLD, criterion='distance')

    return labels


# 5. 決定哪些文章要保留
def find_duplicates(articles: list, cluster_labels: np.ndarray) -> list:
    """
    根據 cluster 結果，找出要標記為重複的文章 ID。

    保留策略：每個 cluster 保留最新和最舊的文章，其他標記為 duplicate。
    若群組只有 2 篇則全保留。
    """
    cluster_groups = {}
    for article, label in zip(articles, cluster_labels):
        label = int(label)
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(article)

    duplicate_ids = []

    for label, group in cluster_groups.items():
        if len(group) <= 2:   # 2 篇以下全保留
            continue

        logger.info(f"   發現相似群組 {label}：{len(group)} 篇")
        for a in group:
            logger.info(f"     - [{a['source_name']}] {a['title'][:40]}...")

        sorted_group = sorted(
            group,
            key=lambda x: x["published_at"] if x["published_at"] else 0
        )

        keep_oldest = sorted_group[0]["id"]
        keep_newest = sorted_group[-1]["id"]

        for article in sorted_group[1:-1]:
            duplicate_ids.append(article["id"])

        logger.info(f"     保留：ID {keep_oldest}（最舊）和 ID {keep_newest}（最新）")
        logger.info(f"     標記重複：{len(sorted_group) - 2} 篇")

    return duplicate_ids


# 6. 主流程
def run_deduplication():
    """
    執行完整的語意去重流程。

    流程：
      1. 初始化 Vertex AI
      2. 從資料庫取出最近 14 天的新聞文章
      3. 用 gemini-embedding-001 做 embedding（first chunk）
      4. Average Linkage clustering
      5. 找出重複的文章
      6. 更新資料庫狀態
    """
    logger.info("=" * 50)
    logger.info(f"開始語意去重（處理範圍：最近 {DAYS_TO_CHECK} 天）")
    logger.info("=" * 50)

    conn = None
    try:
        init_vertex_ai()
        conn = get_db_connection()

        articles = fetch_recent_news_articles(conn)
        logger.info(f"取出 {len(articles)} 篇待去重文章（最近 {DAYS_TO_CHECK} 天）")

        if len(articles) < 2:
            logger.info("   文章數不足，跳過去重")
            return

        embeddings = embed_articles(articles)

        logger.info(f"   開始聚類分析（threshold={DEDUP_THRESHOLD}）")
        cluster_labels = cluster_articles(embeddings)

        unique_clusters = len(set(cluster_labels))
        logger.info(f"   共分成 {unique_clusters} 個群組")

        duplicate_ids = find_duplicates(articles, cluster_labels)
        logger.info(f"   發現 {len(duplicate_ids)} 篇重複文章")

        if duplicate_ids:
            mark_as_duplicate(conn, duplicate_ids)

        logger.info("=" * 50)
        logger.info("去重完成")
        logger.info(f"   處理文章：{len(articles)} 篇")
        logger.info(f"   標記重複：{len(duplicate_ids)} 篇")
        logger.info(f"   保留文章：{len(articles) - len(duplicate_ids)} 篇")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"去重失敗：{e}")
        raise

    finally:
        if conn:
            conn.close()


# 7. 直接執行（測試用）
if __name__ == "__main__":
    import schedule
    import time

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/dedup.log", encoding="utf-8")
        ]
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "schedule"

    if mode == "now":
        run_deduplication()
    else:
        # ✅ 改為每兩週執行一次
        # schedule 套件沒有內建 every 2 weeks，用 every(14).days 實作
        logger.info("排程模式啟動，每 14 天執行去重")
        schedule.every(14).days.do(run_deduplication)

        while True:
            schedule.run_pending()
            time.sleep(60)