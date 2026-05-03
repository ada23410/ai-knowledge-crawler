"""
語意去重模組（Semantic Deduplication）
=====================================================
功能：找出資料庫中語意相似的文章，標記為重複，只保留最新和最舊的

原理說明（給前端工程師）：
  想像每篇文章是一個座標點
  語意相近的文章座標會靠在一起
  我們把靠在一起的點歸成一群（cluster）
  每群只保留最新和最舊的，其他的標記為重複

參考：Bob 的 Duplicate Articles Detection 設計
  embedding model: EmbeddingGemma-300m → 我們換成 sentence-transformers（合規）
  clustering: Average Linkage + Fcluster（threshold=0.15）
"""

import os
import logging
import numpy as np
import psycopg2
from dotenv import load_dotenv

# sklearn = scikit-learn，機器學習套件
# cosine_similarity：計算兩個向量的相似度（0~1，越接近1越相似）
from sklearn.metrics.pairwise import cosine_similarity

# scipy：科學計算套件
# linkage：階層式聚類演算法（把相似的點合併成樹狀結構）
# fcluster：根據 threshold 把樹狀結構切成一個個 cluster
from scipy.cluster.hierarchy import linkage, fcluster

# sentence-transformers：本地 embedding 模型
# 把一段文字轉成一個向量（數字陣列）
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# 設定區（對應 .env 的可調參數）
# ══════════════════════════════════════════════

# 去重 threshold：越小越嚴格，越大越容易誤判
# Bob 建議 0.15，我們先用這個值
DEDUP_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.15"))

# 幾天內的文章才需要做去重
# 不需要把所有歷史文章都比對，只比對最近的
DAYS_TO_CHECK = int(os.getenv("DAYS_TO_CHECK_DUPLICATE", "3"))

# 使用的 embedding 模型
# paraphrase-multilingual-mpnet-base-v2 支援中英文，適合你的場景
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


# ══════════════════════════════════════════════
# 1. 資料庫相關操作
# ══════════════════════════════════════════════

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
    從資料庫取出最近 N 天的「新聞類」文章
    
    為什麼只處理新聞類？
      - arXiv 論文有唯一 ID，天然不重複
      - 重複問題主要出現在 B 類新聞媒體（TechCrunch、科技報橘等）
      - 對應 Bob 的設計：只對新聞來源的文章做去重
    
    回傳：list of dict，每個 dict 是一篇文章
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
                a.content_type = 'rss'          -- 只處理 RSS 新聞
                AND a.governance_status = 'pending'  -- 只處理還沒處理過的
                AND a.fetched_at >= NOW() - INTERVAL '%s days'
            ORDER BY a.published_at DESC
        """, (DAYS_TO_CHECK,))
        
        rows = cur.fetchall()
        columns = ["id", "title", "content", "published_at", 
                   "source_name", "source_tier"]
        return [dict(zip(columns, row)) for row in rows]


def mark_as_duplicate(conn, article_ids: list):
    """
    把被標記為重複的文章，更新 governance_status 為 'duplicate'
    
    這樣後續的內容價值評分和標籤步驟就會跳過這些文章
    原始資料不刪除，只是標記狀態，保留完整記錄
    """
    if not article_ids:
        return
    
    with conn.cursor() as cur:
        # %s 是 psycopg2 的佔位符
        # tuple(article_ids) 轉成 SQL 的 IN (1, 2, 3) 格式
        cur.execute("""
            UPDATE articles
            SET governance_status = 'duplicate'
            WHERE id = ANY(%s)
        """, (article_ids,))
        conn.commit()
    
    logger.info(f"   標記 {len(article_ids)} 篇為重複")


# ══════════════════════════════════════════════
# 2. Embedding 向量化
# ══════════════════════════════════════════════

def load_embedding_model():
    """
    載入 sentence-transformers 模型
    
    第一次執行時會自動下載模型（約 400MB）
    之後會從快取讀取，不需要重新下載
    
    前端類比：就像 npm install，第一次慢，之後快
    """
    logger.info(f"   載入 Embedding 模型：{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("   ✅ 模型載入完成")
    return model


def get_article_text(article: dict) -> str:
    """
    取得文章的代表性文字，用來做 embedding
    
    對應 Bob 的設計：用 first chunk 做代表
    我們簡化成：用標題 + 內文前 500 字
    
    為什麼不用全文？
      - 全文太長，embedding 效果反而不好
      - 標題通常已經包含最核心的主題
    """
    title = article.get("title", "") or ""
    content = article.get("content", "") or ""
    
    # 取內文前 500 字（避免太長）
    content_preview = content[:500] if content else ""
    
    # 組合：標題 + 換行 + 內文前段
    return f"{title}\n{content_preview}".strip()


def embed_articles(model, articles: list) -> np.ndarray:
    """
    把所有文章轉成向量矩陣
    
    輸入：N 篇文章的文字清單
    輸出：N × 768 的矩陣（每篇文章是一個 768 維的向量）
    
    前端類比：
      輸入：["文章A的文字", "文章B的文字", ...]
      輸出：[[0.23, 0.87, ...], [0.51, 0.12, ...], ...]
    """
    texts = [get_article_text(a) for a in articles]
    
    logger.info(f"   對 {len(texts)} 篇文章做向量化...")
    
    # batch_size=32：每次處理 32 篇，避免記憶體不足
    # show_progress_bar=True：顯示進度條
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True   # 正規化，讓 cosine similarity 計算更準確
    )
    
    logger.info(f"   ✅ 向量化完成，維度：{embeddings.shape}")
    return embeddings


# ══════════════════════════════════════════════
# 3. Clustering 聚類分析
# ══════════════════════════════════════════════

def cluster_articles(embeddings: np.ndarray) -> np.ndarray:
    """
    用 Average Linkage + Fcluster 把相似的文章分成群
    
    對應 Bob 的設計：
      clustering algorithm: Average Linkage + Fcluster（threshold=0.15）
    
    步驟：
    1. 計算所有文章兩兩之間的距離（distance = 1 - similarity）
    2. Average Linkage：把距離近的文章逐步合併成樹狀結構
    3. Fcluster：在 threshold 的地方切斷樹，得到各個 cluster
    
    回傳：每篇文章的 cluster 編號陣列
    例如：[1, 1, 1, 2, 2, 3] → 前三篇是同一群，中間兩篇是另一群，最後一篇獨立
    """
    # 計算 cosine similarity 矩陣（N × N）
    # similarity[i][j] = 第 i 篇和第 j 篇的相似度
    similarity_matrix = cosine_similarity(embeddings)
    
    # 轉成距離矩陣（distance = 1 - similarity）
    # 相似度越高 → 距離越小 → 越容易被歸同一群
    distance_matrix = 1 - similarity_matrix
    
    # 確保距離矩陣的值在 [0, 1] 之間（避免浮點數誤差）
    distance_matrix = np.clip(distance_matrix, 0, 1)

    # ★ 強制對角線為 0（修正浮點數誤差）
    np.fill_diagonal(distance_matrix, 0)
    
    # 把矩陣轉成「壓縮距離矩陣」（linkage 需要的格式）
    # squareform：把 N×N 矩陣轉成 N*(N-1)/2 的一維陣列
    from scipy.spatial.distance import squareform
    condensed = squareform(distance_matrix)
    
    # Average Linkage 聚類
    # method='average'：用群與群之間的平均距離來判斷要不要合併
    Z = linkage(condensed, method='average')
    
    # 根據 threshold 切割，得到 cluster 標籤
    # criterion='distance'：用距離來切割
    labels = fcluster(Z, t=DEDUP_THRESHOLD, criterion='distance')
    
    return labels


# ══════════════════════════════════════════════
# 4. 決定哪些文章要保留
# ══════════════════════════════════════════════

def find_duplicates(articles: list, cluster_labels: np.ndarray) -> list:
    """
    根據 cluster 結果，找出要標記為重複的文章 ID
    
    保留策略（對應 Bob 的設計）：
      每個 cluster 保留最新和最舊的文章
      其他的標記為 duplicate
    
    為什麼保留最新和最舊？
      最新 = 最即時的報導
      最舊 = 通常是原始來源或第一手報導
      中間的 = 通常是跟風報導，資訊重複價值低
    
    回傳：要標記為 duplicate 的文章 ID 清單
    """
    # 把文章按 cluster 分組
    # cluster_groups = { cluster_id: [article1, article2, ...] }
    cluster_groups = {}
    for article, label in zip(articles, cluster_labels):
        label = int(label)
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(article)
    
    duplicate_ids = []
    
    for label, group in cluster_groups.items():
        # 只有一篇的 cluster → 沒有重複，跳過
        if len(group) == 1:
            continue
        
        # 有多篇的 cluster → 找出重複的
        logger.info(f"   發現相似群組 {label}：{len(group)} 篇")
        for a in group:
            logger.info(f"     - [{a['source_name']}] {a['title'][:40]}...")
        
        # 按發布時間排序
        sorted_group = sorted(
            group,
            key=lambda x: x["published_at"] if x["published_at"] else 0
        )
        
        # 保留最舊（第一篇）和最新（最後一篇）
        keep_oldest = sorted_group[0]["id"]
        keep_newest = sorted_group[-1]["id"]
        
        # 中間的全部標記為重複
        for article in sorted_group[1:-1]:
            duplicate_ids.append(article["id"])
        
        # 如果只有兩篇，最舊和最新是同樣兩篇，都保留
        logger.info(f"     保留：ID {keep_oldest}（最舊）和 ID {keep_newest}（最新）")
        logger.info(f"     標記重複：{len(sorted_group) - 2} 篇")
    
    return duplicate_ids


# ══════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════

def run_deduplication():
    """
    執行完整的語意去重流程
    
    流程：
    1. 從資料庫取出最近 N 天的新聞文章
    2. 用 sentence-transformers 做 embedding
    3. Average Linkage clustering
    4. 找出重複的文章
    5. 更新資料庫狀態
    """
    logger.info("=" * 50)
    logger.info("開始語意去重")
    logger.info("=" * 50)
    
    conn = None
    try:
        conn = get_db_connection()
        
        # ── Step 1：取出文章 ──
        articles = fetch_recent_news_articles(conn)
        logger.info(f"取出 {len(articles)} 篇待去重文章（最近 {DAYS_TO_CHECK} 天）")
        
        if len(articles) < 2:
            logger.info("   文章數不足，跳過去重")
            return
        
        # ── Step 2：Embedding ──
        model = load_embedding_model()
        embeddings = embed_articles(model, articles)
        
        # ── Step 3：Clustering ──
        logger.info(f"   開始聚類分析（threshold={DEDUP_THRESHOLD}）")
        cluster_labels = cluster_articles(embeddings)
        
        unique_clusters = len(set(cluster_labels))
        logger.info(f"   共分成 {unique_clusters} 個群組")
        
        # ── Step 4：找重複 ──
        duplicate_ids = find_duplicates(articles, cluster_labels)
        logger.info(f"   發現 {len(duplicate_ids)} 篇重複文章")
        
        # ── Step 5：更新資料庫 ──
        if duplicate_ids:
            mark_as_duplicate(conn, duplicate_ids)
        
        logger.info("=" * 50)
        logger.info(f"去重完成")
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


# ══════════════════════════════════════════════
# 6. 直接執行（測試用）
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/dedup.log", encoding="utf-8")
        ]
    )
    
    run_deduplication()