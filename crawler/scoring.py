"""
內容價值評分模組（Content Value Scoring）
=====================================================
功能：對每篇文章依據規則式評分，計算 credibility_score（0～1）
      並更新 governance_status

評分設計參考：
  - 來源等級基礎分（A/B/C）
  - 金融保險業專屬關鍵字加分
  - 負面內容扣分
  - 最終分數決定是否需要人工審稿

Phase 1：純規則式（不需要 LLM）
Phase 2：之後加入本地 Ollama 做語意評分（G-EVAL 概念）
"""

import os
import re
import logging
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1. 評分規則設定
# ══════════════════════════════════════════════

# 來源等級基礎分
TIER_BASE_SCORE = {
    'A': 0.80,   # 官方 / 研究機構：最可信
    'B': 0.50,   # 產業媒體：次之
    'C': 0.60,   # 機構 / 政策：在地化關鍵
}

# 分流門檻
THRESHOLD_AUTO_PASS   = 0.60   # 高於此值 → 自動通過
THRESHOLD_HUMAN_REVIEW = 0.45  # 低於此值 → 必須人工審稿
# 介於兩者之間 → 建議人工審稿

# ── 加分關鍵字（英文）──
BONUS_KEYWORDS_EN = {

    # 監理合規（最重要，金融業核心）
    "regulatory": 0.15,
    "compliance": 0.15,
    "fsa":        0.15,
    "fsb":        0.15,
    "iais":       0.15,
    "basel":      0.15,
    "explainability": 0.12,
    "interpretability": 0.12,
    "model risk": 0.12,
    "gdpr":       0.12,
    "privacy":    0.10,
    "audit":      0.10,
    "governance": 0.10,

    # 保險業應用
    "underwriting":     0.10,
    "claims":           0.10,
    "fraud detection":  0.10,
    "actuarial":        0.10,
    "insurtech":        0.10,
    "reinsurance":      0.08,

    # 銀行 / 金融應用
    "credit scoring":   0.10,
    "aml":              0.10,
    "anti-money laundering": 0.10,
    "kyc":              0.10,
    "risk management":  0.10,
    "fintech":          0.08,

    # 新模型 / 新技術
    "new model":        0.10,
    "release":          0.08,
    "breakthrough":     0.10,
    "state-of-the-art": 0.08,
    "sota":             0.08,
    "benchmark":        0.06,

    # 產業應用落地
    "deployment":       0.08,
    "production":       0.08,
    "real-world":       0.08,
    "use case":         0.08,
    "case study":       0.08,

    # 組織策略 / 市場
    "partnership":      0.06,
    "acquisition":      0.06,
    "investment":       0.06,
    "strategy":         0.06,
}

# ── 加分關鍵字（中文）──
BONUS_KEYWORDS_ZH = {

    # 監理合規
    "金管會":     0.15,
    "監理":       0.15,
    "合規":       0.15,
    "法規":       0.12,
    "個資":       0.12,
    "可解釋":     0.12,
    "模型治理":   0.12,
    "稽核":       0.10,
    "資安":       0.10,
    "隱私":       0.10,

    # 保險業應用
    "核保":       0.10,
    "理賠":       0.10,
    "詐欺":       0.10,
    "精算":       0.10,
    "保險科技":   0.10,
    "再保險":     0.08,

    # 銀行 / 金融應用
    "信用評分":   0.10,
    "洗錢防制":   0.10,
    "反洗錢":     0.10,
    "風控":       0.10,
    "風險管理":   0.10,
    "金融科技":   0.08,

    # 新模型 / 新技術
    "新模型":     0.10,
    "發布":       0.08,
    "突破":       0.10,
    "最新":       0.06,

    # 產業應用落地
    "導入":       0.08,
    "部署":       0.08,
    "實務":       0.08,
    "案例":       0.08,
    "落地":       0.08,

    # 組織策略 / 市場
    "合作":       0.06,
    "投資":       0.06,
    "策略":       0.06,
    "市場":       0.06,
}

# ── 扣分關鍵字（負面過濾）──
PENALTY_KEYWORDS_EN = {
    "sponsored":        -0.15,   # 廣告內容
    "advertisement":    -0.15,
    "press release":    -0.10,   # 新聞稿（可能是行銷）
    "click here":       -0.10,   # 行銷語言
    "subscribe":        -0.05,
    "newsletter":       -0.05,
}

PENALTY_KEYWORDS_ZH = {
    "廣告":       -0.15,
    "贊助":       -0.15,
    "業配":       -0.15,
    "促銷":       -0.10,
    "優惠":       -0.08,
}


# ══════════════════════════════════════════════
# 2. 資料庫操作
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


def fetch_pending_articles(conn) -> list:
    """
    取出所有待評分的文章
    
    條件：
    - governance_status = 'pending'（還沒處理過）
    - 不包含已標記為 duplicate 的文章
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                a.id,
                a.title,
                a.content,
                a.language,
                a.content_type,
                s.source_tier,
                s.name AS source_name
            FROM articles a
            JOIN sources s ON a.source_id = s.id
            WHERE a.governance_status = 'pending'
            ORDER BY a.published_at DESC
        """)
        rows = cur.fetchall()
        columns = ["id", "title", "content", "language",
                   "content_type", "source_tier", "source_name"]
        return [dict(zip(columns, row)) for row in rows]


def update_article_score(conn, article_id: int,
                          score: float, status: str):
    """
    更新文章的評分和狀態
    
    status 可能的值：
      approved      → 自動通過（score > 0.65）
      review        → 建議人工審稿（0.50 ～ 0.65）
      human_required → 必須人工審稿（score < 0.50）
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE articles
            SET 
                credibility_score   = %s,
                governance_status   = %s
            WHERE id = %s
        """, (score, status, article_id))
    conn.commit()


# ══════════════════════════════════════════════
# 3. 評分核心邏輯
# ══════════════════════════════════════════════

def calculate_score(article: dict) -> tuple:
    """
    計算一篇文章的內容價值分數
    
    回傳：(score, status, score_breakdown)
      score           = 最終分數（0～1）
      status          = approved / review / human_required
      score_breakdown = 每個加分項目的明細（除錯用）
    
    計算方式：
      基礎分 + 加分 + 扣分，最後 clip 到 [0, 1]
    """
    breakdown = {}

    # ── Step 1：來源等級基礎分 ──
    tier = article.get("source_tier", "B") or "B"
    base_score = TIER_BASE_SCORE.get(tier, 0.50)
    breakdown["base_score"] = base_score
    score = base_score

    # ── Step 2：準備文字（標題 + 內文前 1000 字）──
    title   = (article.get("title")   or "").lower()
    content = (article.get("content") or "")[:1000].lower()
    text    = f"{title} {content}"

    # ── Step 3：判斷語言，選對應的關鍵字清單 ──
    language = article.get("language", "en") or "en"
    
    if language == "zh":
        bonus_keywords   = BONUS_KEYWORDS_ZH
        penalty_keywords = PENALTY_KEYWORDS_ZH
    else:
        bonus_keywords   = BONUS_KEYWORDS_EN
        penalty_keywords = PENALTY_KEYWORDS_EN

    # ── Step 4：關鍵字加分 ──
    bonus_total = 0.0
    matched_bonus = []

    for keyword, bonus in bonus_keywords.items():
        if keyword.lower() in text:
            bonus_total += bonus
            matched_bonus.append(f"{keyword}(+{bonus})")

    # 加分上限：避免單篇文章因為關鍵字太多而過度加分
    bonus_total = min(bonus_total, 0.30)
    score += bonus_total
    breakdown["bonus"] = bonus_total
    breakdown["matched_bonus"] = matched_bonus

    # ── Step 5：負面關鍵字扣分 ──
    penalty_total = 0.0
    matched_penalty = []

    for keyword, penalty in penalty_keywords.items():
        if keyword.lower() in text:
            penalty_total += penalty
            matched_penalty.append(f"{keyword}({penalty})")

    score += penalty_total   # penalty 本身是負數
    breakdown["penalty"] = penalty_total
    breakdown["matched_penalty"] = matched_penalty

    # ── Step 6：arXiv 額外加分 ──
    # arXiv 論文是一手學術來源，品質穩定
    if article.get("content_type") == "arxiv":
        score += 0.05
        breakdown["arxiv_bonus"] = 0.05

    # ── Step 7：確保分數在 [0, 1] 之間 ──
    score = round(max(0.0, min(1.0, score)), 4)
    breakdown["final_score"] = score

    # ── Step 8：決定狀態 ──
    if score >= THRESHOLD_AUTO_PASS:
        status = "approved"
    elif score >= THRESHOLD_HUMAN_REVIEW:
        status = "review"
    else:
        status = "human_required"

    return score, status, breakdown


# ══════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════

def run_scoring():
    """
    執行完整的內容價值評分流程
    
    流程：
    1. 取出所有 pending 文章
    2. 逐篇計算分數
    3. 更新資料庫
    4. 輸出統計報告
    """
    logger.info("=" * 50)
    logger.info("📊 開始內容價值評分")
    logger.info("=" * 50)

    conn = None

    # 統計用
    stats = {
        "approved":       0,
        "review":         0,
        "human_required": 0,
        "total":          0,
    }

    try:
        conn = get_db_connection()

        # ── Step 1：取出待評分文章 ──
        articles = fetch_pending_articles(conn)
        logger.info(f"待評分文章：{len(articles)} 篇")

        if not articles:
            logger.info("   沒有待評分的文章")
            return

        # ── Step 2：逐篇評分 ──
        for i, article in enumerate(articles, 1):

            score, status, breakdown = calculate_score(article)

            # 更新資料庫
            update_article_score(conn, article["id"], score, status)

            stats[status] += 1
            stats["total"] += 1

            # 每 50 篇輸出一次進度
            if i % 50 == 0:
                logger.info(f"   進度：{i}/{len(articles)} 篇")

            # 高分或低分的文章額外 log（方便確認評分是否合理）
            if score >= 0.75:
                logger.info(
                    f"高分 [{score}] [{article['source_name']}] "
                    f"{article['title'][:40]}..."
                )
            elif score < 0.40:
                logger.info(
                    f"低分 [{score}] [{article['source_name']}] "
                    f"{article['title'][:40]}..."
                )

        # ── Step 3：輸出統計報告 ──
        logger.info("=" * 50)
        logger.info(f"評分完成")
        logger.info(f"   總計處理：{stats['total']} 篇")
        logger.info(f"   自動通過（>= 0.65）：{stats['approved']} 篇")
        logger.info(f"   建議審稿（0.50-0.65）：{stats['review']} 篇")
        logger.info(f"   必須審稿（< 0.50）：{stats['human_required']} 篇")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"評分失敗：{e}")
        raise

    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════
# 5. 單篇測試（除錯用）
# ══════════════════════════════════════════════

def test_single_article(article_id: int):
    """
    測試單篇文章的評分，顯示詳細的評分明細
    方便你確認關鍵字是否正確觸發
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    a.id, a.title, a.content, a.language,
                    a.content_type, s.source_tier, s.name
                FROM articles a
                JOIN sources s ON a.source_id = s.id
                WHERE a.id = %s
            """, (article_id,))
            row = cur.fetchone()

        if not row:
            print(f"找不到文章 ID {article_id}")
            return

        article = {
            "id": row[0], "title": row[1], "content": row[2],
            "language": row[3], "content_type": row[4],
            "source_tier": row[5], "source_name": row[6]
        }

        score, status, breakdown = calculate_score(article)

        print(f"\n{'='*50}")
        print(f"文章 ID：{article['id']}")
        print(f"標題：{article['title']}")
        print(f"來源：{article['source_name']} (Tier {article['source_tier']})")
        print(f"語言：{article['language']}")
        print(f"{'='*50}")
        print(f"基礎分：{breakdown['base_score']}")
        print(f"加分：+{breakdown['bonus']}")
        if breakdown['matched_bonus']:
            print(f"  觸發關鍵字：{', '.join(breakdown['matched_bonus'])}")
        print(f"扣分：{breakdown['penalty']}")
        if breakdown['matched_penalty']:
            print(f"  觸發關鍵字：{', '.join(breakdown['matched_penalty'])}")
        if breakdown.get('arxiv_bonus'):
            print(f"arXiv 加分：+{breakdown['arxiv_bonus']}")
        print(f"{'='*50}")
        print(f"最終分數：{score}")
        print(f"狀態：{status}")
        print(f"{'='*50}\n")

    finally:
        conn.close()


# ══════════════════════════════════════════════
# 6. 程式入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/scoring.log", encoding="utf-8")
        ]
    )

    # 執行模式
    # python scoring.py          → 執行全部評分
    # python scoring.py test 123 → 測試單篇文章（ID=123）
    if len(sys.argv) >= 3 and sys.argv[1] == "test":
        article_id = int(sys.argv[2])
        test_single_article(article_id)
    else:
        run_scoring()