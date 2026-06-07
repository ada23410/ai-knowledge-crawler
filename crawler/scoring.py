"""
內容價值評分模組（Content Value Scoring）
=====================================================
功能：對每篇文章依據規則式評分，計算 credibility_score（0～1）
      並更新 governance_status

評分設計參考：
  - 來源等級基礎分（A/B/C）
  - 金融保險業專屬關鍵字加分
  - 負面內容扣分
  - 最終分數決定是否進入知識庫

狀態判定（v2）：
  approved  → score >= 0.60，自動進入知識庫
  rejected  → score <  0.60，直接淘汰，不進入知識庫

移除項目（v2）：
  - arXiv +0.05 額外加分（避免學術論文過度進入知識庫）
  - review / human_required 中間狀態（簡化為二元判定）

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
    'A': 0.65,   # 官方 / 研究機構：最可信
    'B': 0.50,   # 產業媒體：次之
    'C': 0.60,   # 機構 / 政策：在地化關鍵
}

# 分流門檻（v2：二元判定）
THRESHOLD_APPROVED = 0.60   # 高於此值 → approved，進入知識庫
                             # 低於此值 → rejected，直接淘汰

# ── 加分關鍵字（英文）──
BONUS_KEYWORDS_EN = {

    # 監理合規（最重要，金融業核心）
    "regulatory":            0.15,
    "regulation":            0.15,
    "regulator":             0.15,
    "regulated":             0.15,
    "compliance":            0.15,
    "compliant":             0.15,
    "non-compliance":        0.15,
    "conform":               0.12,
    "conformance":           0.12,
    "adherence":             0.12,
    "fsa":                   0.15,
    "fsb":                   0.15,
    "iais":                  0.15,
    "basel":                 0.15,
    "basel iii":             0.15,
    "basel iv":              0.15,
    "explainability":        0.12,
    "explainable":           0.12,
    "xai":                   0.12,
    "interpretability":      0.12,
    "interpretable":         0.12,
    "model risk":            0.12,
    "model risk management": 0.12,
    "mrm":                   0.12,
    "gdpr":                  0.12,
    "data protection":       0.12,
    "personal data":         0.12,
    "privacy":               0.10,
    "data privacy":          0.10,
    "confidentiality":       0.10,
    "audit":                 0.10,
    "auditing":              0.10,
    "auditor":               0.10,
    "internal audit":        0.10,
    "external audit":        0.10,
    "governance":            0.10,
    "ai governance":         0.12,
    "oversight":             0.10,
    "accountability":        0.10,
    "transparency":          0.10,
    "supervisory":           0.10,
    "supervision":           0.10,

    # 保險業應用
    "underwriting":          0.10,
    "underwriter":           0.10,
    "underwrite":            0.10,
    "risk assessment":       0.10,
    "policy":                0.08,
    "policyholder":          0.10,
    "premium":               0.08,
    "claims":                0.10,
    "claim":                 0.10,
    "claims processing":     0.10,
    "claims management":     0.10,
    "loss ratio":            0.10,
    "fraud detection":       0.10,
    "fraud":                 0.08,
    "fraudulent":            0.08,
    "anti-fraud":            0.10,
    "actuarial":             0.10,
    "actuary":               0.10,
    "mortality":             0.08,
    "morbidity":             0.08,
    "insurtech":             0.10,
    "insurance technology":  0.10,
    "reinsurance":           0.08,
    "reinsurer":             0.08,
    "catastrophe model":     0.08,
    "cat model":             0.08,

    # 銀行 / 金融應用
    "credit scoring":        0.10,
    "credit score":          0.10,
    "creditworthiness":      0.10,
    "credit risk":           0.10,
    "default":               0.08,
    "aml":                   0.10,
    "anti-money laundering": 0.10,
    "money laundering":      0.10,
    "financial crime":       0.10,
    "suspicious activity":   0.10,
    "sar":                   0.08,
    "kyc":                   0.10,
    "know your customer":    0.10,
    "customer due diligence":0.10,
    "cdd":                   0.08,
    "risk management":       0.10,
    "risk mitigation":       0.10,
    "operational risk":      0.10,
    "market risk":           0.10,
    "systemic risk":         0.10,
    "fintech":               0.08,
    "financial technology":  0.08,
    "open banking":          0.08,
    "embedded finance":      0.08,

    # 新模型 / 新技術（只保留金融保險業語境下具體的詞）
    "new model":             0.10,
    "model release":         0.10,
    "unveil":                0.08,
    "breakthrough":          0.10,
    "state-of-the-art":      0.08,
    "sota":                  0.08,
    "frontier":              0.08,
    "cutting-edge":          0.08,

    # 產業應用落地（只保留具體落地動作）
    "deployment":            0.08,
    "deploy":                0.08,
    "deployed":              0.08,
    "rollout":               0.08,
    "go live":               0.08,
    "at scale":              0.08,
    "real-world":            0.08,
    "real world":            0.08,
    "use case":              0.08,
    "use-case":              0.08,
    "case study":            0.08,
    "pilot":                 0.08,
    "proof of concept":      0.08,
}

# ── 加分關鍵字（中文）──
BONUS_KEYWORDS_ZH = {

    # 監理合規
    "金管會":           0.15,
    "金融監督管理委員會": 0.15,
    "fsc":              0.15,
    "監理":             0.15,
    "監管":             0.15,
    "管制":             0.12,
    "合規":             0.15,
    "法遵":             0.15,
    "遵法":             0.12,
    "符合規範":         0.12,
    "法規":             0.12,
    "法令":             0.12,
    "規範":             0.10,
    "準則":             0.10,
    "個資":             0.12,
    "個人資料":         0.12,
    "隱私":             0.10,
    "隱私權":           0.10,
    "可解釋":           0.12,
    "可解釋性":         0.12,
    "透明":             0.10,
    "透明度":           0.10,
    "問責":             0.10,
    "稽核":             0.10,
    "內稽":             0.10,
    "內部稽核":         0.10,
    "外部稽核":         0.10,
    "查核":             0.10,
    "資安":             0.10,
    "資訊安全":         0.10,
    "網路安全":         0.10,
    "治理":             0.10,
    "AI 治理":          0.12,
    "模型治理":         0.12,
    "風控":             0.10,
    "風險控管":         0.10,
    "內控":             0.10,
    "內部控制":         0.10,

    # 保險業應用
    "核保":             0.10,
    "承保":             0.10,
    "保單審查":         0.10,
    "理賠":             0.10,
    "理賠審查":         0.10,
    "給付":             0.10,
    "賠付":             0.10,
    "詐欺":             0.10,
    "詐騙":             0.10,
    "保險詐欺":         0.10,
    "精算":             0.10,
    "精算師":           0.10,
    "死亡率":           0.08,
    "發病率":           0.08,
    "保費":             0.08,
    "損失率":           0.10,
    "保險科技":         0.10,
    "再保險":           0.08,
    "再保":             0.08,
    "巨災模型":         0.08,
    "保戶":             0.08,
    "要保人":           0.08,

    # 銀行 / 金融應用
    "信用評分":         0.10,
    "信用分數":         0.10,
    "信用評等":         0.10,
    "信用風險":         0.10,
    "違約":             0.08,
    "洗錢防制":         0.10,
    "反洗錢":           0.10,
    "洗錢":             0.08,
    "金融犯罪":         0.10,
    "可疑交易":         0.10,
    "可疑交易申報":     0.10,
    "風險管理":         0.10,
    "風險控管":         0.10,
    "作業風險":         0.10,
    "市場風險":         0.10,
    "系統性風險":       0.10,
    "金融科技":         0.08,
    "開放銀行":         0.08,
    "數位金融":         0.08,
    "嵌入式金融":       0.08,

    # 新模型 / 新技術（只保留具體且專屬的詞）
    "新模型":           0.10,
    "模型發布":         0.10,
    "突破":             0.10,
    "前沿":             0.08,
    "尖端":             0.08,

    # 產業應用落地（只保留具體落地動作）
    "導入":             0.08,
    "落地":             0.08,
    "實際應用":         0.08,
    "部署":             0.08,
    "規模化":           0.08,
    "案例":             0.08,
    "成功案例":         0.08,
    "試點":             0.08,
    "概念驗證":         0.08,
}

# ── 扣分關鍵字（負面過濾）──
PENALTY_KEYWORDS_EN = {
    "sponsored":     -0.15,
    "advertisement": -0.15,
    "press release": -0.10,
    "click here":    -0.10,
    "subscribe":     -0.05,
    "newsletter":    -0.05,
    "most influential":  -0.10,   # 排行榜類
    "election safeguard": -0.12,  # 選舉安全類
    "startup forum":     -0.08,   # 新創活動類
}

PENALTY_KEYWORDS_ZH = {
    "廣告": -0.15,
    "贊助": -0.15,
    "業配": -0.15,
    "促銷": -0.10,
    "優惠": -0.08,
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
    取出所有待評分的文章。

    條件：
      - governance_status = 'pending'
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
    更新文章的評分和狀態。

    status 可能的值（v2）：
      approved → score >= 0.60，進入知識庫
      rejected → score <  0.60，直接淘汰
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE articles
            SET
                credibility_score = %s,
                governance_status = %s
            WHERE id = %s
        """, (score, status, article_id))
    conn.commit()


# ══════════════════════════════════════════════
# 3. 評分核心邏輯
# ══════════════════════════════════════════════

def _keyword_match(keyword: str, text: str) -> bool:
    """
    判斷關鍵字是否出現在文本中。

    英文關鍵字使用正則詞邊界（\\b），避免部分匹配誤觸發。
    中文關鍵字直接用 in 比對（中文無詞邊界概念）。

    Args:
        keyword: 關鍵字字串
        text:    搜尋文本（已轉小寫）

    Returns:
        bool，True 代表命中
    """
    kw_lower = keyword.lower()
    if kw_lower.isascii():
        pattern = r"\b" + re.escape(kw_lower) + r"\b"
        return bool(re.search(pattern, text))
    else:
        return kw_lower in text


def calculate_score(article: dict) -> tuple:
    """
    計算一篇文章的內容價值分數。

    回傳：(score, status, score_breakdown)
      score           = 最終分數（0～1）
      status          = approved / rejected
      score_breakdown = 每個加分項目的明細（除錯用）

    計算方式：
      基礎分 + 加分（上限 +0.30）+ 扣分，最後 clip 到 [0, 1]

    注意（v2）：
      - 移除 arXiv +0.05 額外加分，避免學術論文過度進入知識庫
      - 狀態簡化為二元判定：approved（≥0.60）/ rejected（<0.60）

    注意（v3）：
      - arXiv 基礎分從 0.80 降至 0.40，需命中金融保險關鍵字才能通過
      - 移除通用加分關鍵字（benchmark、performance 等）
    """
    breakdown = {}

    # ── Step 1：來源等級基礎分 ──
    # arXiv 論文單獨設定較低基礎分（0.40），
    # 必須命中至少一個金融保險專屬關鍵字（+0.10 以上）才能達到 0.60 門檻，
    # 避免純技術學術論文直接憑 A 類基礎分（0.80）通過審查。
    tier         = article.get("source_tier", "B") or "B"
    content_type = article.get("content_type", "") or ""

    if content_type == "arxiv":
        base_score = 0.40
    else:
        base_score = TIER_BASE_SCORE.get(tier, 0.50)

    breakdown["base_score"]   = base_score
    breakdown["content_type"] = content_type
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

    # ── Step 4：關鍵字加分（上限 +0.30）──
    bonus_total   = 0.0
    matched_bonus = []

    for keyword, bonus in bonus_keywords.items():
        if _keyword_match(keyword, text):
            bonus_total += bonus
            matched_bonus.append(f"{keyword}(+{bonus})")

    bonus_total = min(bonus_total, 0.30)
    score += bonus_total
    breakdown["bonus"]         = bonus_total
    breakdown["matched_bonus"] = matched_bonus

    # ── Step 5：負面關鍵字扣分 ──
    penalty_total   = 0.0
    matched_penalty = []

    for keyword, penalty in penalty_keywords.items():
        if _keyword_match(keyword, text):
            penalty_total += penalty
            matched_penalty.append(f"{keyword}({penalty})")

    score += penalty_total  # penalty 本身是負數
    breakdown["penalty"]         = penalty_total
    breakdown["matched_penalty"] = matched_penalty

    # ── Step 6：確保分數在 [0, 1] 之間 ──
    score = round(max(0.0, min(1.0, score)), 4)
    breakdown["final_score"] = score

    # ── Step 7：二元狀態判定（v2）──
    # approved → 進入知識庫
    # rejected → 直接淘汰，不再進行人工審稿
    status = "approved" if score >= THRESHOLD_APPROVED else "rejected"

    return score, status, breakdown


# ══════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════

def run_scoring():
    """
    執行完整的內容價值評分流程。

    流程：
      1. 取出所有 pending 文章
      2. 逐篇計算分數
      3. 更新資料庫（approved / rejected）
      4. 輸出統計報告
    """
    logger.info("=" * 50)
    logger.info("開始內容價值評分（v2：二元判定）")
    logger.info("=" * 50)

    conn = None

    stats = {
        "approved": 0,
        "rejected": 0,
        "total":    0,
    }

    try:
        conn = get_db_connection()

        articles = fetch_pending_articles(conn)
        logger.info("待評分文章：%d 篇", len(articles))

        if not articles:
            logger.info("   沒有待評分的文章")
            return

        for i, article in enumerate(articles, 1):

            score, status, breakdown = calculate_score(article)
            update_article_score(conn, article["id"], score, status)

            stats[status] += 1
            stats["total"] += 1

            if i % 50 == 0:
                logger.info("   進度：%d/%d 篇", i, len(articles))

            if score >= 0.75:
                logger.info(
                    "高分 [%.4f][%s] %s",
                    score, article["source_name"], article["title"][:40]
                )
            elif score < 0.40:
                logger.info(
                    "低分 [%.4f][%s] %s",
                    score, article["source_name"], article["title"][:40]
                )

        logger.info("=" * 50)
        logger.info("評分完成")
        logger.info("   總計處理：%d 篇", stats["total"])
        logger.info("   approved（>= 0.60）：%d 篇", stats["approved"])
        logger.info("   rejected（<  0.60）：%d 篇", stats["rejected"])
        logger.info("=" * 50)

    except Exception as e:
        logger.error("評分失敗：%s", e)
        raise

    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════
# 5. 單篇測試（除錯用）
# ══════════════════════════════════════════════

def test_single_article(article_id: int):
    """
    測試單篇文章的評分，顯示詳細評分明細。

    執行方式：python scoring.py test <article_id>
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
        print(f"內容類型：{article['content_type']}")
        print(f"{'='*50}")
        print(f"基礎分：{breakdown['base_score']}")
        print(f"加分：+{breakdown['bonus']}")
        if breakdown['matched_bonus']:
            print(f"  觸發關鍵字：{', '.join(breakdown['matched_bonus'])}")
        print(f"扣分：{breakdown['penalty']}")
        if breakdown['matched_penalty']:
            print(f"  觸發關鍵字：{', '.join(breakdown['matched_penalty'])}")
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

    # python scoring.py          → 執行全部評分
    # python scoring.py test 123 → 測試單篇文章（ID=123）
    if len(sys.argv) >= 3 and sys.argv[1] == "test":
        article_id = int(sys.argv[2])
        test_single_article(article_id)
    else:
        run_scoring()