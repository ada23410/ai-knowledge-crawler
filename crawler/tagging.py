"""
tagging.py — 自動標籤模組
===========================
處理範圍：governance_status = 'approved' 且 tags IS NULL 的文章
執行方式：由 main.py 在 scoring.py 完成後呼叫，或獨立執行
策略說明：純關鍵字規則引擎，雙語比對，支援多標籤，不消耗 LLM Token
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# log設定
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/tagging.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# 標籤規則定義
TAG_RULES: dict[str, dict[str, list[str]]] = {
    "監理合規": {
        "zh": [
            "金管會", "合規", "監理", "稽核", "法遵", "KYC", "AML",
            "洗錢防制", "資安", "個資", "個人資料保護", "GDPR", "隱私",
            "資訊安全", "風控", "內控", "內部稽核", "監管",
        ],
        "en": [
            "regulatory", "compliance", "audit", "KYC", "AML",
            "anti-money laundering", "GDPR", "privacy", "data protection",
            "cybersecurity", "information security", "risk control",
            "internal control", "supervision", "oversight",
        ],
    },
    "模型技術": {
        "zh": [
            "模型", "大語言模型", "微調", "預訓練", "訓練", "推論",
            "基準測試", "評測", "架構", "Transformer", "注意力機制",
            "向量", "嵌入", "參數", "量化", "剪枝", "知識蒸餾",
            "多模態", "生成式", "生成 AI", "生成式 AI",
        ],
        "en": [
            "LLM", "large language model", "fine-tuning", "pre-training",
            "training", "inference", "benchmark", "architecture",
            "transformer", "attention", "embedding", "parameter",
            "quantization", "pruning", "distillation", "multimodal",
            "generative AI", "foundation model", "RLHF", "RAG",
            "retrieval augmented", "agentic", "agent",
        ],
    },
    "產業應用": {
        "zh": [
            "導入", "落地", "實作", "應用", "案例", "試點", "POC",
            "概念驗證", "生產環境", "商業化", "解決方案", "賦能",
            "數位轉型", "自動化", "智慧化", "核保", "理賠", "客服",
            "聊天機器人", "推薦系統", "詐欺偵測",
        ],
        "en": [
            "deployment", "case study", "implementation", "use case",
            "production", "commercial", "solution", "pilot", "POC",
            "proof of concept", "digital transformation", "automation",
            "underwriting", "claims", "customer service", "chatbot",
            "recommendation", "fraud detection", "enterprise",
        ],
    },
    "政策法規": {
        "zh": [
            "法規", "政策", "白皮書", "草案", "修法", "立法",
            "行政院", "國發會", "數位部", "科技部", "行政命令",
            "公告", "指引", "準則", "規範", "條例", "辦法",
        ],
        "en": [
            "AI Act", "policy", "regulation", "legislation", "bill",
            "executive order", "guideline", "standard", "framework",
            "white paper", "governance", "principle", "act", "law",
        ],
    },
    "學術研究": {
        "zh": [
            "論文", "研究", "實驗", "資料集", "基準", "評估",
            "實驗結果", "消融實驗", "研究團隊", "學術", "期刊",
            "會議論文", "NeurIPS", "ICML", "ACL", "ICLR", "AAAI",
        ],
        "en": [
            "arXiv", "paper", "research", "experiment", "dataset",
            "ablation", "evaluation", "benchmark", "academic",
            "journal", "conference", "NeurIPS", "ICML", "ACL",
            "ICLR", "AAAI", "preprint", "study",
        ],
    },
    "金融科技": {
        "zh": [
            "金融科技", "FinTech", "數位金融", "開放銀行", "區塊鏈",
            "加密貨幣", "數位資產", "支付", "信用評分", "徵信",
            "財富管理", "保險科技", "InsurTech", "投資", "資產管理",
        ],
        "en": [
            "FinTech", "digital finance", "open banking", "blockchain",
            "cryptocurrency", "digital asset", "payment", "credit scoring",
            "wealth management", "InsurTech", "investment", "asset management",
            "neobank", "robo-advisor",
        ],
    },
    "開源生態": {
        "zh": [
            "開源", "開放原始碼", "GitHub", "社群", "貢獻",
            "授權", "Apache", "MIT 授權",
        ],
        "en": [
            "open source", "open-source", "GitHub", "community",
            "contribution", "license", "Apache", "MIT license",
            "Hugging Face", "open weights", "open model",
        ],
    },
    "AI 安全": {
        "zh": [
            "AI 安全", "對齊", "可解釋", "可解釋性", "偏見",
            "公平性", "幻覺", "越獄", "紅隊", "魯棒性", "可信賴",
        ],
        "en": [
            "AI safety", "alignment", "explainability", "interpretability",
            "bias", "fairness", "hallucination", "jailbreak", "red team",
            "robustness", "trustworthy", "responsible AI", "XAI",
        ],
    },
}

BATCH_SIZE = 250


class ArticleTagger:
    """
    自動標籤模組主類別。

    依據預定義的關鍵字規則，對 governance_status = 'approved' 的文章
    進行雙語標籤比對，並將結果以 JSONB 格式寫回資料庫。

    屬性說明：
        conn: psycopg2 資料庫連線物件
        tag_rules: 標籤名稱 → {語言 → 關鍵字清單} 的規則字典
    """

    def __init__(self) -> None:
        """初始化資料庫連線並載入標籤規則。"""
        self.conn = self._connect_db()
        self.tag_rules = TAG_RULES
        logger.info("ArticleTagger 初始化完成，共載入 %d 個標籤類別", len(self.tag_rules))

    # ──────────────────────────────────────────
    # 資料庫連線
    # ──────────────────────────────────────────

    def _connect_db(self) -> psycopg2.extensions.connection:
        """
        建立 PostgreSQL 連線。

        Returns:
            psycopg2 連線物件

        Raises:
            psycopg2.OperationalError: 連線失敗時拋出
        """
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        conn.autocommit = False
        logger.info("資料庫連線成功")
        return conn

    # ──────────────────────────────────────────
    # 文章撈取
    # ──────────────────────────────────────────

    def _fetch_pending_articles(self) -> list[dict]:
        """
        從資料庫撈取待打標的文章。

        條件：
          - governance_status = 'approved'
          - tags IS NULL（尚未打標）
          - is_deleted = FALSE

        Returns:
            文章資料列表，每筆為 dict，包含 id / title / content / language
        """
        sql = """
            SELECT id, title, content, language
            FROM articles
            WHERE governance_status = 'approved'
              AND tags IS NULL
              AND is_deleted = FALSE
            ORDER BY fetched_at ASC
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        logger.info("待打標文章共 %d 篇", len(rows))
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────
    # 核心比對邏輯
    # ──────────────────────────────────────────

    def _build_search_text(self, article: dict) -> str:
        """
        組合用於關鍵字比對的搜尋文本。

        策略：標題（全文）+ 內文前 1000 字，轉為小寫以利英文比對。

        Args:
            article: 包含 title / content 的文章 dict

        Returns:
            合併後的搜尋文字字串
        """
        title = article.get("title") or ""
        content = article.get("content") or ""
        combined = f"{title} {content[:1000]}"
        return combined.lower()

    def _match_tags(self, article: dict) -> tuple[list[str], dict[str, list[str]]]:
        """
        對單篇文章執行關鍵字比對，回傳命中的標籤與觸發關鍵字。

        雙語策略：
          - language = 'zh'：優先比對中文關鍵字，同時比對英文（技術術語）
          - language = 'en'：只比對英文關鍵字
          - 其他 / 未知：同時比對中英文

        Args:
            article: 文章 dict（含 language）

        Returns:
            (matched_tags, matched_keywords)
              matched_tags: 命中的標籤名稱列表
              matched_keywords: {標籤名稱: [觸發的關鍵字列表]}，用於 log
        """
        language = (article.get("language") or "").strip().lower()
        search_text = self._build_search_text(article)

        matched_tags: list[str] = []
        matched_keywords: dict[str, list[str]] = {}

        for tag_name, lang_dict in self.tag_rules.items():
            # 決定要比對哪些語言的關鍵字
            if language == "zh":
                keywords_to_check = lang_dict.get("zh", []) + lang_dict.get("en", [])
            elif language == "en":
                keywords_to_check = lang_dict.get("en", [])
            else:
                # 未知語言：雙語都比對
                keywords_to_check = lang_dict.get("zh", []) + lang_dict.get("en", [])

            hit_keywords: list[str] = []
            for kw in keywords_to_check:
                # 使用 \b 做英文詞邊界比對；中文直接 in 判斷
                if self._keyword_match(kw, search_text):
                    hit_keywords.append(kw)

            if hit_keywords:
                matched_tags.append(tag_name)
                matched_keywords[tag_name] = hit_keywords

        return matched_tags, matched_keywords

    @staticmethod
    def _keyword_match(keyword: str, text: str) -> bool:
        """
        判斷關鍵字是否出現在文本中。

        英文關鍵字使用正則詞邊界（\\b），避免部分匹配誤觸發。
        中文關鍵字直接用 in 比對（中文無詞邊界概念）。

        Args:
            keyword: 關鍵字字串（已轉小寫）
            text: 搜尋文本（已轉小寫）

        Returns:
            bool，True 代表命中
        """
        kw_lower = keyword.lower()
        # 判斷是否為純 ASCII（英文）
        if kw_lower.isascii():
            pattern = r"\b" + re.escape(kw_lower) + r"\b"
            return bool(re.search(pattern, text))
        else:
            return kw_lower in text

    # ──────────────────────────────────────────
    # 資料庫寫回
    # ──────────────────────────────────────────

    def _update_tags_batch(self, updates: list[dict]) -> int:
        """
        批次更新文章的 tags 欄位。

        Args:
            updates: [{"id": UUID, "tags": [...]}] 列表

        Returns:
            成功更新的筆數
        """
        if not updates:
            return 0

        sql = """
            UPDATE articles
            SET tags = %(tags)s
            WHERE id = %(id)s
        """
        updated = 0
        with self.conn.cursor() as cur:
            for item in updates:
                cur.execute(sql, {
                    "id": item["id"],
                    "tags": json.dumps(item["tags"], ensure_ascii=False),
                })
                updated += cur.rowcount
        self.conn.commit()
        return updated

    # ──────────────────────────────────────────
    # 統計報告
    # ──────────────────────────────────────────

    def _log_summary(
        self,
        total: int,
        tagged: int,
        no_tag: int,
        tag_counter: dict[str, int],
    ) -> None:
        """
        輸出本次打標結果統計至 log。

        Args:
            total: 處理總篇數
            tagged: 至少命中一個標籤的篇數
            no_tag: 未命中任何標籤的篇數
            tag_counter: {標籤名稱: 命中篇數} 統計字典
        """
        logger.info("=" * 60)
        logger.info("【打標完成統計】")
        logger.info("  總處理篇數：%d", total)
        logger.info("  成功打標：%d 篇（%.1f%%）", tagged, tagged / total * 100 if total else 0)
        logger.info("  未命中標籤：%d 篇", no_tag)
        logger.info("  各標籤命中數：")
        for tag, count in sorted(tag_counter.items(), key=lambda x: -x[1]):
            logger.info("    %-12s : %d 篇", tag, count)
        logger.info("=" * 60)

    # ──────────────────────────────────────────
    # 主執行流程
    # ──────────────────────────────────────────

    def run(self) -> None:
        """
        執行完整打標流程。

        步驟：
          1. 撈取待打標文章
          2. 分批（250 篇）進行關鍵字比對
          3. 批次寫回 tags 至資料庫
          4. 輸出統計 log
        """
        start_time = datetime.now(timezone.utc)
        logger.info("====== 自動打標任務開始 %s ======", start_time.isoformat())

        articles = self._fetch_pending_articles()
        if not articles:
            logger.info("無待打標文章，任務結束")
            return

        total = len(articles)
        tagged_count = 0
        no_tag_count = 0
        tag_counter: dict[str, int] = {tag: 0 for tag in self.tag_rules}
        updates: list[dict] = []

        for i, article in enumerate(articles):
            article_id = article["id"]
            title = (article.get("title") or "（無標題）")[:60]

            matched_tags, matched_keywords = self._match_tags(article)

            if matched_tags:
                tagged_count += 1
                for tag in matched_tags:
                    tag_counter[tag] = tag_counter.get(tag, 0) + 1
                logger.info(
                    "[%d/%d] ✅ %s | 標籤：%s | 觸發詞：%s",
                    i + 1, total, title,
                    "、".join(matched_tags),
                    {k: v[:3] for k, v in matched_keywords.items()},  # 只顯示前3個
                )
            else:
                no_tag_count += 1
                matched_tags = ["未分類"]
                logger.info("[%d/%d] ⚠️  %s | 未命中任何標籤，標記為「未分類」", i + 1, total, title)

            updates.append({"id": article_id, "tags": matched_tags})

            # 每 BATCH_SIZE 筆寫一次 DB
            if len(updates) >= BATCH_SIZE:
                written = self._update_tags_batch(updates)
                logger.info("批次寫入 %d 筆至資料庫", written)
                updates.clear()

        # 寫入剩餘的
        if updates:
            written = self._update_tags_batch(updates)
            logger.info("最終批次寫入 %d 筆至資料庫", written)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info("任務耗時：%.2f 秒", elapsed)
        self._log_summary(total, tagged_count, no_tag_count, tag_counter)

    def close(self) -> None:
        """關閉資料庫連線。"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("資料庫連線已關閉")


# 獨立執行入口
if __name__ == "__main__":
    tagger = ArticleTagger()
    try:
        tagger.run()
    finally:
        tagger.close()