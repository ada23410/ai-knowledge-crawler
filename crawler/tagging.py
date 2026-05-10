"""
tagging.py — 語意向量自動標籤模組
=====================================
處理範圍：governance_status = 'approved' 且 tags IS NULL 的文章
執行方式：由 main.py 在 scoring.py 完成後呼叫，或獨立執行
策略說明：Embedding 相似度分類 + Adaptive-K 動態決定標籤數量

參考文獻：Adaptive-K retrieval（arXiv:2506.08479，Taguchi et al., EMNLP 2025）

流程：
  1. 合併每個標籤的 name + description + synonyms 做向量化
  2. 文章 first chunk 向量化
  3. 計算 cosine similarity，取 TopN=20 候選
  4. Adaptive-K 找最大語意斷層，動態決定 TopK
  5. 選 min(TopK, 5) 個標籤寫入資料庫

優點：
  - 不需要手動維護關鍵字清單
  - 同義詞和語意變體自動涵蓋
  - 未分類比例趨近於 0
  - 每個標籤都有相似度分數，可解釋性高
"""

import json
import logging
import os
import numpy as np
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

load_dotenv()

# ────────────────────────────────────────────────
# 日誌設定
# ────────────────────────────────────────────────
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


# ────────────────────────────────────────────────
# 標籤定義（name + description + synonyms）
# ────────────────────────────────────────────────
TAG_DEFINITIONS: dict[str, dict] = {
    "監理合規": {
        "description": "金融業 AI 監管、法規遵循、稽核、資料保護與隱私治理",
        "synonyms": [
            "regulatory", "compliance", "audit", "governance", "oversight",
            "supervision", "GDPR", "data protection", "privacy", "KYC", "AML",
            "accountability", "transparency", "FSA", "FSB", "IAIS", "Basel",
            "anti-money laundering", "know your customer", "data privacy",
            "information security", "risk control", "internal control",
            "金管會", "合規", "監理", "法遵", "稽核", "個資", "隱私",
            "資安", "資訊安全", "風控", "內控", "監管", "法規遵循",
        ],
    },
    "模型技術": {
        "description": "AI 模型架構、訓練方法、推論優化與語言模型技術",
        "synonyms": [
            "LLM", "large language model", "transformer", "architecture",
            "fine-tuning", "training", "inference", "embedding", "attention",
            "neural network", "foundation model", "multimodal", "RAG",
            "RLHF", "quantization", "distillation", "benchmark",
            "retrieval augmented generation", "agentic", "agent",
            "pre-training", "parameter", "pruning", "generative AI",
            "模型", "訓練", "推論", "架構", "微調", "向量", "嵌入",
            "大語言模型", "生成式", "多模態", "注意力機制", "參數",
        ],
    },
    "產業應用": {
        "description": "AI 技術在企業的實際導入、落地案例與商業化應用",
        "synonyms": [
            "deployment", "implementation", "use case", "case study",
            "production", "pilot", "enterprise", "automation", "chatbot",
            "digital transformation", "real-world", "proof of concept",
            "go live", "rollout", "at scale", "underwriting", "claims",
            "fraud detection", "customer service", "recommendation",
            "導入", "落地", "部署", "應用", "案例", "自動化", "商業化",
            "數位轉型", "核保", "理賠", "客服", "聊天機器人", "試點",
        ],
    },
    "政策法規": {
        "description": "政府 AI 政策、立法動態、法律框架與監管準則",
        "synonyms": [
            "AI Act", "policy", "regulation", "legislation", "bill",
            "executive order", "guideline", "standard", "framework",
            "white paper", "governance", "principle", "law",
            "government policy", "regulatory framework", "legal",
            "法規", "政策", "立法", "白皮書", "準則", "指引", "條例",
            "行政命令", "法令", "規範", "辦法", "草案",
        ],
    },
    "學術研究": {
        "description": "AI 學術論文、實驗研究、資料集與基準測試",
        "synonyms": [
            "arXiv", "paper", "research", "experiment", "dataset",
            "ablation", "evaluation", "benchmark", "academic",
            "journal", "conference", "NeurIPS", "ICML", "ACL", "ICLR",
            "AAAI", "preprint", "study", "empirical", "findings",
            "論文", "研究", "實驗", "資料集", "基準", "評估", "學術",
            "期刊", "會議論文", "實驗結果", "消融實驗",
        ],
    },
    "金融科技": {
        "description": "金融科技創新、數位支付、加密資產與保險科技應用",
        "synonyms": [
            "FinTech", "blockchain", "cryptocurrency", "payment",
            "credit scoring", "wealth management", "InsurTech",
            "investment", "asset management", "open banking",
            "digital finance", "neobank", "robo-advisor",
            "digital asset", "stable coin", "embedded finance",
            "金融科技", "區塊鏈", "加密貨幣", "支付", "投資",
            "保險科技", "數位金融", "開放銀行", "資產管理",
        ],
    },
    "開源生態": {
        "description": "開源 AI 模型、社群貢獻、開放原始碼工具與生態系",
        "synonyms": [
            "open source", "open-source", "GitHub", "community",
            "contribution", "license", "Hugging Face", "open weights",
            "open model", "Apache", "MIT license", "open access",
            "開源", "開放原始碼", "社群", "貢獻", "授權",
        ],
    },
    "AI 安全": {
        "description": "AI 對齊、可解釋性、公平性、幻覺與模型安全防護",
        "synonyms": [
            "AI safety", "alignment", "explainability", "interpretability",
            "bias", "fairness", "hallucination", "jailbreak", "red team",
            "robustness", "trustworthy", "responsible AI", "XAI",
            "adversarial", "safety evaluation", "model safety",
            "AI 安全", "對齊", "可解釋", "可解釋性", "偏見",
            "公平性", "幻覺", "越獄", "紅隊", "魯棒性", "可信賴",
        ],
    },
}

# ────────────────────────────────────────────────
# 向量化設定
# ────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION     = os.getenv("GCP_REGION", "us-central1")
MODEL_NAME     = "gemini-embedding-001"
CHUNK_SIZE     = 500
TOP_N          = 20
MAX_TAGS       = 5
BATCH_SIZE     = 250


class ArticleTagger:
    """
    語意向量自動標籤模組主類別。

    使用 gemini-embedding-001 將文章和標籤定義向量化，
    透過 cosine similarity + Adaptive-K 動態決定每篇文章的標籤。

    屬性說明：
        conn:            psycopg2 資料庫連線物件
        tag_definitions: 標籤定義字典（name → description + synonyms）
        tag_names:       標籤名稱清單（和向量矩陣順序對應）
        tag_embeddings:  標籤向量矩陣，shape = (n_tags, 3072)
        model:           Vertex AI TextEmbeddingModel
    """

    def __init__(self) -> None:
        """初始化 Vertex AI、資料庫連線並預先向量化所有標籤。"""
        self._init_vertex_ai()
        self.model           = TextEmbeddingModel.from_pretrained(MODEL_NAME)
        self.tag_definitions = TAG_DEFINITIONS
        self.conn            = self._connect_db()

        logger.info("預先向量化 %d 個標籤定義...", len(TAG_DEFINITIONS))
        self.tag_names, self.tag_embeddings = self._build_tag_embeddings()
        logger.info("ArticleTagger 初始化完成，標籤向量維度：%s",
                    self.tag_embeddings.shape)

    # ──────────────────────────────────────────
    # Vertex AI 初始化
    # ──────────────────────────────────────────

    def _init_vertex_ai(self) -> None:
        """初始化 Vertex AI SDK。"""
        if not GCP_PROJECT_ID:
            raise ValueError("缺少 GCP_PROJECT_ID，請在 .env 設定")
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        logger.info("Vertex AI 初始化完成（project=%s, region=%s）",
                    GCP_PROJECT_ID, GCP_REGION)

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
            host=os.getenv("POSTGRES_HOST", "db"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            dbname=os.getenv("POSTGRES_DB", "ai_knowledge"),
            user=os.getenv("POSTGRES_USER", "crawler_user"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        conn.autocommit = False
        logger.info("資料庫連線成功")
        return conn

    # ──────────────────────────────────────────
    # Embedding 工具函式
    # ──────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        批次向量化文字清單。

        Args:
            texts: 要向量化的文字列表

        Returns:
            np.ndarray，shape = (len(texts), 3072)
        """
        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = [
                TextEmbeddingInput(text=t, task_type="SEMANTIC_SIMILARITY")
                for t in batch
            ]
            response = self.model.get_embeddings(inputs)
            for emb in response:
                all_embeddings.append(emb.values)

        return np.array(all_embeddings, dtype=np.float32)

    # ──────────────────────────────────────────
    # 標籤向量化（初始化時執行一次）
    # ──────────────────────────────────────────

    def _build_tag_embeddings(self) -> tuple[list[str], np.ndarray]:
        """
        把每個標籤的 name + description + synonyms 合併後向量化。

        合併格式：
          "{標籤名稱}: {description}. 相關詞彙：{synonyms 用逗號串接}"

        Returns:
            (tag_names, tag_embeddings)
            tag_names:      標籤名稱清單
            tag_embeddings: shape = (n_tags, 3072)
        """
        tag_names = []
        tag_texts = []

        for tag_name, tag_info in self.tag_definitions.items():
            description = tag_info["description"]
            synonyms    = ", ".join(tag_info["synonyms"])
            combined    = f"{tag_name}: {description}. 相關詞彙：{synonyms}"
            tag_names.append(tag_name)
            tag_texts.append(combined)

        tag_embeddings = self._embed_texts(tag_texts)
        return tag_names, tag_embeddings

    # ──────────────────────────────────────────
    # 文章 First Chunk
    # ──────────────────────────────────────────

    def _get_first_chunk(self, article: dict) -> str:
        """
        取得文章的 first chunk 用來向量化。

        策略：標題 + 內文前 CHUNK_SIZE 字元。
        標題在前確保主題資訊一定被包含。

        Args:
            article: 包含 title / content 的文章 dict

        Returns:
            合併後截斷的文字字串
        """
        title   = article.get("title")   or ""
        content = article.get("content") or ""
        full    = f"{title}\n{content}".strip()
        return full[:CHUNK_SIZE] if full else title

    # ──────────────────────────────────────────
    # Adaptive-K 算法
    # ──────────────────────────────────────────

    def _adaptive_k(self, scores: np.ndarray) -> int:
        """
        Adaptive-K：找相似度分數的最大語意斷層，動態決定 TopK。

        參考：arXiv:2506.08479（Taguchi et al., EMNLP 2025）

        演算法：
          1. 接收已降序排列的 TopN 分數
          2. 計算相鄰分數的差值（gap）
          3. 找最大 gap 的位置，該位置之前即為 TopK

        Args:
            scores: TopN 個候選標籤的相似度分數（已降序排列）

        Returns:
            動態決定的 K 值（至少 1）
        """
        if len(scores) <= 1:
            return 1

        gaps        = scores[:-1] - scores[1:]
        max_gap_idx = int(np.argmax(gaps))
        k           = max_gap_idx + 1
        return max(1, k)

    # ──────────────────────────────────────────
    # 核心打標邏輯
    # ──────────────────────────────────────────

    def _match_tags(self, article: dict) -> tuple[list[str], dict[str, float]]:
        """
        對單篇文章執行語意打標。

        流程：
          1. 取文章 first chunk 向量化
          2. 計算和所有標籤的 cosine similarity
          3. 取 TopN 候選
          4. Adaptive-K 找斷層，決定 TopK
          5. 回傳 min(TopK, MAX_TAGS) 個標籤

        Args:
            article: 文章 dict（含 title / content）

        Returns:
            (matched_tags, score_dict)
            matched_tags: 命中的標籤名稱列表
            score_dict:   {標籤名稱: 相似度分數}，用於 log
        """
        chunk       = self._get_first_chunk(article)
        article_emb = self._embed_texts([chunk])[0].reshape(1, -1)

        similarities = cosine_similarity(article_emb, self.tag_embeddings)[0]

        top_n   = min(TOP_N, len(self.tag_names))
        top_idx = np.argsort(similarities)[::-1][:top_n]
        top_sim = similarities[top_idx]

        k = self._adaptive_k(top_sim)
        k = min(k, MAX_TAGS)

        selected_idx = top_idx[:k]
        matched_tags = [self.tag_names[i] for i in selected_idx]
        score_dict   = {
            self.tag_names[i]: round(float(similarities[i]), 4)
            for i in selected_idx
        }

        return matched_tags, score_dict

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
    # 資料庫寫回
    # ──────────────────────────────────────────

    def _update_tags_batch(self, updates: list[dict]) -> int:
        """
        批次更新文章的 tags 欄位。

        Args:
            updates: [{"id": int, "tags": [...]}] 列表

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
                    "id":   item["id"],
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
        tag_counter: dict[str, int],
        elapsed: float,
    ) -> None:
        """
        輸出本次打標結果統計至 log。

        Args:
            total:       處理總篇數
            tag_counter: {標籤名稱: 命中篇數} 統計字典
            elapsed:     總耗時（秒）
        """
        logger.info("=" * 60)
        logger.info("【打標完成統計】")
        logger.info("  總處理篇數：%d", total)
        logger.info("  耗時：%.2f 秒", elapsed)
        logger.info("  各標籤命中數：")
        for tag, count in sorted(tag_counter.items(), key=lambda x: -x[1]):
            logger.info("    %-12s : %d 篇", tag, count)
        logger.info("=" * 60)

    # ──────────────────────────────────────────
    # 主執行流程
    # ──────────────────────────────────────────

    def run(self) -> None:
        """
        執行完整語意打標流程。

        步驟：
          1. 撈取待打標文章
          2. 逐篇向量化並計算相似度
          3. Adaptive-K 決定標籤數量
          4. 批次寫回 tags 至資料庫
          5. 輸出統計 log
        """
        start_time = datetime.now(timezone.utc)
        logger.info("====== 語意打標任務開始 %s ======", start_time.isoformat())

        articles = self._fetch_pending_articles()
        if not articles:
            logger.info("無待打標文章，任務結束")
            return

        total       = len(articles)
        tag_counter = {tag: 0 for tag in self.tag_names}
        updates     = []

        for i, article in enumerate(articles):
            article_id = article["id"]
            title      = (article.get("title") or "（無標題）")[:60]

            matched_tags, score_dict = self._match_tags(article)

            for tag in matched_tags:
                tag_counter[tag] = tag_counter.get(tag, 0) + 1

            scores_str = " | ".join(
                f"{t}:{s:.4f}" for t, s in score_dict.items()
            )
            logger.info(
                "[%d/%d] %s\n  標籤：%s\n  分數：%s",
                i + 1, total, title,
                "、".join(matched_tags),
                scores_str,
            )

            updates.append({"id": article_id, "tags": matched_tags})

            if len(updates) >= BATCH_SIZE:
                written = self._update_tags_batch(updates)
                logger.info("批次寫入 %d 筆至資料庫", written)
                updates.clear()

        if updates:
            written = self._update_tags_batch(updates)
            logger.info("最終批次寫入 %d 筆至資料庫", written)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        self._log_summary(total, tag_counter, elapsed)

    def close(self) -> None:
        """關閉資料庫連線。"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("資料庫連線已關閉")


# ────────────────────────────────────────────────
# 獨立執行入口
# ────────────────────────────────────────────────
if __name__ == "__main__":
    tagger = ArticleTagger()
    try:
        tagger.run()
    finally:
        tagger.close()