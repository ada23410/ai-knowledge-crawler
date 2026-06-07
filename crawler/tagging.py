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

限流機制：
  - 每篇文章間隔 ARTICLE_INTERVAL 秒（預設 1.0 秒）
  - 每批次完成後休息 BATCH_INTERVAL 秒（預設 5.0 秒）
  - 遇到 429 / quota 錯誤時指數退避重試（60s → 120s → 240s，最多 3 次）

優點：
  - 不需要手動維護關鍵字清單
  - 同義詞和語意變體自動涵蓋
  - 未分類比例趨近於 0
  - 每個標籤都有相似度分數，可解釋性高
"""

import json
import logging
import os
import time
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
# 適用對象：AI 數據學院內勤同仁
# 核心任務：員工 AI 教育訓練 × 追蹤特定主題趨勢
# ────────────────────────────────────────────────
TAG_DEFINITIONS: dict[str, dict] = {

    "AI 工具應用": {
        "description": (
            "市面上可直接使用的 AI 工具、平台與 SaaS 服務評測，"
            "涵蓋生產力工具、AI 助理、企業導入方案，"
            "協助內勤同仁評估哪些工具值得學習或推廣。"
        ),
        "synonyms": [
            # English
            "AI tool", "AI platform", "SaaS", "productivity tool",
            "AI assistant", "Copilot", "ChatGPT", "Gemini", "Claude",
            "no-code", "low-code", "tool evaluation", "review",
            "enterprise AI", "AI adoption", "workflow tool",
            "AI application", "software", "plugin", "integration",
            # 繁體中文
            "AI 工具", "工具評測", "平台", "生產力工具", "助理",
            "企業導入", "工具推薦", "工具介紹", "軟體", "外掛",
            "應用程式", "試用", "評估", "導入工具",
        ],
    },

    "流程自動化": {
        "description": (
            "RPA、文件處理、表單自動化、AI Agent 在行政流程的落地應用，"
            "聚焦內勤單位可直接參考的自動化場景與實作案例，"
            "包含人機協作、作業流程再造等主題。"
        ),
        "synonyms": [
            # English
            "RPA", "robotic process automation", "workflow automation",
            "process automation", "AI agent", "agentic", "document processing",
            "OCR", "intelligent document processing", "IDP",
            "form automation", "back office", "operation automation",
            "business process", "human in the loop", "task automation",
            "automation tool", "digital worker", "bot",
            # 繁體中文
            "流程自動化", "機器人流程自動化", "自動化", "作業自動化",
            "文件處理", "表單自動化", "行政自動化", "智能文件",
            "人機協作", "流程再造", "後台自動化", "作業流程",
            "自動填表", "批次處理", "無人化",
        ],
    },

    "核保理賠 AI": {
        "description": (
            "AI 輔助核保決策、理賠審查、醫療文件辨識、詐欺偵測與客戶風險評估，"
            "聚焦保險業 AI 落地應用場景，"
            "協助核保與理賠同仁掌握最新技術趨勢與業界案例。"
        ),
        "synonyms": [
            # English
            "underwriting", "claims processing", "fraud detection",
            "insurance AI", "InsurTech", "actuarial", "loss ratio",
            "claims automation", "injury assessment",  # ← 移除 STP, straight through processing, risk assessment
            "subrogation", "reinsurance",
            "insurance underwriting", "claims adjudication",
            # 繁體中文
            "核保", "理賠", "保險 AI", "詐欺偵測",
            "精算", "理賠自動化", "核保決策",
            "代位求償", "再保險",
            "核保輔助", "理賠審核", "保險科技",
            # 移除：傷害評估
        ],
    },

    "風控法遵 AI": {
        "description": (
            "AI 應用於金融風險控管、反洗錢（AML）、客戶身分驗證（KYC）、"
            "信用評分與監理科技（RegTech），"
            "協助風控與法遵同仁追蹤 AI 在合規領域的最新發展。"
        ),
        "synonyms": [
            # English
            "AML", "anti-money laundering", "KYC", "know your customer",
            "credit scoring", "risk model", "RegTech", "regulatory technology",
            "fraud prevention", "financial crime", "sanctions screening",
            "transaction monitoring", "model risk", "stress testing",
            "credit risk", "market risk", "operational risk",
            "risk management", "compliance AI", "suspicious activity",
            # 繁體中文
            "洗錢防制", "客戶身分驗證", "信用評分", "風控模型",
            "監理科技", "金融犯罪", "交易監控", "制裁名單篩查",
            "模型風險", "壓力測試", "信用風險", "市場風險",
            "作業風險", "風險管理", "法遵 AI", "可疑交易",
            "風控", "法遵", "合規 AI",
        ],
    },

    "監理政策": {
        "description": (
            "主管機關 AI 相關公告、金融監理政策、個資保護法規、"
            "AI 治理框架與國際監管動態，"
            "協助同仁掌握法規變化對業務的影響。"
        ),
        "synonyms": [
            # English
            "AI Act", "regulation", "policy", "legislation", "guideline",
            "white paper", "regulatory framework", "executive order",
            "standard", "principle", "governance framework", "legal",
            "GDPR", "data protection", "privacy law", "FSA", "FSB",
            "IAIS", "Basel", "supervisory", "enforcement",
            # 繁體中文
            "金管會", "監理政策", "法規", "政策", "公告", "指引",
            "白皮書", "準則", "條例", "草案", "個資法", "隱私",
            "治理框架", "監管", "行政命令", "法令", "辦法",
            "監理規範", "主管機關", "法規遵循",
        ],
    },

    "AI 教育訓練": {
        "description": (
            "員工 AI 素養培訓、課程設計、學習資源、教學工具與企業內訓案例，"
            "包含 AI 技能地圖、培訓計畫規劃、學習平台評比，"
            "為 AI 數據學院核心關注主題。"
        ),
        "synonyms": [
            # English
            "AI literacy", "AI training", "upskilling", "reskilling",
            "workforce training", "learning and development", "L&D",
            "corporate training", "employee training", "talent development",
            "AI skill", "training program", "competency framework",
            "skill development", "certification",
            # 繁體中文
            "AI 素養", "員工訓練", "教育訓練", "培訓",
            "技能提升", "再培訓", "內訓",
            "技能地圖", "認證", "訓練計畫", "人才培育",
            "學習發展",
        ],
    },

    "產業趨勢": {
        "description": (
            "金融保險業 AI 導入現況、競品動態、市場研究報告與產業前瞻，"
            "提供內勤同仁掌握整體市場脈動，"
            "作為提交主管週報或教育訓練選題的參考素材。"
        ),
        "synonyms": [
            # English
            "industry trend", "market report", "market research",
            "competitive landscape", "benchmark report", "industry insight",
            "financial services", "banking AI", "insurance industry",
            "digital transformation", "AI adoption rate", "survey",
            "forecast", "outlook", "annual report", "analyst report",
            "Gartner", "McKinsey", "Deloitte", "industry news",
            # 繁體中文
            "產業趨勢", "市場報告", "市場研究", "競品動態",
            "金融業", "保險業", "數位轉型", "導入現況", "產業前瞻",
            "趨勢報告", "年度報告", "調查報告", "分析報告",
            "市場動態", "業界動態", "產業洞察", "產業分析",
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

# ────────────────────────────────────────────────
# 限流設定
# ────────────────────────────────────────────────
ARTICLE_INTERVAL = 1.0    # 每篇文章間隔秒數
BATCH_INTERVAL   = 5.0    # 每批次完成後休息秒數
RETRY_MAX        = 3      # 429 最大重試次數
RETRY_BASE_WAIT  = 60     # 第一次重試等待秒數（指數退避：60 → 120 → 240）


class ArticleTagger:
    """
    語意向量自動標籤模組主類別。

    使用 gemini-embedding-001 將文章和標籤定義向量化，
    透過 cosine similarity + Adaptive-K 動態決定每篇文章的標籤。
    內建 429 指數退避重試與每篇 / 每批限流機制。

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
    # Embedding 工具函式（含限流與網路錯誤重試）
    # ──────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        批次向量化文字清單，遇到可重試錯誤時自動指數退避重試。

        可重試錯誤類型：
          - 429 / quota / resource exhausted：API 限流
          - 503 / socket closed / unavailable：網路連線中斷

        重試策略：
          - 第 1 次重試：等待 60 秒
          - 第 2 次重試：等待 120 秒
          - 第 3 次重試：等待 240 秒
          - 超過 3 次則拋出例外

        Args:
            texts: 要向量化的文字列表

        Returns:
            np.ndarray，shape = (len(texts), 3072)

        Raises:
            Exception: 不可重試錯誤，或重試次數耗盡時拋出
        """
        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch  = texts[i:i + BATCH_SIZE]
            inputs = [
                TextEmbeddingInput(text=t, task_type="SEMANTIC_SIMILARITY")
                for t in batch
            ]

            for attempt in range(RETRY_MAX):
                try:
                    response = self.model.get_embeddings(inputs)
                    for emb in response:
                        all_embeddings.append(emb.values)
                    break  # 成功，跳出重試迴圈

                except Exception as e:
                    err_str = str(e).lower()
                    is_retryable = any(keyword in err_str for keyword in [
                        "429", "quota", "resource exhausted",  # 限流
                        "503", "socket closed", "unavailable",  # 網路中斷
                        "deadline exceeded", "timeout",         # 逾時
                    ])

                    if is_retryable:
                        wait = RETRY_BASE_WAIT * (2 ** attempt)  # 60 / 120 / 240
                        logger.warning(
                            "⚠️  可重試錯誤，第 %d/%d 次重試，等待 %d 秒... (錯誤：%s)",
                            attempt + 1, RETRY_MAX, wait, e,
                        )
                        time.sleep(wait)
                        if attempt == RETRY_MAX - 1:
                            logger.error("❌ 重試次數耗盡，拋出例外")
                            raise
                    else:
                        raise  # 不可重試錯誤直接拋出

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

        涵蓋範圍：rss / arxiv / html 全部來源。
        governance_status 邏輯：
          - approved  → 打標 ✅
          - rejected  → 跳過 ❌（未達門檻，不進入知識庫）
          - duplicate → 跳過 ❌（重複內容不打標）

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
            logger.info("    %-15s : %d 篇", tag, count)
        logger.info("=" * 60)

    # ──────────────────────────────────────────
    # 主執行流程
    # ──────────────────────────────────────────

    def run(self) -> None:
        """
        執行完整語意打標流程（含限流）。

        步驟：
          1. 撈取待打標文章
          2. 逐篇向量化並計算相似度（每篇間隔 ARTICLE_INTERVAL 秒）
          3. Adaptive-K 決定標籤數量
          4. 每 BATCH_SIZE 篇批次寫回資料庫，批次間休息 BATCH_INTERVAL 秒
          5. 輸出統計 log
        """
        start_time = datetime.now(timezone.utc)
        logger.info("====== 語意打標任務開始 %s ======", start_time.isoformat())
        logger.info("限流設定：每篇間隔 %.1fs｜批次間隔 %.1fs｜429 退避 %ds/%ds/%ds",
                    ARTICLE_INTERVAL, BATCH_INTERVAL,
                    RETRY_BASE_WAIT, RETRY_BASE_WAIT * 2, RETRY_BASE_WAIT * 4)

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

            # 每篇間隔，避免 API 過載
            time.sleep(ARTICLE_INTERVAL)

            if len(updates) >= BATCH_SIZE:
                written = self._update_tags_batch(updates)
                logger.info("批次寫入 %d 筆至資料庫", written)
                updates.clear()
                logger.info("批次休息 %.1f 秒...", BATCH_INTERVAL)
                time.sleep(BATCH_INTERVAL)

        # 寫入最後一批
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