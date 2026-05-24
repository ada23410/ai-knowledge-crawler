"""
AI 知識庫 蒐集層主程式
功能：定時從 RSS Feed 和 arXiv 抓取 AI 相關文章並存入 PostgreSQL 資料庫
"""

# 引入套件
import os                    # 讀取系統環境變數
import logging               # 記錄 log
import schedule              # 排程工具
import time                  # 時間相關操作
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv  # 讀取 .env 檔案

import feedparser            # 解析 RSS XML
import requests              # 發送 HTTP 請求
import trafilatura           # 從網頁萃取純文字
import arxiv                 # arXiv 官方 SDK

import psycopg2              # 連接 PostgreSQL
from psycopg2.extras import execute_values  # 批次寫入資料用
from tagging import ArticleTagger

# 讀取 .env 設定
load_dotenv()

# 設定 Log 系統
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/crawler.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# 1. 資料庫連線管理
def get_db_connection():
    """建立 PostgreSQL 資料庫連線"""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "ai_knowledge"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )
    return conn


def article_exists(conn, url: str) -> bool:
    """檢查這篇文章是否已經存在資料庫（用 URL 判斷）"""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM articles WHERE url = %s", (url,))
        return cur.fetchone() is not None


def save_article(conn, article_data: dict) -> bool:
    """將一篇文章存入資料庫"""
    if article_exists(conn, article_data["url"]):
        return False

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO articles (
                source_id, title, url, content, summary,
                published_at, language, content_type, source_tier, governance_status
            ) VALUES (
                %(source_id)s, %(title)s, %(url)s, %(content)s, %(summary)s,
                %(published_at)s, %(language)s, %(content_type)s, %(source_tier)s, %(governance_status)s
            )
        """, article_data)
        conn.commit()
    return True


def save_fetch_log(conn, log_data: dict):
    """記錄這次抓取的執行結果到 fetch_logs 表"""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO fetch_logs (
                source_id, source_name, status,
                articles_found, articles_new, error_message
            ) VALUES (
                %(source_id)s, %(source_name)s, %(status)s,
                %(articles_found)s, %(articles_new)s, %(error_message)s
            )
        """, log_data)
        conn.commit()


def load_sources(conn) -> list:
    """從資料庫讀取所有啟用中的來源"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, name, url, language, type, source_tier, resources_type, fetcher
            FROM sources
            WHERE is_active = TRUE
            ORDER BY source_tier, name
        """)
        rows = cur.fetchall()
        columns = ["id", "name", "url", "language", "type", "source_tier", "resources_type", "fetcher"]
        return [dict(zip(columns, row)) for row in rows]

# 2. RSS 蒐集器
def fetch_rss(source: dict, conn) -> dict:
    """
    抓取一個 RSS 來源的最新文章

    參數：
      source = 來源資料（dict），含 id / name / url / language 等
      conn   = 資料庫連線
    回傳：
      包含執行結果的 dict（用來記 log）
    """
    logger.info(f"開始抓取 RSS：{source['name']}")
    
    days_limit = int(os.getenv("FETCH_DAYS_LIMIT", 7))
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_limit)
    
    articles_found = 0
    articles_new = 0

    try:
        feed = feedparser.parse(source["url"])
        
        if feed.bozo and not feed.entries:
            raise Exception(f"RSS 解析失敗：{feed.bozo_exception}")

        logger.info(f"   找到 {len(feed.entries)} 篇文章")
        articles_found = len(feed.entries)

        for entry in feed.entries:
            url = entry.get("link", "").strip()
            if not url:
                continue

            published_at = parse_date(entry)
            
            if published_at and published_at < cutoff_date:
                continue

            title = entry.get("title", "（無標題）").strip()
            content = extract_content(url, entry)

            article_data = {
                "source_id":    source["id"],
                "title":        title,
                "url":          url,
                "content":      content,
                "summary":      content[:200] if content else None,
                "published_at": published_at,
                "language":     source["language"],
                "content_type": "rss",
                "source_tier":  source["source_tier"],
                "governance_status": "pending",
            }
            
            if save_article(conn, article_data):
                articles_new += 1
                logger.info(f"新文章：{title[:50]}...")

        return {
            "source_id":      source["id"],
            "source_name":    source["name"],
            "status":         "success",
            "articles_found": articles_found,
            "articles_new":   articles_new,
            "error_message":  None,
        }

    except Exception as e:
        logger.error(f"抓取失敗 {source['name']}：{e}")
        return {
            "source_id":      source["id"],
            "source_name":    source["name"],
            "status":         "failed",
            "articles_found": articles_found,
            "articles_new":   articles_new,
            "error_message":  str(e),
        }


def extract_content(url: str, entry) -> str:
    """
    從文章 URL 萃取純文字內文

    策略：
    1. 先用 trafilatura 從網頁直接抓（最完整）
    2. 如果失敗，就用 RSS entry 裡的摘要（備案）
    3. 如果還是沒有，回傳空字串
    """
    try:
        downloaded = trafilatura.fetch_url(url, timeout=15)
        if downloaded:
            content = trafilatura.extract(downloaded)
            if content and len(content) > 100:
                return content
    except Exception:
        pass

    summary = entry.get("summary", "") or entry.get("description", "")
    import re
    clean = re.sub(r"<[^>]+>", "", summary).strip()
    return clean if clean else ""


def parse_date(entry) -> datetime:
    """解析 RSS entry 的發布時間"""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            import calendar
            ts = calendar.timegm(entry.published_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)

# 3. arXiv 蒐集器
def fetch_arxiv(source: dict, conn) -> dict:
    """抓取 arXiv 最新 AI 論文"""
    logger.info(f"開始抓取 arXiv：{source['name']}")
    
    max_results = int(os.getenv("ARXIV_MAX_RESULTS", 50))
    articles_new = 0

    try:
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3,
            num_retries=3,
        )
        
        search = arxiv.Search(
            query="cat:cs.AI",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = list(client.results(search))
        logger.info(f"   找到 {len(results)} 篇論文")

        for paper in results:
            url = paper.entry_id
            title = paper.title.replace("\n", " ").strip()
            content = paper.summary.replace("\n", " ").strip()
            
            authors = ", ".join(str(a) for a in paper.authors[:5])
            if len(paper.authors) > 5:
                authors += " et al."
            
            full_content = f"[Authors] {authors}\n\n{content}"

            article_data = {
                "source_id":    source["id"],
                "title":        title,
                "url":          url,
                "content":      full_content,
                "summary":      content[:200],
                "published_at": paper.published,
                "language":     "en",
                "content_type": "arxiv",
                "source_tier":  "A",
                "governance_status": "pending",
            }
            
            if save_article(conn, article_data):
                articles_new += 1
                logger.info(f"新論文：{title[:50]}...")

        return {
            "source_id":      source["id"],
            "source_name":    source["name"],
            "status":         "success",
            "articles_found": len(results),
            "articles_new":   articles_new,
            "error_message":  None,
        }

    except Exception as e:
        logger.error(f"arXiv 抓取失敗：{e}")
        return {
            "source_id":      source["id"],
            "source_name":    source["name"],
            "status":         "failed",
            "articles_found": 0,
            "articles_new":   articles_new,
            "error_message":  str(e),
        }

# 4. 主排程任務
def run_daily_fetch():
    """
    每日定時執行的主要任務

    執行順序：
    1. 連線資料庫
    2. 讀取所有啟用的來源
    3. 依來源類型選擇對應的蒐集器
    4. 記錄每個來源的執行結果
    5. 執行自動評分與打標
    6. 輸出今日統計摘要
    """
    conn = None
    total_new    = 0   # ✅ 在使用前先宣告並初始化
    total_failed = 0

    try:
        conn = get_db_connection()
        logger.info("資料庫連線成功")

        sources = load_sources(conn)
        logger.info(f"本次要抓取的來源數：{len(sources)}")

        # ── 逐一處理每個來源 ──
        for source in sources:
            fetcher = source.get("fetcher", "rss")

            if fetcher == "arxiv":
                log_data = fetch_arxiv(source, conn)
            elif fetcher == "skip":
                logger.info(f"跳過（skip）：{source['name']}")
                continue
            else:
                log_data = fetch_rss(source, conn)

            save_fetch_log(conn, log_data)

            if log_data["status"] == "success":
                total_new += log_data["articles_new"]
            else:
                total_failed += 1

        # ── 輸出今日爬蟲摘要 ──
        logger.info("=" * 50)
        logger.info("今日蒐集完成")
        logger.info(f"   新增文章：{total_new} 篇")
        logger.info(f"   失敗來源：{total_failed} 個")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"主任務發生嚴重錯誤：{e}")

    finally:
        if conn:
            conn.close()
            logger.info("資料庫連線已關閉")

    # ── 執行自動評分（pending 文章）──
    logger.info("開始執行自動評分...")
    try:
        from scoring import run_scoring
        run_scoring()
    except Exception as e:
        logger.error(f"自動評分失敗：{e}")
    logger.info("自動評分完成")

    # ── 執行自動打標（approved 文章）──
    logger.info("開始執行自動打標...")
    tagger = ArticleTagger()
    try:
        tagger.run()
    except Exception as e:
        logger.error(f"自動打標失敗：{e}")
    finally:
        tagger.close()
    logger.info("自動打標完成")


def run_biweekly_pipeline():
    """
    每兩週執行一次的完整治理流程。

    執行順序：
      1. dedup（語意去重）→ 標記重複文章
      2. scoring（內容評分）→ approved / rejected
      3. tagging（自動打標）→ approved 文章打標
      4. sharepoint_trigger → 寄送 JSON 附件觸發 Power Automate

    排程：每兩週一 09:00
    """
    logger.info("=" * 50)
    logger.info("兩週治理流程啟動")
    logger.info("=" * 50)

    # ── Step 1：語意去重 ──
    logger.info("Step 1／4：開始語意去重...")
    try:
        from dedup import run_deduplication
        run_deduplication()
    except Exception as e:
        logger.error(f"去重失敗：{e}")
    logger.info("去重完成")

    # ── Step 2：內容評分 ──
    logger.info("Step 2／4：開始內容評分...")
    try:
        from scoring import run_scoring
        run_scoring()
    except Exception as e:
        logger.error(f"評分失敗：{e}")
    logger.info("評分完成")

    # ── Step 3：自動打標 ──
    logger.info("Step 3／4：開始自動打標...")
    tagger = ArticleTagger()
    try:
        tagger.run()
    except Exception as e:
        logger.error(f"打標失敗：{e}")
    finally:
        tagger.close()
    logger.info("打標完成")

    # ── Step 4：SharePoint 同步 ──
    logger.info("Step 4／4：開始同步至 SharePoint...")
    try:
        from sharepoint_trigger import run_sharepoint_sync
        run_sharepoint_sync()
    except Exception as e:
        logger.error(f"SharePoint 同步失敗：{e}")
    logger.info("SharePoint 同步完成")

    logger.info("=" * 50)
    logger.info("兩週治理流程完成")
    logger.info("=" * 50)

# 5. 初始化資料庫（第一次執行時）
def init_db():
    """初始化資料庫：建立資料表 + 匯入來源清單"""
    logger.info("初始化資料庫...")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            with open("/db/init.sql", "r", encoding="utf-8") as f:
                sql = f.read()
            cur.execute(sql)
            conn.commit()
            logger.info("資料表建立完成")

        seed_sources(conn)

    finally:
        conn.close()


def seed_sources(conn):
    """從 sources.csv 讀取來源清單並寫入資料庫"""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sources")
        count = cur.fetchone()[0]
        if count > 0:
            logger.info(f"   sources 表已有 {count} 筆資料，略過匯入")
            return

    csv_path = "/db/sources.csv"

    if not os.path.exists(csv_path):
        logger.error(f"找不到 sources.csv：{csv_path}")
        return

    import csv
    sources_data = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_active = row["is_active"].strip().lower() == "true"
            sources_data.append((
                row["resources_type"],
                row["name"],
                row["url"],
                row["language"],
                row["type"],
                row["description"],
                row["source_tier"],
                row["fetcher"],
                is_active,
            ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO sources
                (resources_type, name, url, language, type, description,
                 source_tier, fetcher, is_active)
            VALUES %s
            ON CONFLICT (url) DO NOTHING
        """, sources_data)
        conn.commit()

    logger.info(f"已從 sources.csv 匯入 {len(sources_data)} 筆來源")

# 6. 程式入口點
if __name__ == "__main__":
    import sys
    
    os.makedirs("logs", exist_ok=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "schedule"

    if mode == "init":
        init_db()
        logger.info("初始化完成！現在可以執行 python main.py now 測試抓取")

    elif mode == "now":
        logger.info("立即執行模式（測試用）")
        run_daily_fetch()

    else:
        fetch_hour   = int(os.getenv("FETCH_HOUR", 8))
        fetch_minute = int(os.getenv("FETCH_MINUTE", 0))
        run_time     = f"{fetch_hour:02d}:{fetch_minute:02d}"

        biweekly_time = "09:00"

        logger.info(f"排程模式啟動")
        logger.info(f"  每天 {run_time}：爬蟲抓取")
        logger.info(f"  每兩週一 {biweekly_time}：去重 → 評分 → 打標")

        # 每天爬蟲
        schedule.every().day.at(run_time).do(run_daily_fetch)

        # 每兩週一 09:00 執行完整治理流程
        # schedule 套件無內建「每兩週」，用計數器實作
        biweekly_counter = {"count": 0}

        def biweekly_monday_job():
            biweekly_counter["count"] += 1
            if biweekly_counter["count"] % 2 == 0:
                run_biweekly_pipeline()
            else:
                logger.info("本週為非治理週，跳過 dedup/scoring/tagging")

        schedule.every().monday.at(biweekly_time).do(biweekly_monday_job)

        while True:
            schedule.run_pending()
            time.sleep(60)