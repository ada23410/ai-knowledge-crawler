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
# 這行讓 Python 讀取專案根目錄的 .env 檔案
load_dotenv()

# 設定 Log 系統
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    # asctime = 時間，levelname = 等級，message = 訊息
    handlers=[
        logging.StreamHandler(),                           # 輸出到終端機
        logging.FileHandler("logs/crawler.log", encoding="utf-8")  # 同時存成檔案
    ]
)
logger = logging.getLogger(__name__)

# 1. 資料庫連線管理
def get_db_connection():
    """
    建立 PostgreSQL 資料庫連線
    """
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "ai_knowledge"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )
    return conn


def article_exists(conn, url: str) -> bool:
    """
    檢查這篇文章是否已經存在資料庫（用 URL 判斷）
    
    用途：去重。同一篇文章不要存兩次。
    """
    with conn.cursor() as cur:
        # %s 是 psycopg2 的參數佔位符，防止 SQL injection
        # 就像你前端用 template literal 但更安全
        cur.execute("SELECT 1 FROM articles WHERE url = %s", (url,))
        return cur.fetchone() is not None


def save_article(conn, article_data: dict) -> bool:
    """
    將一篇文章存入資料庫
    """
    # 先檢查是否重複
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
        # commit = 確認寫入（不 commit 的話資料不會真的存進去）
        conn.commit()
    return True


def save_fetch_log(conn, log_data: dict):
    """
    記錄這次抓取的執行結果到 fetch_logs 表
    """
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
    """
    從資料庫讀取所有啟用中的來源
    """
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
    
    # 設定幾天內的文章才要抓
    days_limit = int(os.getenv("FETCH_DAYS_LIMIT", 7))
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_limit)
    
    articles_found = 0
    articles_new = 0

    try:
        # ── 步驟一：用 feedparser 解析 RSS ──
        # feedparser 會自動發 HTTP 請求並解析 XML
        # feed.entries 是文章清單（list）
        feed = feedparser.parse(source["url"])
        
        # 檢查是否有錯誤
        # bozo = feedparser 用來標記「這個 feed 有問題」的旗標
        if feed.bozo and not feed.entries:
            raise Exception(f"RSS 解析失敗：{feed.bozo_exception}")

        logger.info(f"   找到 {len(feed.entries)} 篇文章")
        articles_found = len(feed.entries)

        # ── 步驟二：逐篇處理 ──
        # feed.entries 裡每個 entry 就是一篇文章
        for entry in feed.entries:
            
            # 取得文章網址
            # .get() = 取值，如果不存在就回傳 None（不會報錯）
            # 等同 JS 的 entry?.link
            url = entry.get("link", "").strip()
            if not url:
                continue  # 沒有網址就跳過（continue ≈ JS 的 continue）

            # 取得發布時間
            published_at = parse_date(entry)
            
            # 只抓 N 天內的文章
            if published_at and published_at < cutoff_date:
                continue

            # 取得文章標題
            title = entry.get("title", "（無標題）").strip()
            
            # 步驟三：取得文章內文 ──
            content = extract_content(url, entry)

            # 步驟四：存入資料庫 ──
            article_data = {
                "source_id":    source["id"],
                "title":        title,
                "url":          url,
                "content":      content,
                "summary":      content[:200] if content else None,  # 前 200 字當摘要
                "published_at": published_at,
                "language":     source["language"],
                "content_type": "rss",
                "source_tier":  source["source_tier"],
                "governance_status": "pending",
            }
            
            # save_article 回傳 True = 新文章存入成功
            if save_article(conn, article_data):
                articles_new += 1
                logger.info(f"新文章：{title[:50]}...")

        # 回傳成功的 log 資料
        return {
            "source_id":      source["id"],
            "source_name":    source["name"],
            "status":         "success",
            "articles_found": articles_found,
            "articles_new":   articles_new,
            "error_message":  None,
        }

    except Exception as e:
        # 發生任何錯誤，記錄下來但不讓整個程式崩潰
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
    
    timeout = 請求等待秒數，超過就放棄
    """
    try:
        downloaded = trafilatura.fetch_url(url, timeout=15)
        if downloaded:
            content = trafilatura.extract(downloaded)
            if content and len(content) > 100:  # 太短的不算
                return content
    except Exception:
        pass  # 失敗就靜默跳過，試備案

    # 備案：用 RSS 裡的摘要
    summary = entry.get("summary", "") or entry.get("description", "")
    # 清掉 HTML 標籤（簡單版）
    import re
    clean = re.sub(r"<[^>]+>", "", summary).strip()
    return clean if clean else ""


def parse_date(entry) -> datetime:
    """
    解析 RSS entry 的發布時間
    
    RSS 的日期格式五花八門，feedparser 幫我們標準化了
    published_parsed 是 time.struct_time 格式，需要轉換
    """
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            # struct_time → timestamp → datetime
            import calendar
            ts = calendar.timegm(entry.published_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)  # 解析失敗就用現在時間

# 3. arXiv 蒐集器
def fetch_arxiv(source: dict, conn) -> dict:
    """
    抓取 arXiv 最新 AI 論文
    """
    logger.info(f"開始抓取 arXiv：{source['name']}")
    
    max_results = int(os.getenv("ARXIV_MAX_RESULTS", 50))
    articles_new = 0

    try:
        # 建立 arXiv 查詢
        # 這就像你在 arXiv 網站搜尋 "cs.AI" 分類的最新論文
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3,   # 每次請求間隔 3 秒，避免被封鎖
            num_retries=3,     # 失敗最多重試 3 次
        )
        
        search = arxiv.Search(
            query="cat:cs.AI",          # 搜尋 CS.AI 分類
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,  # 最新優先
        )

        results = list(client.results(search))
        logger.info(f"   找到 {len(results)} 篇論文")

        # ── 逐篇處理 ──
        for paper in results:
            url = paper.entry_id  # arXiv 論文的唯一 URL
            title = paper.title.replace("\n", " ").strip()
            
            # arXiv 論文的 abstract（摘要）就是最好的內容
            content = paper.summary.replace("\n", " ").strip()
            
            # 作者清單 → 轉成字串
            authors = ", ".join(str(a) for a in paper.authors[:5])
            if len(paper.authors) > 5:
                authors += " et al."
            
            # 把作者資訊加到內文前面
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
                "source_tier":  "A",  # arXiv 屬於 A 類來源
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
    5. 輸出今日統計摘要
    """
    # ── 執行自動打標（approved 文章）──
    logger.info("開始執行自動打標...")
    tagger = ArticleTagger()
    try:
        tagger.run()
    finally:
        tagger.close()
    logger.info("自動打標完成")

    # 輸出今日摘要
    logger.info("=" * 50)
    logger.info(f"今日蒐集完成")
    logger.info(f"   新增文章：{total_new} 篇")
    logger.info(f"   失敗來源：{total_failed} 個")
    logger.info("=" * 50)

    conn = None
    total_new = 0
    total_failed = 0

    try:
        # 建立資料庫連線
        conn = get_db_connection()
        logger.info("資料庫連線成功")

        # 從資料庫讀取來源清單
        sources = load_sources(conn)
        logger.info(f"📋 本次要抓取的來源數：{len(sources)}")

        # ── 逐一處理每個來源 ──
        for source in sources:
            
            # 根據 fetcher 欄位決定用哪個蒐集器
            # 這個值來自 sources.csv，比靠名稱判斷更可靠
            fetcher = source.get("fetcher", "rss")

            if fetcher == "arxiv":
                log_data = fetch_arxiv(source, conn)
            elif fetcher == "skip":
                logger.info(f"⏭跳過（skip）：{source['name']}")
                continue
            else:
                log_data = fetch_rss(source, conn)

            # 記錄執行結果到資料庫
            save_fetch_log(conn, log_data)

            # 累計統計
            if log_data["status"] == "success":
                total_new += log_data["articles_new"]
            else:
                total_failed += 1

        # ── 輸出今日摘要 ──
        logger.info("=" * 50)
        logger.info(f"今日蒐集完成")
        logger.info(f"   新增文章：{total_new} 篇")
        logger.info(f"   失敗來源：{total_failed} 個")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"主任務發生嚴重錯誤：{e}")

    finally:
        # finally 區塊無論成功失敗都會執行
        # 確保資料庫連線一定會被關閉（釋放資源）
        if conn:
            conn.close()
            logger.info("資料庫連線已關閉")

# 5. 初始化資料庫（第一次執行時）
def init_db():
    """
    初始化資料庫：建立資料表 + 匯入來源清單
    
    只需要第一次執行，之後資料表已存在就不會重建
    （因為 SQL 裡用了 CREATE TABLE IF NOT EXISTS）
    """
    logger.info("🔧 初始化資料庫...")
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # 讀取並執行 SQL 檔案
            # open() ≈ JS 的 fs.readFileSync()
            with open("/db/init.sql", "r", encoding="utf-8") as f:
                sql = f.read()
            cur.execute(sql)
            conn.commit()
            logger.info("資料表建立完成")

        # 匯入來源清單（如果 sources 表是空的）
        seed_sources(conn)

    finally:
        conn.close()


def seed_sources(conn):
    """
    從 sources.csv 讀取來源清單並寫入資料庫
    
    為什麼從 CSV 讀取而不是寫死在程式碼？
      → 要新增或停用來源，只需要改 CSV，不用動程式碼
      → 非工程師（如 PM）也可以直接編輯 CSV 管理來源
      → 跟 Excel 盤點表格式一致，方便維護
    
    CSV 位置：../db/sources.csv（相對於 crawler/ 資料夾）
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM sources")
        count = cur.fetchone()[0]
        if count > 0:
            logger.info(f"   sources 表已有 {count} 筆資料，略過匯入")
            return

    # 找 CSV 檔案路徑
    # __file__ = 目前這個 Python 檔案的路徑（main.py）
    # os.path.dirname() = 取得資料夾路徑
    # os.path.join() ≈ JS 的 path.join()
    csv_path = "/db/sources.csv"

    if not os.path.exists(csv_path):
        logger.error(f" 找不到 sources.csv：{csv_path}")
        logger.error("   請確認 db/sources.csv 存在")
        return

    # 讀取 CSV
    # csv.DictReader 把每一列讀成 dict，key = 欄位名稱
    # 就像 JS 的 array of objects
    import csv
    sources_data = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # is_active 在 CSV 是字串 "true"/"false"，要轉成 Python bool
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
    """
    if __name__ == "__main__" 
    
    當你直接執行 `python main.py` 時，
    Python 會把 __name__ 設為 "__main__"
    """
    import sys
    
    # 確保 logs 資料夾存在
    os.makedirs("logs", exist_ok=True)

    # 判斷執行模式
    # sys.argv = 命令列參數的清單
    # 執行 `python main.py init`   → sys.argv = ["main.py", "init"]
    # 執行 `python main.py now`    → sys.argv = ["main.py", "now"]
    # 執行 `python main.py`        → sys.argv = ["main.py"]（啟動排程）
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "schedule"

    if mode == "init":
        # 初始化模式：建立資料表 + 匯入來源
        init_db()
        logger.info("初始化完成！現在可以執行 python main.py now 測試抓取")

    elif mode == "now":
        # 立即執行模式：馬上抓一次（測試用）
        logger.info("⚡ 立即執行模式（測試用）")
        run_daily_fetch()

    else:
        # 排程模式：每天指定時間自動執行
        fetch_hour   = int(os.getenv("FETCH_HOUR", 8))
        fetch_minute = int(os.getenv("FETCH_MINUTE", 0))
        run_time     = f"{fetch_hour:02d}:{fetch_minute:02d}"

        logger.info(f"排程模式啟動，每天 {run_time} 執行")
        
        # schedule.every().day.at("08:00") = 每天早上 8 點執行
        schedule.every().day.at(run_time).do(run_daily_fetch)

        # 無限迴圈，讓程式持續運行等待排程觸發
        # 這就是為什麼 Docker 容器要一直跑著
        while True:
            schedule.run_pending()  # 檢查是否有任務需要執行
            time.sleep(60)          # 每 60 秒檢查一次