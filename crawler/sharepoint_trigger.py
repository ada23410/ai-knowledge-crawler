"""
sharepoint_trigger.py — SharePoint 同步觸發模組
=================================================
功能：撈出資料庫所有 approved 文章，以 JSON 附件寄送至內網信箱
      觸發 Power Automate Flow 自動寫入 SharePoint List
執行時機：每週一 LLM 摘要生成完成後自動觸發
寄信方式：Gmail SMTP（App 密碼認證）
"""

import json
import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1. 設定區
# ══════════════════════════════════════════════

GMAIL_SENDER           = os.getenv("GMAIL_SENDER")
GMAIL_APP_PASSWORD     = os.getenv("GMAIL_APP_PASSWORD")
SHAREPOINT_NOTIFY_MAIL = os.getenv("SHAREPOINT_NOTIFY_MAIL")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


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


def fetch_approved_articles(conn) -> list[dict]:
    """
    撈出所有尚未同步至 SharePoint 的 approved 文章

    Returns:
        文章列表，每筆包含 SharePoint 所需欄位
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                a.title,
                a.url,
                a.language,
                a.credibility_score,
                a.published_at,
                a.tags,
                a.summary,
                s.name          AS source_name,
                s.resources_type
            FROM articles a
            JOIN sources s ON a.source_id = s.id
            WHERE a.governance_status  = 'approved'
              AND a.is_deleted         = FALSE
              AND a.sharepoint_synced  = FALSE
            ORDER BY a.published_at DESC
            LIMIT 5
        """)
        rows    = cur.fetchall()
        columns = [
            "title", "url", "language", "credibility_score",
            "published_at", "tags", "summary", 
            "source_name", "resources_type",
        ]
        articles = [dict(zip(columns, row)) for row in rows]

    logger.info("撈出待同步 approved 文章：%d 篇", len(articles))
    return articles


def mark_articles_synced(conn, urls: list[str]) -> None:
    """
    將已寄送的文章標記為 sharepoint_synced = TRUE
    避免下次重複寄送

    Args:
        conn: 資料庫連線
        urls: 已寄送的文章 URL 列表
    """
    if not urls:
        return
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE articles
            SET sharepoint_synced = TRUE
            WHERE url = ANY(%s)
        """, (urls,))
    conn.commit()
    logger.info("已標記 %d 篇文章為 sharepoint_synced", len(urls))


# ══════════════════════════════════════════════
# 3. 寄信觸發 Power Automate
# ══════════════════════════════════════════════

def send_sharepoint_trigger(articles: list[dict]) -> bool:
    """
    將 approved 文章以 JSON 附件寄送至內網信箱
    觸發 Power Automate Flow 自動寫入 SharePoint List

    Args:
        articles: fetch_approved_articles 回傳的文章列表

    Returns:
        True = 寄送成功，False = 失敗
    """
    if not SHAREPOINT_NOTIFY_MAIL:
        logger.error("SHAREPOINT_NOTIFY_MAIL 未設定")
        return False

    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD:
        logger.error("GMAIL_SENDER 或 GMAIL_APP_PASSWORD 未設定")
        return False

    if not articles:
        logger.info("無 approved 文章需要同步")
        return True

    try:
        msg = MIMEMultipart()
        msg["Subject"] = "AI_DIGEST_IMPORT"
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = SHAREPOINT_NOTIFY_MAIL

        # 組裝 JSON 附件，對應 Power Automate Schema
        payload = [
            {
                "title":          a.get("title") or "",
                "source_name":    a.get("source_name") or "",
                "url":            a.get("url") or "",
                "summary":        a.get("summary") or "",
                "tags":           a.get("tags") or [],          # ← 直接保留陣列，不用 json.dumps
                "score":          float(a.get("credibility_score") or 0),
                "published_at":   a["published_at"].isoformat() if a.get("published_at") else "",
            }
            for a in articles
        ]

        attachment = MIMEBase("application", "octet-stream")
        attachment.set_payload(
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        )
        encoders.encode_base64(attachment)
        attachment.add_header(
            "Content-Disposition",
            "attachment",
            filename="digest.json"
        )
        msg.attach(attachment)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.send_message(msg)

        logger.info("SharePoint 觸發信寄送成功，共 %d 篇", len(payload))
        return True

    except Exception as e:
        logger.error("SharePoint 觸發信寄送失敗：%s", e)
        return False


# ══════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════

def run_sharepoint_sync():
    """
    執行 SharePoint 同步觸發流程

    流程：
    1. 撈出所有 approved 且尚未同步的文章
    2. 組裝 JSON 附件寄送至內網信箱
    3. 標記已寄送文章為 sharepoint_synced
    """
    logger.info("====== 開始執行 SharePoint 同步 ======")

    conn = None
    try:
        conn     = get_db_connection()
        articles = fetch_approved_articles(conn)

        if not articles:
            logger.info("無待同步文章，結束")
            return

        success = send_sharepoint_trigger(articles)

        if success:
            urls = [a["url"] for a in articles if a.get("url")]
            mark_articles_synced(conn, urls)
            logger.info("====== SharePoint 同步完成，共 %d 篇 ======", len(articles))
        else:
            logger.error("====== SharePoint 同步失敗 ======")

    except Exception as e:
        logger.error("SharePoint 同步流程發生錯誤：%s", e)
        raise

    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════
# 5. 程式入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/sharepoint_trigger.log", encoding="utf-8"),
        ],
    )

    run_sharepoint_sync()