"""
email_notifier.py — 待審稿文章 Email 通知模組
================================================
功能：每週彙整 review 和 human_required 的文章，寄送 Email 給指定同事
執行時機：每週一 dedup → scoring → tagging 完成後自動觸發
寄信方式：Gmail SMTP（App 密碼認證）
"""

import os
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1. 設定區
# ══════════════════════════════════════════════

GMAIL_SENDER       = os.getenv("GMAIL_SENDER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NOTIFY_RECIPIENTS  = os.getenv("NOTIFY_RECIPIENTS", "")

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


def fetch_review_articles(conn) -> dict:
    """
    從資料庫撈出本週需要人工審稿的文章

    回傳：
      {
        "human_required": [...],   # 必須審稿
        "review": [...]            # 建議審稿
      }
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                a.id,
                a.title,
                a.url,
                a.language,
                a.credibility_score,
                a.published_at,
                s.name AS source_name,
                s.source_tier
            FROM articles a
            JOIN sources s ON a.source_id = s.id
            WHERE a.governance_status IN ('review', 'human_required')
              AND a.is_deleted = FALSE
            ORDER BY
                a.governance_status DESC,
                a.credibility_score DESC,
                a.published_at DESC
        """)
        rows = cur.fetchall()
        columns = ["id", "title", "url", "language", "credibility_score",
                   "published_at", "source_name", "source_tier"]
        articles = [dict(zip(columns, row)) for row in rows]

    result = {
        "human_required": [a for a in articles if a["credibility_score"] is not None and a["credibility_score"] < 0.45],
        "review":         [a for a in articles if a["credibility_score"] is not None and a["credibility_score"] >= 0.45],
    }

    logger.info(
        "撈出待審稿文章：human_required %d 篇，review %d 篇",
        len(result["human_required"]),
        len(result["review"]),
    )
    return result


# ══════════════════════════════════════════════
# 3. Email 內容組裝
# ══════════════════════════════════════════════

def build_email_body(articles: dict) -> tuple[str, str]:
    """
    組裝 Email 的純文字版和 HTML 版內容

    Args:
        articles: fetch_review_articles 回傳的字典

    Returns:
        (text_body, html_body)
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    total = len(articles["human_required"]) + len(articles["review"])

    # ── 純文字版 ──
    lines = []
    lines.append(f"AI 知識庫｜本週待審稿文章清單 {date_str}")
    lines.append(f"本週共有 {total} 篇文章需要人工審稿")
    lines.append("")

    if articles["human_required"]:
        lines.append(f"【必須審稿】{len(articles['human_required'])} 篇（分數 < 0.45）")
        lines.append("=" * 50)
        for i, a in enumerate(articles["human_required"], 1):
            score    = f"{a['credibility_score']:.2f}" if a["credibility_score"] is not None else "N/A"
            pub_date = a["published_at"].strftime("%Y-%m-%d") if a["published_at"] else "未知"
            lines.append(f"{i}. {a['title']}")
            lines.append(f"   來源：{a['source_name']}（Tier {a['source_tier']}）｜發布：{pub_date}｜分數：{score}｜語言：{a['language']}")
            lines.append(f"   連結：{a['url']}")
            lines.append("")

    if articles["review"]:
        lines.append(f"【建議審稿】{len(articles['review'])} 篇（分數 0.45～0.60）")
        lines.append("=" * 50)
        for i, a in enumerate(articles["review"], 1):
            score    = f"{a['credibility_score']:.2f}" if a["credibility_score"] is not None else "N/A"
            pub_date = a["published_at"].strftime("%Y-%m-%d") if a["published_at"] else "未知"
            lines.append(f"{i}. {a['title']}")
            lines.append(f"   來源：{a['source_name']}（Tier {a['source_tier']}）｜發布：{pub_date}｜分數：{score}｜語言：{a['language']}")
            lines.append(f"   連結：{a['url']}")
            lines.append("")

    lines.append("---")
    lines.append("此信由 AI 知識庫系統自動發送，請勿直接回覆。")
    text_body = "\n".join(lines)

    # ── HTML 版 ──
    def article_rows(article_list: list) -> str:
        rows = ""
        for a in article_list:
            score    = f"{a['credibility_score']:.2f}" if a["credibility_score"] is not None else "N/A"
            pub_date = a["published_at"].strftime("%Y-%m-%d") if a["published_at"] else "未知"
            title    = a["title"] or "（無標題）"
            url      = a["url"] or "#"
            rows += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #eee;">
                    <a href="{url}" style="color: #1a73e8; text-decoration: none; font-weight: 500;">
                        {title}
                    </a>
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #eee; white-space: nowrap; color: #555;">
                    {a['source_name']}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; white-space: nowrap; color: #555;">
                    {pub_date}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; white-space: nowrap;">
                    <span style="background: #f1f3f4; padding: 2px 8px; border-radius: 4px; font-size: 13px;">
                        {score}
                    </span>
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; color: #555;">
                    {a['language'].upper()}
                </td>
            </tr>
            """
        return rows

    def table_header(color_bg: str, color_text: str) -> str:
        return f"""
        <thead>
            <tr style="background: {color_bg};">
                <th style="padding: 10px; text-align: left; color: {color_text};">文章標題</th>
                <th style="padding: 10px; text-align: left; color: {color_text};">來源</th>
                <th style="padding: 10px; text-align: center; color: {color_text};">發布日期</th>
                <th style="padding: 10px; text-align: center; color: {color_text};">分數</th>
                <th style="padding: 10px; text-align: center; color: {color_text};">語言</th>
            </tr>
        </thead>
        """

    human_section = ""
    if articles["human_required"]:
        human_section = f"""
        <div style="margin-bottom: 32px;">
            <h2 style="color: #d93025; font-size: 16px; margin-bottom: 8px;">
                ⚠️ 必須審稿｜{len(articles['human_required'])} 篇（分數 &lt; 0.45）
            </h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                {table_header('#fce8e6', '#d93025')}
                <tbody>
                    {article_rows(articles['human_required'])}
                </tbody>
            </table>
        </div>
        """

    review_section = ""
    if articles["review"]:
        review_section = f"""
        <div style="margin-bottom: 32px;">
            <h2 style="color: #e37400; font-size: 16px; margin-bottom: 8px;">
                📋 建議審稿｜{len(articles['review'])} 篇（分數 0.45～0.60）
            </h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                {table_header('#fef7e0', '#e37400')}
                <tbody>
                    {article_rows(articles['review'])}
                </tbody>
            </table>
        </div>
        """

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 24px; color: #333;">

        <div style="border-bottom: 3px solid #1a73e8; padding-bottom: 16px; margin-bottom: 24px;">
            <h1 style="font-size: 20px; margin: 0; color: #1a73e8;">
                AI 知識庫｜本週待審稿文章清單
            </h1>
            <p style="margin: 4px 0 0; color: #555; font-size: 14px;">
                {date_str}｜共 {total} 篇需要人工審稿
            </p>
        </div>

        {human_section}
        {review_section}

        <div style="border-top: 1px solid #eee; padding-top: 16px; margin-top: 24px;
                    font-size: 12px; color: #999;">
            此信由 AI 知識庫系統自動發送，請勿直接回覆。
        </div>

    </body>
    </html>
    """

    return text_body, html_body


# ══════════════════════════════════════════════
# 4. 寄信
# ══════════════════════════════════════════════

def send_email(subject: str, text_body: str, html_body: str) -> bool:
    """
    用 Gmail SMTP 寄送 Email

    Args:
        subject:   信件主旨
        text_body: 純文字版內容（備援）
        html_body: HTML 版內容（主要）

    Returns:
        True = 寄送成功，False = 失敗
    """
    recipients = [r.strip() for r in NOTIFY_RECIPIENTS.split(",") if r.strip()]

    if not recipients:
        logger.error("NOTIFY_RECIPIENTS 未設定，無法寄信")
        return False

    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD:
        logger.error("GMAIL_SENDER 或 GMAIL_APP_PASSWORD 未設定")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = ", ".join(recipients)

        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html",  "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, recipients, msg.as_string())

        logger.info("Email 寄送成功，收件人：%s", ", ".join(recipients))
        return True

    except Exception as e:
        logger.error("Email 寄送失敗：%s", e)
        return False


# ══════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════

def run_notification():
    """
    執行完整的 Email 通知流程

    流程：
    1. 從資料庫撈出待審稿文章
    2. 如果沒有待審稿文章，跳過不寄信
    3. 組裝 Email 內容
    4. 寄送 Email
    """
    logger.info("====== 開始執行 Email 通知 ======")

    conn = None
    try:
        conn = get_db_connection()
        articles = fetch_review_articles(conn)

        total = len(articles["human_required"]) + len(articles["review"])

        if total == 0:
            logger.info("本週無待審稿文章，不寄送通知")
            return

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        subject = f"【AI 知識庫】本週待審稿文章清單 {date_str}（共 {total} 篇）"

        text_body, html_body = build_email_body(articles)
        success = send_email(subject, text_body, html_body)

        if success:
            logger.info("====== Email 通知完成 ======")
        else:
            logger.error("====== Email 通知失敗 ======")

    except Exception as e:
        logger.error("通知流程發生錯誤：%s", e)
        raise

    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════
# 6. 程式入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/notifier.log", encoding="utf-8"),
        ],
    )

    run_notification()