-- =============================================
-- AI 知識庫蒐集層 資料庫結構
-- =============================================

-- 來源管理表
-- 對應你 Excel 裡的 25 個來源
CREATE TABLE IF NOT EXISTS sources (
    id              SERIAL PRIMARY KEY,
    resources_type  VARCHAR(50),               -- Institution / Official / Media / Community
    name            VARCHAR(200) NOT NULL,     -- 來源名稱
    url             TEXT NOT NULL UNIQUE,      -- RSS 或 API 網址
    language        VARCHAR(10),               -- zh / en
    type            VARCHAR(50),               -- Research / Technical / News / Industry
    description     TEXT,                      -- 來源說明
    is_active       BOOLEAN DEFAULT TRUE,       -- 是否啟用
    source_tier     CHAR(1) DEFAULT 'B',        -- A=官方研究 B=媒體 C=機構政策
    fetcher         VARCHAR(10) DEFAULT 'rss',  -- rss / arxiv / skip
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 文章主表
-- 每一篇抓回來的文章都存在這裡
CREATE TABLE IF NOT EXISTS articles (
    id              SERIAL PRIMARY KEY,
    source_id       INTEGER REFERENCES sources(id),
    title           TEXT NOT NULL,             -- 文章標題
    url             TEXT NOT NULL UNIQUE,       -- 文章網址（唯一，防重複）
    content         TEXT,                      -- 文章內文
    summary         TEXT,                      -- 摘要（200字以內）
    published_at    TIMESTAMP,                 -- 原始發布時間
    fetched_at      TIMESTAMP DEFAULT NOW(),   -- 抓取時間

    -- 治理層欄位（蒐集層先留空，後面用）
    source_tier     CHAR(1),
    governance_status VARCHAR(20) DEFAULT 'pending',
    credibility_score FLOAT,

    -- 分類標記
    language        VARCHAR(10),
    content_type    VARCHAR(20),               -- rss / arxiv

    created_at      TIMESTAMP DEFAULT NOW()
);

-- 抓取記錄表
-- 每次執行結果都記在這，方便除錯
CREATE TABLE IF NOT EXISTS fetch_logs (
    id              SERIAL PRIMARY KEY,
    source_id       INTEGER REFERENCES sources(id),
    source_name     VARCHAR(200),
    status          VARCHAR(20),               -- success / failed / skipped
    articles_found  INTEGER DEFAULT 0,         -- 這次找到幾篇
    articles_new    INTEGER DEFAULT 0,         -- 這次新增幾篇
    error_message   TEXT,                      -- 如果失敗，原因是什麼
    executed_at     TIMESTAMP DEFAULT NOW()
);

-- 常用索引（讓查詢速度更快）
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_governance_status ON articles(governance_status);
CREATE INDEX IF NOT EXISTS idx_fetch_logs_executed_at ON fetch_logs(executed_at);