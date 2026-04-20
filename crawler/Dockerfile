# Python 3.11 輕量版作為基底
# alpine = 超小型 Linux，讓容器體積最小化
FROM python:3.11-slim

# 設定工作目錄（之後的指令都在這個路徑執行）
WORKDIR /app

# 安裝系統依賴
# lxml 需要這些系統套件才能編譯
RUN apt-get update && apt-get install -y \
    gcc \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# 先複製 requirements.txt（利用 Docker cache）
# 只要 requirements.txt 沒變，這層就不用重新安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 建立 logs 資料夾
RUN mkdir -p logs

# 預設執行指令（啟動排程模式）
CMD ["python", "main.py"]