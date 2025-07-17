# 使用 Python 3.10 為基礎映像檔
FROM python:3.10

# 設定工作目錄
WORKDIR /app

# 複製本機所有檔案到容器中
COPY . /app

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 啟動 FastAPI 服務
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
