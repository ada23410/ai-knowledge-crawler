"""
Vertex AI 連線測試腳本
=====================================================
執行方式：
  python test_vertex_ai.py

測試項目：
  1. 檢查 .env 設定是否齊全
  2. 確認 GCP 憑證可以讀取
  3. 實際呼叫 gemini-embedding-001，送一句話進去
  4. 確認回傳的向量維度正確（3072）
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════
# Step 1：檢查 .env 設定
# ══════════════════════════════════════════════

print("=" * 50)
print("Step 1：檢查 .env 設定")
print("=" * 50)

required_vars = {
    "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
    "GCP_REGION": os.getenv("GCP_REGION", "us-central1"),
    "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
}

all_ok = True
for key, value in required_vars.items():
    if value:
        print(f"  {key} = {value}")
    else:
        print(f"  {key} 未設定")
        all_ok = False

if not all_ok:
    print("\n請先在 .env 補齊以上設定，再重新執行。")
    sys.exit(1)

# ══════════════════════════════════════════════
# Step 2：確認 JSON 金鑰檔案存在
# ══════════════════════════════════════════════

print("\n" + "=" * 50)
print("Step 2：確認 JSON 金鑰檔案")
print("=" * 50)

cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if os.path.exists(cred_path):
    print(f"  金鑰檔案存在：{cred_path}")
else:
    print(f"  找不到金鑰檔案：{cred_path}")
    print("   請確認路徑是否正確，以及檔案是否已放到對應位置。")
    sys.exit(1)

# ══════════════════════════════════════════════
# Step 3：初始化 Vertex AI
# ══════════════════════════════════════════════

print("\n" + "=" * 50)
print("Step 3：初始化 Vertex AI SDK")
print("=" * 50)

try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

    vertexai.init(
        project=os.getenv("GCP_PROJECT_ID"),
        location=os.getenv("GCP_REGION", "us-central1"),
    )
    print("  Vertex AI 初始化成功")

except ImportError:
    print("  找不到 google-cloud-aiplatform 套件")
    print("  請執行：pip install google-cloud-aiplatform==1.49.0")
    sys.exit(1)

except Exception as e:
    print(f" 初始化失敗：{e}")
    sys.exit(1)

# ══════════════════════════════════════════════
# Step 4：實際呼叫 embedding API
# ══════════════════════════════════════════════

print("\n" + "=" * 50)
print("Step 4：呼叫 gemini-embedding-001")
print("=" * 50)

TEST_TEXT = "Google 與 NVIDIA 合作開發新一代 AI 晶片，預計 2026 年量產。"
print(f"  測試文字：{TEST_TEXT}")

try:
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

    inputs = [TextEmbeddingInput(text=TEST_TEXT, task_type="SEMANTIC_SIMILARITY")]
    response = model.get_embeddings(inputs)

    vector = response[0].values
    print(f"  呼叫成功")
    print(f"  向量維度：{len(vector)}（預期：3072）")
    print(f"  向量前 5 個值：{[round(v, 6) for v in vector[:5]]}")

    if len(vector) == 3072:
        print("  維度正確")
    else:
        print(f"  維度異常，預期 3072，實際 {len(vector)}")

except Exception as e:
    print(f" API 呼叫失敗：{e}")
    print()
    print("  常見原因：")
    print("    - Vertex AI API 尚未啟用（確認 GCP Console 已 Enable）")
    print("    - 服務帳戶缺少 Vertex AI User 角色")
    print("    - GCP_PROJECT_ID 填錯")
    print("    - 金鑰 JSON 對應的專案與 GCP_PROJECT_ID 不同")
    sys.exit(1)

# ══════════════════════════════════════════════
# 完成
# ══════════════════════════════════════════════

print("\n" + "=" * 50)
print("所有測試通過，可以執行 dedup.py 了")
print("=" * 50)