from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from io import BytesIO
from bson import ObjectId
import time

app = FastAPI()

# 使用 motor 進行非同步連線
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.get_database("WoundCareApp")  
collection_wound_records = db["wound_records"]

# 使用 motor 來建立 GridFS 儲存桶
fs = AsyncIOMotorGridFSBucket(db)

# 儲存使用者資訊
@app.post("/add_user/")
async def add_user(username: str, email: str):
    """
    儲存使用者資訊
    """
    data = {
        "name": username,
        "email": email,
    }
    result = await db.users.insert_one(data)
    return {"inserted_id": str(result.inserted_id)}

# 儲存傷口紀錄
@app.post("/add_wound/")
async def add_wound(file: UploadFile = File(...)):
    """
    接收 JSON 數據和圖片，並存入 MongoDB 和 GridFS
    """
    now_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())  # 獲取當前時間
    data = {
        "patient_id": str(ObjectId()),
        "wound_type": "Diabetic Ulcer",
        "wound_location": "Foot",
        "notes": " ",
        "created_at": now_time,
        "updated_at": now_time,
    }
    # 儲存圖片到 GridFS
    image_data = await file.read()  # 讀取圖片檔案
    image_stream = BytesIO(image_data)  # 將圖片轉成二進位流

    # 上傳圖片至 GridFS
    file_id = await fs.upload_from_stream(filename=file.filename, source=image_stream) 

    # 將圖片的 GridFS file_id 加入傷口記錄
    data["image_file_id"] = str(file_id)

    # 儲存傷口記錄到 MongoDB
    result = await collection_wound_records.insert_one(data)

    return {"inserted_id": str(result.inserted_id), "image_file_id": str(file_id)}

@app.get("/get_wounds/")
async def get_wounds():
    """
    獲取所有傷口記錄（不包含圖片）
    """
    records = await collection_wound_records.find({}, {"_id": 0}).to_list(length=100)
    return {"data": records}

@app.get("/get_wound_image/{file_id}")
async def get_wound_image(file_id: str):
    """
    根據圖片的 file_id 從 GridFS 下載圖片
    """
    grid_out = await fs.open_download_stream(ObjectId(file_id))  # 非同步下載
    image_data = await grid_out.read()  # 讀取圖片內容

    # 透過 StreamingResponse 回傳圖片
    return StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) #執行：uvicorn main:app --reload，並搜尋http://127.0.0.1:8000/docs

