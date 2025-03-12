from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from io import BytesIO
from bson import ObjectId
import time
from pydantic import BaseModel, EmailStr
from typing import List
from datetime import date, datetime
from passlib.context import CryptContext

app = FastAPI()

# 使用 motor 進行非同步連線
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.get_database("WoundCareApp")  
collection_wound_records = db["wound_records"]
collection_users = db["users"]
collection_ml_predictions = db["ml_predictions"]

# 使用 motor 來建立 GridFS 儲存桶
fs = AsyncIOMotorGridFSBucket(db)

# 創建加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 生成雜湊密碼
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# 驗證密碼
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# 定義 Pydantic Model
class PatientCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    day_of_birth: date
    phone: str
    address: str
    medical_history: List[str] = []

class MedicalStaffCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    day_of_birth: date
    phone: str
    address: str
    patients: List[str] = []

# 儲存病患資訊
@app.post("/add_patient/")
async def add_patient(patient: PatientCreate):
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))  # 獲取當前時間
    now_time = now.strftime('%Y/%m/%d %H:%M:%S')  # 格式化時間，如2021/10/19 14:48:38
    password_hash = hash_password(patient.password)  # 將密碼加密

    data = dict(patient)  # 將 Pydantic Model 轉換成字典
    data["password_hash"] = password_hash
    data["role"] = "patient"
    data["created_at"] = now_time
    data["updated_at"] = now_time
    del data["password"]  # 不儲存明文密碼

    result = await collection_users.insert_one(data)
    return {"inserted_id": str(result.inserted_id)}

# 儲存醫護人員資訊
@app.post("/add_medical_staff/")
async def add_medical_staff(staff: MedicalStaffCreate):
    now = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))  # 獲取當前時間
    now_time = now.strftime('%Y/%m/%d %H:%M:%S')  # 格式化時間，如2021/10/19 14:48:38
    password_hash = hash_password(staff.password)

    data = dict(staff)
    data["password_hash"] = password_hash
    data["role"] = "medical_staff"
    data["created_at"] = now_time
    data["updated_at"] = now_time
    del data["password"]

    result = await collection_users.insert_one(data)
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

