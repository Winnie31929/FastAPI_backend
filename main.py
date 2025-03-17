from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from pymongo import ReturnDocument
import io
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from io import BytesIO
from bson import ObjectId
import time
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from passlib.context import CryptContext
import datetime as dt
from datetime import date, datetime
import re

app = FastAPI()

# 使用 motor 進行非同步連線
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.get_database("WoundCareApp")  
collection_wound_records = db["Wound_Records"]
collection_users = db["Users"]
collection_ml_predictions = db["ML_Predictions"]
collection_doctor_patient = db["Doctor_Patient"]

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


# 手機號碼正則表達式：09開頭，後面接8位數字
PHONE_REGEX = r"^09\d{8}$"

# 定義 Pydantic Model
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)
    day_of_birth: date
    phone: str = Field(..., pattern=PHONE_REGEX)
    address: Optional[str] = None
    medical_history: Optional[str] = None
    role: str

    # 密碼：至少8字元，包含大寫、小寫、數字，可有特殊符號
    @field_validator("password")
    @classmethod
    def validate_password(cls, value):
        """檢查密碼是否符合強度要求"""
        if not re.search(r"[a-z]", value):
            raise ValueError("密碼必須包含至少一個小寫字母")
        if not re.search(r"[A-Z]", value):
            raise ValueError("密碼必須包含至少一個大寫字母")
        if not re.search(r"\d", value):
            raise ValueError("密碼必須包含至少一個數字")
        return value

    

# 儲存使用者資訊
@app.post("/add_user/")
async def add_patient(user: UserCreate, doctor_id: Optional[str] = None):
    now_time = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))  # 獲取當前時間
    password_hash = hash_password(user.password)  # 將密碼加密

    # 轉換 `day_of_birth` 為 `datetime.datetime`
    day_of_birth_dt = datetime(user.day_of_birth.year, user.day_of_birth.month, user.day_of_birth.day)

    # 確保 doctor_id 存在於資料庫
    if doctor_id:
        doctor = await collection_users.find_one({"_id": ObjectId(doctor_id), "role": "medical_staff"})
        if not doctor:
            raise HTTPException(status_code=404, detail="指定的醫生不存在")

    # 建立資料
    data = dict(user)  # 將 Pydantic Model 轉換成字典
    data["password_hash"] = password_hash
    data["day_of_birth"] = day_of_birth_dt
    data["created_at"] = now_time
    data["updated_at"] = now_time
    del data["password"]  # 不儲存明文密碼

    # 將資料儲存到 MongoDB
    result = await collection_users.insert_one(data)
    patient_id = str(result.inserted_id)

    # 如果有指定醫生，則建立醫生與病人的關係
    if doctor_id:
        doctor_patient_data = {
            "doctor_id": doctor_id,
            "patient_id": patient_id,
            "assigned_date": now_time,
        }
        await collection_doctor_patient.insert_one(doctor_patient_data)
    return {"inserted_id": patient_id}

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

