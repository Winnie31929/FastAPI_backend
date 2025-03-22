from fastapi import FastAPI, File, UploadFile, Form
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
import json

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

# Pydantic Model
class CorrectionRequest(BaseModel):
    corrected_by: str # 醫生 ID
    wound_id: str # 要改的傷口紀錄 ID
    corrected_class: Optional[str] = None # 修改的傷口類別
    corrected_severity: Optional[str]  = None# 修改的傷口嚴重程度
    corrected_treatment_suggestions: Optional[str] = None # 修改的治療建議

# 儲存使用者資訊
@app.post("/add_user/")
async def add_patient(user: UserCreate, doctor_id: Optional[str] = None):
    """
    儲存使用者資訊到 MongoDB
    """
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


# 更新使用者資訊的 Pydantic Model
class UserUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(..., pattern=PHONE_REGEX)
    address: str 
    medical_history: str 

# 更新使用者資訊 API
@app.put("/update_user/{user_id}/")
async def update_user(user_id: str, update_data: UserUpdate):
    """ 更新使用者資訊 """

    # 確保 user_id 是合法的 MongoDB ObjectId
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="無效的 user_id")

    user = await collection_users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")

    # 轉換更新資料為字典
    update_dict = dict(update_data)

    # 更新時間戳
    update_dict["updated_at"] = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))

    # 執行更新
    await collection_users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_dict}
    )

    return {"message": "使用者資料已更新"}

# Pydantic Model: 修改密碼請求
class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=8)  # 舊密碼
    new_password: str = Field(..., min_length=8)  # 新密碼
    confirm_new_password: str = Field(..., min_length=8)  # 確認新密碼

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


# 修改密碼 API
@app.put("/change_password/{user_id}/")
async def change_password(user_id: str, request: ChangePasswordRequest):
    """ 使用者變更密碼 """

    # 確保 user_id 是合法的 MongoDB ObjectId
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="無效的 user_id")

    user = await collection_users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")

    # 檢查舊密碼是否正確
    if not verify_password(request.old_password, user["password_hash"]):
        raise HTTPException(status_code=403, detail="舊密碼不正確")

    # 檢查新密碼與確認密碼是否相符
    if request.new_password != request.confirm_new_password:
        raise HTTPException(status_code=400, detail="新密碼與確認密碼不一致")

    # 檢查密碼強度
    ChangePasswordRequest.validate_password(request.new_password)

    # 加密新密碼並更新
    new_password_hash = hash_password(request.new_password)
    await collection_users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_hash": new_password_hash, "updated_at": datetime.now()}}
    )

    return {"message": "密碼已成功更新"}

# 使用者登入
@app.post("/login/")
async def login(email: str = Form(...), password: str = Form(...)):
    """
    使用者登入
    """
    user = await collection_users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="使用者不存在")

    # 驗證密碼
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="密碼錯誤")

    return {"message": "登入成功"}

# 儲存傷口紀錄
# {"patient_id":"67d796801cc030761516faa5","title": "右腳掌傷口1","wound_location":"右腳掌"}
@app.post("/add_wound/")
async def add_wound(wound_json: str = Form(...), file: UploadFile = File(...)):
    """
    儲存傷口紀錄，接收JSON數據和圖片，並存入 MongoDB 和 GridFS
    """
    now_time = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))  # 獲取當前時間
    wound_data = json.loads(wound_json)  # 將 json 轉換成字典
    wound_data["created_at"] = now_time
    wound_data["updated_at"] = now_time
    # 儲存圖片到 GridFS
    image_data = await file.read()  # 讀取圖片檔案
    image_stream = BytesIO(image_data)  # 將圖片轉成二進位流

    # 上傳圖片至 GridFS
    file_id = await fs.upload_from_stream(filename=file.filename, source=image_stream) 

    # 將圖片的 GridFS file_id 加入傷口記錄
    wound_data["image_file_id"] = str(file_id)

    # 儲存傷口記錄到 MongoDB
    result = await collection_wound_records.insert_one(wound_data)

    return {"inserted_id": str(result.inserted_id), "image_file_id": str(file_id)}

# 更新傷口紀錄
@app.put("/update_wound/{wound_id}/")
async def update_wound(wound_id: str, update_data: dict):
    """ 更新傷口紀錄 """

    # 確保 wound_id 是合法的 MongoDB ObjectId
    if not ObjectId.is_valid(wound_id):
        raise HTTPException(status_code=400, detail="無效的 wound_id")

    wound = await collection_wound_records.find_one({"_id": ObjectId(wound_id)})
    if not wound:
        raise HTTPException(status_code=404, detail="傷口紀錄不存在")

    # 轉換更新資料為字典
    update_dict = dict(update_data)

    # 更新時間戳
    update_dict["updated_at"] = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))

    # 執行更新
    await collection_wound_records.update_one(
        {"_id": ObjectId(wound_id)},
        {"$set": update_dict}
    )

    return {"message": "傷口紀錄已更新"}

# 儲存 ML 預測結果
"""
{
    "wound_id": "67da78f1bd148b1bd6b8d866",
    "model_version": "v1",
    "predicted_class": "N",
    "predicted_severity": "W0",
    "predicted_treatment_suggestions": "No"
}
"""
@app.post("/add_ml_prediction/")
async def add_ml_prediction(prediction_json: str = Form(...), file: UploadFile = File(...)):
    """
    儲存ML預測結果，接收JSON數據和圖片，並存入 MongoDB 和 GridFS
    """
    now_time = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))  # 獲取當前時間
    prediction_data = json.loads(prediction_json)  # 將 json 轉換成字典
    prediction_data["predicted_date"] = now_time

    # 儲存已框出傷口邊界的圖片到 GridFS
    image_data = await file.read()  # 讀取圖片檔案
    image_stream = BytesIO(image_data)  # 將圖片轉成二進位流

    # 上傳圖片至 GridFS
    file_id = await fs.upload_from_stream(filename=file.filename, source=image_stream) 

    # 將圖片的 GridFS file_id 加入傷口記錄
    prediction_data["predicted_image_file_id"] = str(file_id)

    # 儲存ML預測結果到 MongoDB
    result = await collection_ml_predictions.insert_one(prediction_data)

    return {"inserted_id": str(result.inserted_id), "image_file_id": str(file_id)}

# ML預測結果加入醫生更正資訊
@app.put("/correct_ml_prediction/")
async def correct_ml_prediction(correction: CorrectionRequest):
    """
    ML預測結果加入醫生更正資訊
    """
    # 查找 wound_id 是否存在
    prediction = await collection_ml_predictions.find_one({"wound_id": correction.wound_id})

    if not prediction:
        raise HTTPException(status_code=404, detail="Wound ID not found in Ml_prediction")

    now_time = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))
    correction_data = dict(correction)
    correction_data["corrected_date"] = now_time

    # 更新 MongoDB 記錄
    await collection_ml_predictions.update_one(
        {"wound_id": correction.wound_id}, {"$set": correction_data}
    )

    return {"message": "Add correct information successfully"}

# 獲取醫生所有的病患名字
@app.get("/get_patients/{doctor_id}")
async def get_patients(doctor_id: str):
    """
    獲取醫生所有的病患名字
    """
    # 1. 查詢病患 ID
    patient_ids = await collection_doctor_patient.find(
        {"doctor_id": doctor_id}, {"_id": 0, "patient_id": 1}
    ).to_list(length=100)

    # 取得 patient_id 清單
    patient_ids = [p["patient_id"] for p in patient_ids]
    if not patient_ids:
        raise HTTPException(status_code=404, detail="No patients found for this doctor")

    # 轉換 patient_id 為 ObjectId
    patient_object_ids = [ObjectId(patient_id) for patient_id in patient_ids]
    # 2. 透過 patient_id 查詢病患名字
    patients = await collection_users.find(
        {"_id": {"$in": patient_object_ids}}, {"_id": 0, "name": 1}
    ).to_list(length=100)

    return {"patients": patients}

# 根據病患ID，獲取所有傷口記錄和其ML預測分類和分級結果（不包含圖片）
@app.get("/get_wound_list/{patient_id}")
async def get_wound_list(patient_id: str):
    """
    根據病患ID，獲取所有傷口記錄和其ML預測分類和分級結果（不包含圖片）
    """
    # 1. 查詢病患的所有傷口記錄
    records = await collection_wound_records.find({"patient_id": patient_id}, {"_id": 1}).to_list(length=100)
    if not records:
        raise HTTPException(status_code=404, detail="No wound records found for this patient")
    
    # 取得 wound_id 清單
    wound_ids = [str(r["_id"]) for r in records]

    # 2. 透過 wound_id 查詢 ML 預測結果
    ml_predictions = await collection_ml_predictions.find(
        {"wound_id": {"$in": wound_ids}}, {"_id": 0, "predicted_class": 1, "predicted_severity": 1, "predicted_date": 1}
    ).to_list(length=100)

    return {"data": ml_predictions}

# 根據傷口ID，獲取ML預測結果
@app.get("/get_predicted_result/{wound_id}")
async def get_predicted_result(wound_id: str):
    """
    根據傷口ID，獲取ML預測結果
    """
    # 1. 查詢 ML 預測結果
    prediction = await collection_ml_predictions.find_one({"wound_id": wound_id}, {"_id": 0})
    if not prediction:
        raise HTTPException(status_code=404, detail="No ML prediction found for this wound")

    return prediction

# 根據圖片的 file_id 從 GridFS 下載圖片
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

