from fastapi import FastAPI, File, UploadFile, Form, Depends, Security
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from fastapi.security import OAuth2PasswordBearer
import io
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from io import BytesIO
from bson import ObjectId
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from passlib.context import CryptContext
import datetime as dt
from datetime import date, datetime, timedelta
import re
import json
import jwt
from jwt import PyJWTError
import pandas as pd

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


# JWT 相關設定
SECRET_KEY = "your_secret_key"  # 設定一個強壯的密鑰
ALGORITHM = "HS256"  # 使用 HS256 加密算法
ACCESS_TOKEN_EXPIRE_MINUTES = 120  # 設定 Token 有效時間

# 創建加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 生成雜湊密碼
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# 驗證密碼
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# OAuth2PasswordBearer 用於接收前端傳來的 JWT Token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# 產生 JWT Token
def create_access_token(data: dict):
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = data.copy()
    expire = datetime.now(dt.timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt_token(token: str = Depends(oauth2_scheme)):
    """ 驗證 JWT Token 是否有效 """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        role: str = payload.get("role")
        if user_id is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return {"user_id": user_id, "role": role}

    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/protected/")
async def protected_route(user_data: dict = Depends(verify_jwt_token)):
    """ 需要登入的 API """
    return {"message": f"Hello, {user_data['username']}! Your role is {user_data['role']}."}

# 手機號碼正則表達式：09開頭，後面接8位數字
PHONE_REGEX = r"^09\d{8}$"

# 定義 Pydantic Model
class UserCreate(BaseModel):
    account: str = Field(..., min_length=1, max_length=100)
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
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"[A-Z]", value):
            raise ValueError("passwords must contain at least one upper case letter.")
        if not re.search(r"\d", value):
            raise ValueError("Password must contain at least one number.")
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
async def add_patient(user: UserCreate):
    """
    儲存使用者資訊到 MongoDB
    """
    now_time = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))  # 獲取當前時間
    password_hash = hash_password(user.password)  # 將密碼加密

    # 轉換 `day_of_birth` 為 `datetime.datetime`
    day_of_birth_dt = datetime(user.day_of_birth.year, user.day_of_birth.month, user.day_of_birth.day)

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
    return {"inserted_id": patient_id}

# 根據姓名、生日或電話號碼搜尋病人
@app.get("/search_patient/")
async def search_patient(name: Optional[str] = None, day_of_birth: Optional[date] = None, phone: Optional[str] = None):
    """
    根據姓名、生日或電話號碼搜尋病人
    """
    # 組合查詢條件
    query = {}
    if name:
        query["name"] = name
    if day_of_birth:
        query["day_of_birth"] = datetime(day_of_birth.year, day_of_birth.month, day_of_birth.day)
    if phone:
        query["phone"] = phone

    # 查詢病人
    patients_list = []
    async for patient in collection_users.find(query):
        patient["_id"] = str(patient["_id"])  # 轉換 ObjectId
        patients_list.append(patient)

    return {"patients": patients_list}

# 醫生搜尋病人並新增關係
@app.post("/add_doctor_patient/")
async def add_doctor_patient(doctor_id: str, patient_id: str):
    """
    醫生搜尋病人並新增關係
    """
    # 確保 doctor_id 和 patient_id 是合法的 MongoDB ObjectId
    if not ObjectId.is_valid(doctor_id) or not ObjectId.is_valid(patient_id):
        raise HTTPException(status_code=400, detail="Invalid doctor_id or patient_id.")

    # 確保 doctor_id 和 patient_id 存在於資料庫
    doctor = await collection_users.find_one({"_id": ObjectId(doctor_id), "role": "medical_staff"})
    patient = await collection_users.find_one({"_id": ObjectId(patient_id), "role": "patient"})
    if not doctor or not patient:
        raise HTTPException(status_code=404, detail="The specified doctor or patient does not exist.")

    # 建立醫生與病人的關係
    doctor_patient_data = {
        "doctor_id": doctor_id,
        "patient_id": patient_id,
        "assigned_date": datetime.now(tz=dt.timezone(dt.timedelta(hours=8))),
    }
    result = await collection_doctor_patient.insert_one(doctor_patient_data)

    return {"inserted_id": str(result.inserted_id)}


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
        raise HTTPException(status_code=400, detail="Invalid user_id")

    user = await collection_users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User does not exist.")

    # 轉換更新資料為字典
    update_dict = dict(update_data)

    # 更新時間戳
    update_dict["updated_at"] = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))

    # 執行更新
    await collection_users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": update_dict}
    )

    return {"message": "User information updated."}

# Pydantic Model: 修改密碼請求
class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=8)  # 舊密碼
    new_password: str = Field(..., min_length=8)  # 新密碼
    confirm_new_password: str = Field(..., min_length=8)  # 確認新密碼

    @classmethod
    def validate_password(cls, value):
        """檢查密碼是否符合強度要求"""
        if not re.search(r"[a-z]", value):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"[A-Z]", value):
            raise ValueError("passwords must contain at least one upper case letter.")
        if not re.search(r"\d", value):
            raise ValueError("Password must contain at least one number.")
        return value


# 修改密碼 API
@app.put("/change_password/{user_id}/")
async def change_password(user_id: str, request: ChangePasswordRequest):
    """ 使用者變更密碼 """

    # 確保 user_id 是合法的 MongoDB ObjectId
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")

    user = await collection_users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User does not exist.")

    # 檢查舊密碼是否正確
    if not verify_password(request.old_password, user["password_hash"]):
        raise HTTPException(status_code=403, detail="The old password is incorrect.")

    # 檢查新密碼與確認密碼是否相符
    if request.new_password != request.confirm_new_password:
        raise HTTPException(status_code=400, detail="The new password and the confirmed password do not match.")

    # 檢查密碼強度
    ChangePasswordRequest.validate_password(request.new_password)

    # 加密新密碼並更新
    new_password_hash = hash_password(request.new_password)
    await collection_users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_hash": new_password_hash, "updated_at": datetime.now()}}
    )

    return {"message": "Password updated successfully."}

class LoginRequest(BaseModel):
    account: str
    pw_login: str
    role: str

# 使用者登入
@app.post("/login/")
async def login(request: LoginRequest):
    """
    使用者登入
    """
    user = await collection_users.find_one({"account": request.account})
    if not user:
        raise HTTPException(status_code=404, detail="User does not exist.")

    # 驗證密碼
    if not verify_password(request.pw_login, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Wrong password")

    # 驗證角色
    if user["role"] != request.role:
        error_message = f"Not {request.role}. Please use the correct login channel."
        raise HTTPException(status_code=400, detail=error_message)
    
    user_id = str(user["_id"])
     # 產生 JWT Token
    token = create_access_token({"sub": user_id, "role": request.role})

    return {"access_token": token, "token_type": "bearer"}

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
        raise HTTPException(status_code=400, detail="Invalid wound_id")

    wound = await collection_wound_records.find_one({"_id": ObjectId(wound_id)})
    if not wound:
        raise HTTPException(status_code=404, detail="Wound record does not exist.")

    # 轉換更新資料為字典
    update_dict = dict(update_data)

    # 更新時間戳
    update_dict["updated_at"] = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))

    # 執行更新
    await collection_wound_records.update_one(
        {"_id": ObjectId(wound_id)},
        {"$set": update_dict}
    )

    return {"message": "Wound record updated."}

# 儲存 ML 預測結果
"""
{
    "wound_id": "67da78f1bd148b1bd6b8d866",
    "model_version": "v2",
    "predicted_class": "D",
    "predicted_severity": "W1",
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
    # 先建立一個空的修改欄位
    prediction_data["corrected_by"] = None  # 醫生 ID
    prediction_data["corrected_class"] = None  # 醫生更正的傷口類別
    prediction_data["corrected_severity"] = None  # 醫生更正的傷口嚴重程度
    prediction_data["corrected_treatment_suggestions"] = None
    prediction_data["corrected_date"] = None  # 醫生更正的時間

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
# 更改Wound_records的內容
@app.put("/update_wound_records/")
async def update_wound_records(wound_id: str, update_data: dict):
    """ 更新傷口紀錄 """
    # 確保 wound_id 是合法的 MongoDB ObjectId
    if not ObjectId.is_valid(wound_id):
        raise HTTPException(status_code=400, detail="Invalid wound_id")

    wound = await collection_wound_records.find_one({"_id": ObjectId(wound_id)})
    if not wound:
        raise HTTPException(status_code=404, detail="Wound record does not exist.")

    # 轉換更新資料為字典
    update_dict = dict(update_data)

    # 更新時間戳
    update_dict["updated_at"] = datetime.now(tz=dt.timezone(dt.timedelta(hours=8)))

    # 執行更新
    await collection_wound_records.update_one(
        {"_id": ObjectId(wound_id)},
        {"$set": update_dict}
    )

    return {"message": "Wound record updated."}

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
        raise HTTPException(status_code=404, detail="No patients found for this doctor.")

    # 轉換 patient_id 為 ObjectId
    patient_object_ids = [ObjectId(patient_id) for patient_id in patient_ids]
    # 2. 透過 patient_id 查詢病患名字
    patients = await collection_users.find(
        {"_id": {"$in": patient_object_ids}}, {"_id": 0, "name": 1}
    ).to_list(length=100)
    # 3. 加入病患 ID
    for i in range(len(patients)):
        patients[i]["patient_id"] = patient_ids[i]
        
    return {"patients": patients}

# 根據病患ID，獲取所有傷口記錄和其ML預測分類和分級結果（不包含圖片）
@app.get("/get_wound_list/{patient_id}")
async def get_wound_list(patient_id: str):
    """
    根據病患ID，獲取所有傷口記錄和其ML預測分類和分級結果（不包含圖片）
    """
    # 1. 查詢病患的所有傷口記錄
    records = await collection_wound_records.find({"patient_id": patient_id}, {"_id": 1, "title": 1, "created_at":1, "wound_location":1}).to_list(length=100)
    if not records:
        raise HTTPException(status_code=404, detail="No wound records found for this patient.")
    
    # 將 records 轉成 DataFrame
    df_records = pd.DataFrame(records)
    # 將 _id 轉成字串（因為 MongoDB 的 ObjectId 不是 JSON serializable）
    df_records["_id"] = df_records["_id"].astype(str)
    # created_at 轉成字串格式
    df_records["date"] = df_records["created_at"].dt.strftime("%Y/%m/%d")
    # drop created_at 欄位
    df_records.drop(columns=["created_at"], inplace=True)

    # 2. 透過 wound_id 查詢 ML 預測結果
    ml_predictions = await collection_ml_predictions.find(
        {"wound_id": {"$in": df_records["_id"].tolist()}},
        {
            "_id": 0,
            "wound_id": 1,
            "class": {"$ifNull": ["$corrected_class", "$predicted_class"]},
            "severity": {"$ifNull": ["$corrected_severity", "$predicted_severity"]},
        }
    ).to_list(length=100)

    df_predictions = pd.DataFrame(ml_predictions)

    # 將 _id 改名為 wound_id，方便合併
    df_records.rename(columns={"_id": "wound_id"}, inplace=True)

    # 合併紀錄與預測結果
    df_merged = pd.merge(df_records, df_predictions, on="wound_id", how="left")


    return {"data": df_merged.to_dict(orient="records")}

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

