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
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.now(dt.timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


token = create_access_token({"sub": "testuser"}, timedelta(minutes=30))
decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
print(decoded["exp"])  # 這應該是 UTC 時間
