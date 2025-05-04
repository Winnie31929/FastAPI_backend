from datetime import datetime, timezone, timedelta
import datetime as dt

# 模擬你從資料庫抓出來的 UTC 字串
utc_time_str = "2025-05-04T02:39:54.338Z"

# 先轉成 datetime（注意：要去掉 Z，並設定為 UTC 時區）
utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))

# 轉換為台灣時間（UTC+8）
tw_timezone = timezone(timedelta(hours=8))
tw_time = utc_time.astimezone(tw_timezone)
print(tw_time)
print("台灣時間：", tw_time.strftime("%Y/%m/%d"))  # ➜ 2025-05-04T10:39:54.338+08:00

    