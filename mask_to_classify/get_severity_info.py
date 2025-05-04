import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 得到傷口區域的 mask
def get_wound_mask(image):
    # 載入預訓練模型
    weight_file_name = "2025-01-22_21-26-41.hdf5"  
    # 載入模型（不編譯）
    model = load_model('./woundSeverity/' + weight_file_name, compile=False)
    # 影像預處理 (縮放到模型輸入大小)
    input_image = cv2.resize(image, (256, 256))  # 根據你的模型大小調整
    input_image = input_image / 255.0  # 正規化
    input_image = np.expand_dims(input_image, axis=0)  # 增加 batch 維度

    # 讓模型預測傷口區域
    mask = model.predict(input_image)[0, :, :, 0]  # 取出 mask
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # 轉回原圖大小

    # 轉為二值圖：將機率 > 0.5 的區域視為傷口
    binary_mask = (mask > 0.3).astype(np.uint8)

    return binary_mask


##### 使用 HSV+KMeans分群 來分類傷口嚴重程度 #####

def classify_wound_severity(image, mask, num_clusters=3):
    # 轉 HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 取得傷口區域索引
    wound_pixels = np.where(mask == 1)
    if len(wound_pixels[0]) == 0:
        print("⚠️ 沒偵測到傷口，回傳全 0 分群圖")
        return np.zeros_like(mask, dtype=np.uint8)

    # 擷取傷口區域 HSV 值
    wound_hsv = hsv_image[wound_pixels]

    # KMeans 分群
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(wound_hsv)

    # 建立分群結果圖
    #severity_map = np.zeros_like(mask, dtype=np.uint8)
    #severity_map[wound_pixels[0], wound_pixels[1]] = labels + 1  # 從 1 開始，0 當背景

    # ##### 改動：根據 V 值排序 #####
    centers = kmeans.cluster_centers_
    sorted_indices = np.argsort(centers[:, 2])  # 按 V（亮度）排序
    label_to_severity = {old_label: severity + 1 for severity, old_label in enumerate(sorted_indices)}

    # 建立分群結果圖
    severity_map = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(len(wound_pixels[0])):
        i, j = wound_pixels[0][idx], wound_pixels[1][idx]
        original_label = labels[idx]
        severity_map[i, j] = label_to_severity[original_label]
    # ##### 改動結束 #####

    # 顯示群中心（可用來理解每類代表什麼）
    print("HSV 群中心 (H, S, V)：\n", kmeans.cluster_centers_)

    return severity_map

##### 傷口嚴重程度的顏色對應 #####

def visualize_classification(image, severity_map):
    classified_image = image.copy()

    # 確保 severity_map 尺寸正確
    if severity_map.shape[:2] != image.shape[:2]:
        raise ValueError("severity_map shape does not match image shape")

    # 繪製分類結果（假設不同分類用不同顏色）(RGB)
    classified_image[severity_map == 1] = [0, 0, 255]  # 藍色(最嚴重)
    classified_image[severity_map == 2] = [255, 255, 0]  # 黃色(中度)
    classified_image[severity_map == 3] = [255, 0, 0]    # 紅色(輕微)

    return classified_image
"""
# 讀取影像
image_path = "./photo/test_images/foot-ulcer-0028_dataset.png"  # 影像路徑
image = cv2.imread(image_path)
mask = get_wound_mask(image) # 取得傷口區域的 mask
mask = (mask > 0.5).astype(np.uint8)  # 轉為 0-1 格式

# 假設你的影像是 RGB (H, W, 3)
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 執行分類
severity_map = classify_wound_severity(image, mask, num_clusters=3)
visual_result = +visualize_classification(image_rgb, severity_map)

# 依一開始的顏色(BGR)
severity_colors = {
    1: (255, 0, 0),     # W3 - 最嚴重(藍色)
    2: (0, 255, 255),   # W2 - 中度(黃色)
    3: (0, 0, 255),     # W1 - 輕微(紅色)
}
overlay = image.copy()
for severity, color in severity_colors.items():
    mask = (severity_map == severity).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=1)

# 讓邊框疊合原圖後，原圖顏色正常
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# 畫圖

plt.figure(figsize=(6,6))
plt.imshow(overlay_rgb)  # 用轉換後的圖
plt.axis("off")
plt.show()
"""
# 需要回傳給前端的資料：1.有什麼嚴重程度w1、w2、w3 2.疊合後的圖
# 封裝成函式
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64

def analyze_wound(image: np.ndarray):
    # 取得 mask
    mask = get_wound_mask(image)
    mask = (mask > 0.5).astype(np.uint8)

    # 轉成 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 分類傷口嚴重程度
    severity_map = classify_wound_severity(image, mask, num_clusters=3)

    # 標示傷口分類結果
    visual_result = visualize_classification(image_rgb.copy(), severity_map)

    # 畫輪廓用 BGR（再轉成 RGB 顯示）
    severity_colors = {
        1: (255, 0, 0),     # W3 - 最嚴重(藍色)
        2: (0, 255, 255),   # W2 - 中度(黃色)
        3: (0, 0, 255),     # W1 - 輕微(紅色)
    }
    overlay = image.copy()
    for severity, color in severity_colors.items():
        mask = (severity_map == severity).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # 將圖轉為 base64 傳回
    pil_img = Image.fromarray(overlay_rgb)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 判斷有哪些嚴重程度存在
    present_severities = sorted(list(set(np.unique(severity_map)) - {0}))  # 排除背景

    severity_labels = {
        1: "W3",  # 最嚴重
        2: "W2",  # 中度
        3: "W1",  # 輕微
    }

    present_labels = [severity_labels[s] for s in present_severities if s in severity_labels]

    return {
        "present_severities": present_labels,
        "image_base64": img_str
    }
