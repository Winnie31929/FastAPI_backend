
#####只對傷口區域進行 #####

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans

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

# 讀取影像
image_path = "./photo/test_images/foot-ulcer-0028_dataset.png"  # 影像路徑
image = cv2.imread(image_path)
mask = get_wound_mask(image)


# 顯示結果
cv2.imshow("Wound Mask", mask * 255)  # 乘 255 讓掩碼變白色
cv2.waitKey(0)
cv2.destroyAllWindows()

#####可視化-左側是傷口原圖，右側是傷口mask位置 #####
import matplotlib.pyplot as plt
mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
plt.figure(figsize=(6,6))
plt.subplot(1,2,1)
plt.imshow(image)  # 原圖
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")  # 預測的傷口 mask
plt.axis("off")

plt.show()


# 傷口掩碼 (來自 UNet / YOLO)

mask = get_wound_mask(image)
mask = (mask > 0.5).astype(np.uint8)  # 轉為 0-1 格式

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
    severity_map = np.zeros_like(mask, dtype=np.uint8)
    severity_map[wound_pixels[0], wound_pixels[1]] = labels + 1  # 從 1 開始，0 當背景

    # 顯示群中心（可用來理解每類代表什麼）
    print("HSV 群中心 (H, S, V)：\n", kmeans.cluster_centers_)

    return severity_map

##### 傷口嚴重程度的顏色對應 #####

def visualize_classification(image, severity_map):
    classified_image = image.copy()

    # 確保 severity_map 尺寸正確
    if severity_map.shape[:2] != image.shape[:2]:
        raise ValueError("severity_map shape does not match image shape")

    # 繪製分類結果（假設不同分類用不同顏色）
    classified_image[severity_map == 1] = [0, 255, 0]  # 綠色
    classified_image[severity_map == 2] = [255, 0, 0]  # 藍色
    classified_image[severity_map == 3] = [0, 255, 255]    # 黃色

    return classified_image

# 假設你的影像是 RGB (H, W, 3)
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 執行分類
severity_map = classify_wound_severity(image, mask, num_clusters=3)
visual_result = +visualize_classification(image_rgb, severity_map)

###### 顯示-三張圖(左-傷口原圖 中-mask 右-嚴重程度分類結果) 但還沒框出 #####
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(visual_result)
plt.title("location")
plt.axis("off")

plt.tight_layout()
plt.show()

###疊合原圖後框出分類結果#####

severity_colors = {
    1: (0, 0, 255),    # W2 - 中度 (紅色)
    2: (255, 0, 0),    # W1 - 輕微 (藍色)
    3: (0, 255, 0),  # W0 - 快癒合 (綠色)
}

overlay = image.copy()
for severity, color in severity_colors.items():
    mask = (severity_map == severity).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness=1)

# 讓邊框疊合原圖後，原圖顏色正常
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6,6))
plt.imshow(overlay_rgb)  # 用轉換後的圖
plt.axis("off")
plt.show()
