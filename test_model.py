import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def classify_wound_by_rgb(image, mask):
    """
    根據實際傷口區域 RGB 比例分類 W1:紅、W2:黃、W3:黑
    """
    mask = (mask > 0.5).astype(np.uint8)
    wound_pixels = np.where(mask == 1)

    if len(wound_pixels[0]) == 0:
        print("⚠️ 沒有偵測到傷口，回傳全零的分類圖")
        return np.zeros_like(mask, dtype=np.uint8), wound_pixels

    wound_rgb = image[wound_pixels]
    R = wound_rgb[:, 0].astype(np.float32)
    G = wound_rgb[:, 1].astype(np.float32)
    B = wound_rgb[:, 2].astype(np.float32)
    total = R + G + B + 1e-6

    r_ratio = R / total
    g_ratio = G / total
    b_ratio = B / total

    severity_map = np.zeros_like(mask, dtype=np.uint8)

    # 新分類條件（根據你實際的色彩比例）
    w1 = (r_ratio > 0.31) & (g_ratio > 0.27) & (b_ratio < 0.39)
    w2 = (r_ratio > 0.30) & (g_ratio > 0.30) & (b_ratio < 0.36)
    w3 = (r_ratio < 0.26) & (g_ratio < 0.26) & (b_ratio < 0.38)

    severity_map[wound_pixels[0][w1], wound_pixels[1][w1]] = 1  # W1 紅
    severity_map[wound_pixels[0][w2], wound_pixels[1][w2]] = 2  # W2 黃
    severity_map[wound_pixels[0][w3], wound_pixels[1][w3]] = 3  # W3 黑

    return severity_map, wound_pixels

import os

def process_wound_image(image_path: str, save_overlay_path: str ):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"⚠️ 圖片讀取失敗，請確認路徑是否正確：{image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = get_wound_mask(image)
    severity_map, wound_pixels = classify_wound_by_rgb(image, mask)
    #severity_map, wound_pixels = classify_wound_severity(image, mask, num_clusters=3)

    colored_severity = cm.get_cmap('jet')(severity_map / 3.0)[..., :3]
    colored_severity = (colored_severity * 255).astype(np.uint8)

    alpha = 0.5
    overlay = (image * (1 - alpha) + colored_severity * alpha).astype(np.uint8)

    # 自動確保資料夾存在
    save_dir = os.path.dirname(save_overlay_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_overlay_path, overlay_bgr)

    return {
        "classes_in_result": np.unique(severity_map).tolist(),
        "classes_in_mask": np.unique(mask).tolist(),
        "wound_pixel_count": int(len(wound_pixels[0])),
        "overlay_path": save_overlay_path
    }
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

    return severity_map, wound_pixels

def visualize_classification(image, severity_map):
    classified_image = image.copy()

    # 確保 severity_map 尺寸正確
    if severity_map.shape[:2] != image.shape[:2]:
        raise ValueError("severity_map shape does not match image shape")

    # 繪製分類結果（假設不同分類用不同顏色）
    classified_image[severity_map == 1] = [0, 255, 0]  # 綠色
    classified_image[severity_map == 2] = [255, 0, 0]  # 藍色
    classified_image[severity_map == 3] = [0, 255, 255]    # 黃色
    #classified_image[severity_map == 4] = [0, 0, 255]   #紅色


    return classified_image

# Example usage
if __name__ == "__main__":
    image_path = "./photo/test_images/diabetic_foot_ulcer_0028.jpg"  # Replace with your image path
    

    result = process_wound_image(image_path, "./photo/test_prediction/overlay_result3.png")
    print("Processing result:", result)

    # 顯示疊圖
    overlay_result = cv2.imread(result["overlay_path"])
    overlay_result = cv2.imread(result["overlay_path"])
    if overlay_result is None:
        raise ValueError(f"⚠️ 找不到疊圖檔案，檢查路徑：{result['overlay_path']}")

    plt.imshow(cv2.cvtColor(overlay_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()