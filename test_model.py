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

# Example usage
if __name__ == "__main__":
    image_path = "./photo/test_images/foot-ulcer-0028_dataset.png"  
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"⚠️ 圖片讀取失敗，請確認路徑是否正確：{image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 預測傷口區域
    wound_mask = get_wound_mask(image)
    plt.imshow(wound_mask, cmap='gray')
    plt.axis('off')
    plt.show()

    # 儲存傷口區域
    cv2.imwrite("wound_mask_dataset.png", wound_mask * 255)  # 儲存為二值圖
    