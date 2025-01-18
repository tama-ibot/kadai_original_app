##テンプレートマッチングの機能

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ロゴ（テンプレート）画像と検索対象の画像を指定
image_path = "/content/ロゴを使用したクリエイティブ例.jpg"  # 検索対象の画像
logo_path = "/content/sample_logo_snakes.jpeg"    # ロゴ（テンプレート）画像

# グレースケールで画像を読み込む
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)

# テンプレート画像をリサイズする
# 検索対象の画像のサイズを取得
image_height, image_width = image.shape

# テンプレート画像が検索対象の画像よりも大きい場合、リサイズする
if logo.shape[0] > image_height or logo.shape[1] > image_width:
    # リサイズ後のサイズを計算 (検索対象の画像の半分に縮小)
    resize_height = int(image_height / 2)
    resize_width = int(image_width / 2)

    # リサイズを実行
    logo = cv2.resize(logo, (resize_width, resize_height))

# ロゴ画像の高さと幅を取得
h, w = logo.shape

# テンプレートマッチングを実行
result = cv2.matchTemplate(image, logo, cv2.TM_CCOEFF_NORMED)

# 最大値とその位置を取得
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 一致した領域を取得
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# 元画像に一致領域を描画
matched_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)

# 結果を表示
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Matched Result")
plt.imshow(matched_image)
plt.axis("off")
plt.show()

# マッチング結果を表示
print(f"テンプレートの一致度: {max_val:.2f}")