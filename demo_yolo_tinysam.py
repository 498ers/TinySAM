import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import sys
sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    """显示分割掩码"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax, label=""):
    """显示检测框"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    if label:
        ax.text(x0, y0-5, label, color='green', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# ========== 1. 初始化 TinySAM 模型 ==========
print("Loading TinySAM model...")
model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="./weights/tinysam_42.3.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
predictor = SamPredictor(sam)
print(f"Model loaded on {device}")

# ========== 2. 读取图片 ==========
image = cv2.imread('fig/picture1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
print(f"Image shape: {image.shape}")

# ========== 3. 模拟 YOLO 检测结果 ==========
# YOLO 输出格式: [x_center, y_center, width, height, confidence, class_id]
# 这里我们模拟几个检测框
print("\n模拟 YOLO 检测结果...")

# 模拟的 YOLO 检测框 (格式: [x1, y1, x2, y2, confidence, class_id])
# 在实际应用中，这些框来自 YOLO 的检测结果
yolo_detections = [
    # [x1, y1, x2, y2, confidence, class_id, class_name]
    [100, 100, 500, 400, 0.95, 0, "dog"],      # 狗
    [300, 50, 600, 300, 0.88, 16, "person"],   # 人
]

print(f"检测到 {len(yolo_detections)} 个目标")

# ========== 4. 使用 TinySAM 对每个检测框生成分割掩码 ==========
print("\n使用 TinySAM 生成分割掩码...")

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("原始图片 + YOLO 检测框")
plt.axis('off')

# 显示 YOLO 检测框
ax1 = plt.gca()
for det in yolo_detections:
    x1, y1, x2, y2, conf, cls_id, cls_name = det
    box = np.array([x1, y1, x2, y2])
    show_box(box, ax1, f"{cls_name} {conf:.2f}")

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.title("TinySAM 分割结果")
plt.axis('off')

ax2 = plt.gca()

# 对每个 YOLO 检测框进行分割
for i, det in enumerate(yolo_detections):
    x1, y1, x2, y2, conf, cls_id, cls_name = det
    
    # 将检测框转换为 TinySAM 输入格式
    input_box = np.array([x1, y1, x2, y2])
    
    # 使用 TinySAM 预测分割掩码
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
    )
    
    # 选择得分最高的掩码
    best_mask_idx = np.argmax(scores)
    print(f"  {cls_name}: 分割完成 (confidence: {conf:.2f}, IoU score: {scores[best_mask_idx]:.3f})")
    
    # 显示分割结果
    show_mask(masks[best_mask_idx], ax2, random_color=True)
    show_box(input_box, ax2, f"{cls_name} {conf:.2f}")

plt.tight_layout()
plt.savefig("demo_yolo_tinysam_output.png", dpi=150, bbox_inches='tight')
print("\n结果已保存到: demo_yolo_tinysam_output.png")

# ========== 5. 展示如何整合真实的 YOLO ==========
print("\n" + "="*60)
print("如何整合真实的 YOLO 模型:")
print("="*60)
print("""
# 安装 ultralytics (YOLOv8)
pip install ultralytics

# 使用 YOLO + TinySAM 的完整流程:
from ultralytics import YOLO

# 1. 加载 YOLO 模型
yolo_model = YOLO('yolov8n.pt')  # 或 yolov8s.pt, yolov8m.pt 等

# 2. YOLO 检测
results = yolo_model(image)

# 3. 提取检测框
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confs = result.boxes.conf.cpu().numpy()  # confidence
    clses = result.boxes.cls.cpu().numpy()   # class_id
    
    # 4. 对每个框使用 TinySAM 分割
    for box, conf, cls in zip(boxes, confs, clses):
        masks, scores, _ = predictor.predict(
            box=box[None, :]
        )
        best_mask = masks[np.argmax(scores)]
        # 使用生成的掩码...
""")
print("="*60)

