import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import sys
sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor
from ultralytics import YOLO

def show_mask(mask, ax, random_color=False):
    """Display segmentation mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax, label=""):
    """Display detection bounding box"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    if label:
        ax.text(x0, y0-5, label, color='green', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# ========== 1. Initialize YOLO Model ==========
print("Loading YOLO model...")
yolo_model = YOLO('yolov8n.pt')  # YOLOv8 nano model (fastest)
print("YOLO model loaded")

# ========== 2. Initialize TinySAM Model ==========
print("Loading TinySAM model...")
model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="./weights/tinysam_42.3.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
predictor = SamPredictor(sam)
print(f"TinySAM model loaded on {device}")

# ========== 3. Load and Process Image ==========
image = cv2.imread('fig/picture2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image shape: {image.shape}")

# ========== 4. Run YOLO Detection ==========
print("\nRunning YOLO detection...")
results = yolo_model(image, conf=0.25, verbose=False)  # Lower confidence threshold

# Extract detection results
boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
confs = results[0].boxes.conf.cpu().numpy()  # confidence scores
clses = results[0].boxes.cls.cpu().numpy().astype(int)  # class IDs
names = yolo_model.names  # class names dictionary

print(f"Detected {len(boxes)} objects")

# ========== 5. Use TinySAM to Generate Segmentation Masks ==========
print("\nGenerating segmentation masks with TinySAM...")
predictor.set_image(image)

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image + YOLO Detections")
plt.axis('off')

# Display YOLO detection boxes
ax1 = plt.gca()
for box, conf, cls_id in zip(boxes, confs, clses):
    cls_name = names[cls_id]
    show_box(box, ax1, f"{cls_name} {conf:.2f}")

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.title("TinySAM Segmentation Results")
plt.axis('off')

ax2 = plt.gca()

# Generate segmentation masks for each YOLO detection
for box, conf, cls_id in zip(boxes, confs, clses):
    cls_name = names[cls_id]
    
    # Use TinySAM to predict segmentation mask
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
    )
    
    # Select the mask with highest IoU score
    best_mask_idx = np.argmax(scores)
    print(f"  {cls_name}: Segmentation completed (confidence: {conf:.2f}, IoU score: {scores[best_mask_idx]:.3f})")
    
    # Display segmentation result
    show_mask(masks[best_mask_idx], ax2, random_color=True)
    show_box(box, ax2, f"{cls_name} {conf:.2f}")

plt.tight_layout()
plt.savefig("demo_yolo_tinysam_output.png", dpi=150, bbox_inches='tight')
print("\nResults saved to: demo_yolo_tinysam_output.png")

# ========== 6. Usage Instructions ==========
print("\n" + "="*60)
print("YOLO + TinySAM Integration Summary:")
print("="*60)
print("""
This demo shows the complete pipeline:
1. YOLO detects objects and generates bounding boxes
2. TinySAM refines each detection into precise segmentation masks
3. Results combine fast detection with accurate segmentation

Available YOLO models:
- yolov8n.pt (nano - fastest)
- yolov8s.pt (small)
- yolov8m.pt (medium)
- yolov8l.pt (large)
- yolov8x.pt (extra large - most accurate)

This approach is ideal for:
- Instance segmentation tasks
- Real-time video processing
- Autonomous driving
- Medical image analysis
- Industrial quality inspection
""")
print("="*60)

