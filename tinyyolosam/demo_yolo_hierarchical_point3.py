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
        color = np.concatenate([np.random.random(3), np.array([0.35])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.35])
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

def show_points(coords, labels, ax, marker_size=200):
    """Display sampling points"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25, zorder=10)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25, zorder=10)

def generate_points_from_box(box, num_points=9):
    """
    Generate grid sampling points inside bounding box
    These points are similar to high-confidence points in hierarchical everything
    
    Args:
        box: Bounding box in [x1, y1, x2, y2] format
        num_points: Number of sampling points (default 9 points, 3x3 grid)
    
    Returns:
        points: Point coordinates in [N, 2] format (x, y)
    """
    x1, y1, x2, y2 = box
    
    # Calculate grid size
    grid_size = int(np.sqrt(num_points))
    
    # Generate grid points inside box, avoiding edges
    margin = 0.1  # Leave 10% margin
    x_margin = (x2 - x1) * margin
    y_margin = (y2 - y1) * margin
    
    x_coords = np.linspace(x1 + x_margin, x2 - x_margin, grid_size)
    y_coords = np.linspace(y1 + y_margin, y2 - y_margin, grid_size)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            points.append([x, y])
    
    return np.array(points)

# ========== 1. Initialize YOLO Model ==========
print("Loading YOLO model...")
yolo_model = YOLO('yolov8n.pt')  # YOLOv8 nano - lightweight model
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
results = yolo_model(image, conf=0.25, verbose=False)

# Extract detection results
boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
confs = results[0].boxes.conf.cpu().numpy()  # confidence scores
clses = results[0].boxes.cls.cpu().numpy().astype(int)  # class IDs
names = yolo_model.names  # class names dictionary

print(f"Detected {len(boxes)} objects")

# ========== 5. Generate Sampling Points from YOLO Boxes ==========
print("\nGenerating sampling points from YOLO boxes (replacing hierarchical everything step 1)...")
predictor.set_image(image)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Subplot 1: YOLO detection results
axes[0].imshow(image)
axes[0].set_title("Step 1: YOLO Detection Results", fontsize=14, weight='bold')
axes[0].axis('off')
for box, conf, cls_id in zip(boxes, confs, clses):
    cls_name = names[cls_id]
    show_box(box, axes[0], f"{cls_name} {conf:.2f}")

# Subplot 2: Sampling points generated from boxes
axes[1].imshow(image)
axes[1].set_title("Step 2: Sampling Points in YOLO Boxes (Replace hierarchical high-conf points)", fontsize=14, weight='bold')
axes[1].axis('off')

all_points = []
all_labels = []
for box, conf, cls_id in zip(boxes, confs, clses):
    cls_name = names[cls_id]
    show_box(box, axes[1], f"{cls_name} {conf:.2f}")
    
    # Generate sampling points inside box (similar to hierarchical everything high-conf points)
    points = generate_points_from_box(box, num_points=9)  # 3x3 grid
    labels = np.ones(len(points))  # all positive points
    
    # Display sampling points
    show_points(points, labels, axes[1], marker_size=100)
    
    all_points.append(points)
    all_labels.append(labels)

# Subplot 3: TinySAM segmentation results
axes[2].imshow(image)
axes[2].set_title("Step 3: TinySAM Segmentation Results Based on Sampling Points", fontsize=14, weight='bold')
axes[2].axis('off')

# Generate segmentation masks using sampling points
print("\nGenerating segmentation masks using sampling points...")
all_masks = []

for idx, (box, points, labels, conf, cls_id) in enumerate(zip(boxes, all_points, all_labels, confs, clses)):
    cls_name = names[cls_id]
    
    # Use multiple sampling points as prompt
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
    )
    
    # Select mask with highest score
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    best_score = scores[best_mask_idx]
    
    print(f"  {cls_name} (#{idx+1}): Segmentation completed (confidence: {conf:.2f}, IoU score: {best_score:.3f}, sampling points: {len(points)})")
    
    # Display segmentation result
    show_mask(best_mask, axes[2], random_color=True)
    show_box(box, axes[2], f"{cls_name}")
    
    all_masks.append({
        'segmentation': best_mask,
        'area': best_mask.sum(),
        'bbox': box,
        'predicted_iou': best_score,
        'point_coords': points,
        'class_name': cls_name,
        'confidence': conf
    })

plt.tight_layout()
output_path = "demo_yolo_hierarchical_output.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nResults saved to: {output_path}")

# ========== 6. Summary ==========
print("\n" + "="*70)
print("YOLO + TinySAM Hierarchical Segmentation Demo Summary:")
print("="*70)
print(f"""
This demo demonstrates the complete pipeline:

1. YOLO Detection → Generate bounding boxes
   - Detected {len(boxes)} objects
   - Replaces the coarse detection step in hierarchical everything

2. In-box Sampling Point Generation → Replace hierarchical high-confidence points
   - Generated 3x3 grid points (9 points) per box
   - These points serve as TinySAM prompts
   - Multi-point input generates more precise masks than single box input

3. TinySAM Segmentation → Precise instance segmentation
   - Generated high-quality masks using multiple sampling points
   - Total masks generated: {len(all_masks)}

Advantages:
✓ Faster than pure hierarchical everything (YOLO detection is faster than SAM auto point generation)
✓ More precise than pure box prompts (multi-point prompts provide richer information)
✓ Combines advantages of both methods
✓ More controllable (adjustable sampling point quantity and position)

Parameters:
- num_points: Adjust number of points per box (4, 9, 16, 25...)
- conf: YOLO detection confidence threshold (current: 0.25)
- margin: Distance of sampling points from box edges (current: 10%)

Model Configuration:
- YOLO: YOLOv8n (nano) - lightweight and fast
- SAM: TinySAM with vit_t backbone - efficient segmentation
""")
print("="*70)

# Display image
plt.show()

