import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
        ax.text(x0, y0-5, label, color='green', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

def show_points(coords, labels, ax, marker_size=100):
    """Display sampling points"""
    pos_points = coords[labels==1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.0, zorder=10)

def generate_points_from_box(box, num_points=9):
    """Generate grid sampling points inside bounding box"""
    x1, y1, x2, y2 = box
    grid_size = int(np.sqrt(num_points))
    margin = 0.1
    x_margin = (x2 - x1) * margin
    y_margin = (y2 - y1) * margin
    
    x_coords = np.linspace(x1 + x_margin, x2 - x_margin, grid_size)
    y_coords = np.linspace(y1 + y_margin, y2 - y_margin, grid_size)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            points.append([x, y])
    
    return np.array(points)

def generate_dense_points_outside_boxes(image_shape, boxes, points_per_side=6):
    """Generate dense sampling points outside YOLO boxes"""
    h, w = image_shape[:2]
    x_coords = np.linspace(0, w, points_per_side)
    y_coords = np.linspace(0, h, points_per_side)
    
    all_points = []
    for x in x_coords:
        for y in y_coords:
            inside_box = False
            for box in boxes:
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    inside_box = True
                    break
            if not inside_box:
                all_points.append([x, y])
    
    return np.array(all_points) if len(all_points) > 0 else np.empty((0, 2))

# Initialize models
print("Loading YOLO model...")
yolo_model = YOLO('yolov8n.pt')
print("Loading TinySAM model...")
model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint="./weights/tinysam_42.3.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
predictor = SamPredictor(sam)
print(f"Models loaded on {device}")

# Load image
image = cv2.imread('fig/picture2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image shape: {image.shape}")

# Run YOLO detection
print("\nRunning YOLO detection...")
results = yolo_model(image, conf=0.25, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()
confs = results[0].boxes.conf.cpu().numpy()
clses = results[0].boxes.cls.cpu().numpy().astype(int)
names = yolo_model.names
print(f"Detected {len(boxes)} objects")

# Set image for TinySAM
predictor.set_image(image)

# Generate and segment high-confidence regions
print("\n=== High-Confidence Segmentation (YOLO boxes) ===")
high_conf_masks = []
for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, clses)):
    cls_name = names[cls_id]
    points = generate_points_from_box(box, num_points=9)
    labels = np.ones(len(points))
    
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels)
    best_mask = masks[np.argmax(scores)]
    high_conf_masks.append(best_mask)
    print(f"  {cls_name} (#{idx+1}): Done (conf: {conf:.2f})")

# Create combined high-confidence mask
combined_high_conf_mask = np.zeros_like(high_conf_masks[0], dtype=bool) if len(high_conf_masks) > 0 else None
if combined_high_conf_mask is not None:
    for mask in high_conf_masks:
        combined_high_conf_mask = combined_high_conf_mask | mask

# Generate low-confidence points
print("\n=== Low-Confidence Segmentation (Outside boxes) ===")
low_conf_points = generate_dense_points_outside_boxes(image.shape, boxes, points_per_side=6)
print(f"Generated {len(low_conf_points)} low-confidence points")

low_conf_masks = []
if len(low_conf_points) > 0:
    # Process only a few points to demonstrate the concept
    sample_points = low_conf_points[:min(3, len(low_conf_points))]  # Only process 3 points
    for idx, point in enumerate(sample_points):
        try:
            masks, scores, _ = predictor.predict(
                point_coords=point.reshape(1, 2),
                point_labels=np.ones(1)
            )
            best_mask = masks[np.argmax(scores)]
            
            # Check overlap
            if combined_high_conf_mask is not None:
                overlap = np.sum(best_mask & combined_high_conf_mask) / (np.sum(best_mask) + 1e-6)
                if overlap < 0.5 and np.sum(best_mask) > 100:
                    low_conf_masks.append(best_mask)
                    print(f"  Point {idx+1}: Segmented (overlap: {overlap:.2%})")
        except:
            pass

print(f"Generated {len(low_conf_masks)} low-confidence masks")

# Create visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Plot 1: YOLO detections
axes[0].imshow(image)
axes[0].set_title(f"Step 1: YOLO Detection ({len(boxes)} objects)", fontsize=12, weight='bold')
axes[0].axis('off')
for box, conf, cls_id in zip(boxes, confs, clses):
    show_box(box, axes[0], f"{names[cls_id]} {conf:.2f}")

# Plot 2: High-confidence points
axes[1].imshow(image)
axes[1].set_title("Step 2a: High-Confidence Points (in boxes)", fontsize=12, weight='bold')
axes[1].axis('off')
for box, conf, cls_id in zip(boxes, confs, clses):
    show_box(box, axes[1])
    points = generate_points_from_box(box, num_points=9)
    show_points(points, np.ones(len(points)), axes[1], marker_size=60)

# Plot 3: Low-confidence points
axes[2].imshow(image)
axes[2].set_title(f"Step 2b: Low-Confidence Points ({len(low_conf_points)} points)", fontsize=12, weight='bold')
axes[2].axis('off')
for box in boxes:
    show_box(box, axes[2])
if len(low_conf_points) > 0:
    show_points(low_conf_points, np.ones(len(low_conf_points)), axes[2], marker_size=20)

# Plot 4: Combined results
axes[3].imshow(image)
axes[3].set_title(f"Step 3: Combined Masks ({len(high_conf_masks)}H + {len(low_conf_masks)}L)", 
                  fontsize=12, weight='bold')
axes[3].axis('off')
for mask in high_conf_masks:
    show_mask(mask, axes[3], random_color=True)
for mask in low_conf_masks:
    show_mask(mask, axes[3], random_color=True)
for box, cls_id in zip(boxes, clses):
    show_box(box, axes[3], names[cls_id])

plt.tight_layout()
output_path = "demo_yolo_hierarchical_simple_output.png"
plt.savefig(output_path, dpi=120)
plt.close('all')
print(f"\nResults saved to: {output_path}")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print(f"""
Pipeline demonstrated:
1. YOLO Detection: {len(boxes)} objects detected
2. High-Confidence Segmentation: {len(high_conf_masks)} masks from YOLO boxes
3. Low-Confidence Sampling: {len(low_conf_points)} points outside boxes
4. Low-Confidence Segmentation: {len(low_conf_masks)} additional masks
5. Total masks: {len(high_conf_masks) + len(low_conf_masks)}

This approach combines:
✓ Fast YOLO detection for main objects
✓ Point-based prompts for precise segmentation
✓ Dense sampling for complete coverage
✓ Overlap filtering to avoid redundancy

Model: YOLOv8n (nano) + TinySAM (vit_t) - both lightweight
""")
print("="*70)


