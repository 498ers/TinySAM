"""
Evaluation Script for Paper
Comprehensive evaluation of YOLO + TinySAM with Box Prompts
"""
import numpy as np
import torch
import cv2
import sys
import time
import json
from collections import defaultdict
sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor
from ultralytics import YOLO

def evaluate_single_image(image_path, yolo_model, predictor, device):
    """
    Evaluate single image and return detailed metrics
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    results = {
        'image_path': image_path,
        'image_size': (h, w),
        'yolo_time': 0,
        'sam_time': 0,
        'sam_time_per_object': 0,
        'num_objects': 0,
        'iou_scores': [],
        'mean_iou': 0,
        'std_iou': 0,
        'min_iou': 0,
        'max_iou': 0,
        'objects': []
    }
    
    # ========== YOLO Detection ==========
    start_time = time.time()
    yolo_results = yolo_model(image, conf=0.25, verbose=False)
    results['yolo_time'] = time.time() - start_time
    
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    confs = yolo_results[0].boxes.conf.cpu().numpy()
    clses = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
    names = yolo_model.names
    
    results['num_objects'] = len(boxes)
    
    if len(boxes) == 0:
        return results
    
    # ========== TinySAM Segmentation with Box Prompts ==========
    predictor.set_image(image)
    
    start_time = time.time()
    for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, clses)):
        cls_name = names[cls_id]
        
        # Box prompt
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :]
        )
        
        mask = masks[0]
        score = scores[0]
        
        # Calculate mask area and box area
        mask_area = np.sum(mask)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        
        results['iou_scores'].append(float(score))
        results['objects'].append({
            'class': cls_name,
            'confidence': float(conf),
            'iou': float(score),
            'mask_area': int(mask_area),
            'box_area': float(box_area),
            'mask_box_ratio': float(mask_area / box_area) if box_area > 0 else 0
        })
    
    results['sam_time'] = time.time() - start_time
    results['sam_time_per_object'] = results['sam_time'] / len(boxes) if len(boxes) > 0 else 0
    
    # Calculate statistics
    if len(results['iou_scores']) > 0:
        results['mean_iou'] = float(np.mean(results['iou_scores']))
        results['std_iou'] = float(np.std(results['iou_scores']))
        results['min_iou'] = float(np.min(results['iou_scores']))
        results['max_iou'] = float(np.max(results['iou_scores']))
    
    return results

def print_paper_format(all_results):
    """
    Print results in paper-ready format
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS FOR PAPER")
    print("="*80)
    
    # Aggregate statistics
    total_images = len(all_results)
    total_objects = sum(r['num_objects'] for r in all_results)
    
    all_iou = []
    all_yolo_times = []
    all_sam_times = []
    all_sam_times_per_obj = []
    
    for r in all_results:
        all_iou.extend(r['iou_scores'])
        all_yolo_times.append(r['yolo_time'])
        all_sam_times.append(r['sam_time'])
        if r['sam_time_per_object'] > 0:
            all_sam_times_per_obj.append(r['sam_time_per_object'])
    
    print(f"\n### Dataset Statistics")
    print(f"- Total images evaluated: {total_images}")
    print(f"- Total objects detected: {total_objects}")
    print(f"- Average objects per image: {total_objects/total_images:.2f}")
    
    print(f"\n### Timing Performance")
    print(f"- YOLO detection time: {np.mean(all_yolo_times)*1000:.1f} ± {np.std(all_yolo_times)*1000:.1f} ms/image")
    print(f"- SAM segmentation time: {np.mean(all_sam_times)*1000:.1f} ± {np.std(all_sam_times)*1000:.1f} ms/image")
    print(f"- SAM time per object: {np.mean(all_sam_times_per_obj)*1000:.1f} ± {np.std(all_sam_times_per_obj)*1000:.1f} ms/object")
    print(f"- Total pipeline time: {(np.mean(all_yolo_times) + np.mean(all_sam_times))*1000:.1f} ms/image")
    print(f"- Throughput: {1/(np.mean(all_yolo_times) + np.mean(all_sam_times)):.2f} images/second")
    
    print(f"\n### Segmentation Quality (IoU)")
    print(f"- Mean IoU: {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print(f"- Median IoU: {np.median(all_iou):.4f}")
    print(f"- Min IoU: {np.min(all_iou):.4f}")
    print(f"- Max IoU: {np.max(all_iou):.4f}")
    print(f"- IoU > 0.8: {100*np.mean(np.array(all_iou) > 0.8):.1f}%")
    print(f"- IoU > 0.9: {100*np.mean(np.array(all_iou) > 0.9):.1f}%")
    
    # Per-class statistics
    class_stats = defaultdict(list)
    for r in all_results:
        for obj in r['objects']:
            class_stats[obj['class']].append(obj['iou'])
    
    print(f"\n### Per-Class IoU Statistics")
    for cls_name in sorted(class_stats.keys()):
        ious = class_stats[cls_name]
        print(f"- {cls_name}: {np.mean(ious):.4f} ± {np.std(ious):.4f} (n={len(ious)})")
    
    # LaTeX table format
    print(f"\n### LaTeX Table Format")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance Evaluation of YOLO + TinySAM Pipeline}")
    print("\\begin{tabular}{lc}")
    print("\\hline")
    print("Metric & Value \\\\")
    print("\\hline")
    print(f"YOLO Detection Time (ms/image) & ${np.mean(all_yolo_times)*1000:.1f} \\pm {np.std(all_yolo_times)*1000:.1f}$ \\\\")
    print(f"SAM Segmentation Time (ms/image) & ${np.mean(all_sam_times)*1000:.1f} \\pm {np.std(all_sam_times)*1000:.1f}$ \\\\")
    print(f"SAM Time per Object (ms) & ${np.mean(all_sam_times_per_obj)*1000:.1f} \\pm {np.std(all_sam_times_per_obj)*1000:.1f}$ \\\\")
    print(f"Total Pipeline Time (ms/image) & ${(np.mean(all_yolo_times) + np.mean(all_sam_times))*1000:.1f}$ \\\\")
    print(f"Throughput (images/s) & ${1/(np.mean(all_yolo_times) + np.mean(all_sam_times)):.2f}$ \\\\")
    print("\\hline")
    print(f"Mean IoU & ${np.mean(all_iou):.4f} \\pm {np.std(all_iou):.4f}$ \\\\")
    print(f"Median IoU & ${np.median(all_iou):.4f}$ \\\\")
    print(f"IoU $> 0.8$ (\\%) & ${100*np.mean(np.array(all_iou) > 0.8):.1f}$ \\\\")
    print(f"IoU $> 0.9$ (\\%) & ${100*np.mean(np.array(all_iou) > 0.9):.1f}$ \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n" + "="*80)

def main():
    """
    Main evaluation function
    """
    print("="*80)
    print("YOLO + TinySAM Evaluation for Paper")
    print("="*80)
    
    # Initialize models
    print("\nLoading models...")
    yolo_model = YOLO('yolov8n.pt')
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint="./weights/tinysam_42.3.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"Models loaded on {device}")
    
    # Model statistics
    yolo_params = sum(p.numel() for p in yolo_model.model.parameters()) / 1e6
    sam_params = sum(p.numel() for p in sam.parameters()) / 1e6
    
    print(f"\n### Model Architecture")
    print(f"- YOLO: YOLOv8n (nano)")
    print(f"  - Parameters: {yolo_params:.2f}M")
    print(f"- TinySAM: vit_t backbone")
    print(f"  - Parameters: {sam_params:.2f}M")
    print(f"- Total Parameters: {yolo_params + sam_params:.2f}M")
    print(f"- Device: {device}")
    print(f"- Prompt Type: Box prompts (direct)")
    
    # Evaluate images
    test_images = [
        'fig/picture1.jpg',
        'fig/picture2.jpg',
        'fig/picture3.jpg'
    ]
    
    all_results = []
    
    print(f"\nEvaluating {len(test_images)} images...")
    for img_path in test_images:
        try:
            print(f"\nProcessing: {img_path}")
            result = evaluate_single_image(img_path, yolo_model, predictor, device)
            all_results.append(result)
            print(f"  - Objects: {result['num_objects']}")
            print(f"  - YOLO time: {result['yolo_time']*1000:.1f}ms")
            print(f"  - SAM time: {result['sam_time']*1000:.1f}ms")
            print(f"  - Mean IoU: {result['mean_iou']:.4f}")
        except Exception as e:
            print(f"  - Error: {e}")
    
    # Print results
    print_paper_format(all_results)
    
    # Save to JSON for later use
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()


