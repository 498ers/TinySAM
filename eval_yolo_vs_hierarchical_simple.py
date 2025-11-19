"""
Evaluation Script: YOLO vs Original Hierarchical Everything
Compares performance metrics (Speed, Accuracy, FPS-mAP curves)
Supports batch evaluation on COCO val2017 dataset

Configuration:
- NUM_TEST_IMAGES: Number of images to evaluate (default: 50)
- Hierarchical method is slow, so it's only evaluated on every 5th image
- Set NUM_TEST_IMAGES=10 for quick test, 100+ for full evaluation

Usage:
    python eval_yolo_vs_hierarchical.py
    
Output:
    - comparison_yolo_vs_hierarchical.png (6 comparison plots)
    - comparison_results.json (detailed metrics)
"""
import numpy as np
import torch
import cv2
import sys
import time
import json
import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor, SamHierarchicalMaskGenerator
from ultralytics import YOLO

def evaluate_yolo_method(image_path, yolo_model, predictor):
    """
    Evaluate YOLO + TinySAM method (our approach)
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = {
        'method': 'YOLO + TinySAM (Ours)',
        'image_path': image_path,
        'yolo_time': 0,
        'sam_time': 0,
        'total_time': 0,
        'fps': 0,
        'num_masks': 0,
        'mean_iou': 0,
        'iou_scores': [],
        'mask_areas': []
    }
    
    # YOLO Detection
    start_time = time.time()
    yolo_results = yolo_model(image, conf=0.25, verbose=False)
    results['yolo_time'] = time.time() - start_time
    
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        return results
    
    # TinySAM Segmentation with Box Prompts
    predictor.set_image(image)
    
    start_time = time.time()
    masks = []
    iou_scores = []
    
    for box in boxes:
        # Use BOX prompt
        pred_masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :]
        )
        best_mask = pred_masks[np.argmax(scores)]
        best_score = scores[np.argmax(scores)]
        
        masks.append(best_mask)
        iou_scores.append(float(best_score))
        results['mask_areas'].append(int(np.sum(best_mask)))
    
    results['sam_time'] = time.time() - start_time
    results['total_time'] = results['yolo_time'] + results['sam_time']
    results['fps'] = 1.0 / results['total_time'] if results['total_time'] > 0 else 0
    results['num_masks'] = len(masks)
    results['iou_scores'] = iou_scores
    results['mean_iou'] = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0
    
    return results

def evaluate_hierarchical_method(image_path, mask_generator):
    """
    Evaluate original Hierarchical Everything method
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = {
        'method': 'Hierarchical Everything (Original)',
        'image_path': image_path,
        'yolo_time': 0,  # N/A for this method
        'sam_time': 0,
        'total_time': 0,
        'fps': 0,
        'num_masks': 0,
        'mean_iou': 0,
        'iou_scores': [],
        'mask_areas': []
    }
    
    # Hierarchical mask generation
    start_time = time.time()
    try:
        masks = mask_generator.hierarchical_generate(image)
        results['sam_time'] = time.time() - start_time
        results['total_time'] = results['sam_time']
        results['fps'] = 1.0 / results['total_time'] if results['total_time'] > 0 else 0
        results['num_masks'] = len(masks)
        
        # Extract IoU scores and areas
        for mask_data in masks:
            if 'predicted_iou' in mask_data:
                results['iou_scores'].append(float(mask_data['predicted_iou']))
            results['mask_areas'].append(int(mask_data['area']))
        
        results['mean_iou'] = float(np.mean(results['iou_scores'])) if len(results['iou_scores']) > 0 else 0
    except Exception as e:
        print(f"Error in hierarchical method: {e}")
    
    return results

def plot_comparison(yolo_results, hierarchical_results):
    """
    Create comprehensive comparison plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Collect data
    yolo_fps = [r['fps'] for r in yolo_results]
    yolo_iou = [r['mean_iou'] for r in yolo_results]
    yolo_masks = [r['num_masks'] for r in yolo_results]
    yolo_time = [r['total_time'] * 1000 for r in yolo_results]  # ms
    
    hier_fps = [r['fps'] for r in hierarchical_results if r['fps'] > 0]
    hier_iou = [r['mean_iou'] for r in hierarchical_results if r['mean_iou'] > 0]
    hier_masks = [r['num_masks'] for r in hierarchical_results]
    hier_time = [r['total_time'] * 1000 for r in hierarchical_results if r['total_time'] > 0]
    
    # Plot 1: FPS Comparison
    ax = axes[0, 0]
    methods = ['YOLO+SAM\n(Ours)', 'Hierarchical\nEverything']
    fps_means = [np.mean(yolo_fps), np.mean(hier_fps) if hier_fps else 0]
    fps_stds = [np.std(yolo_fps), np.std(hier_fps) if hier_fps else 0]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(methods, fps_means, yerr=fps_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('FPS', fontsize=12, weight='bold')
    ax.set_title('Speed Comparison (FPS)', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, fps_means)):
        ax.text(bar.get_x() + bar.get_width()/2, val + fps_stds[i], 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Processing Time
    ax = axes[0, 1]
    time_means = [np.mean(yolo_time), np.mean(hier_time) if hier_time else 0]
    time_stds = [np.std(yolo_time), np.std(hier_time) if hier_time else 0]
    bars = ax.bar(methods, time_means, yerr=time_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Time (ms)', fontsize=12, weight='bold')
    ax.set_title('Processing Time', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, time_means)):
        ax.text(bar.get_x() + bar.get_width()/2, val + time_stds[i], 
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Mean IoU
    ax = axes[0, 2]
    iou_means = [np.mean(yolo_iou), np.mean(hier_iou) if hier_iou else 0]
    iou_stds = [np.std(yolo_iou), np.std(hier_iou) if hier_iou else 0]
    bars = ax.bar(methods, iou_means, yerr=iou_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Mean IoU', fontsize=12, weight='bold')
    ax.set_title('Segmentation Quality', fontsize=14, weight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, iou_means)):
        ax.text(bar.get_x() + bar.get_width()/2, val + iou_stds[i], 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: FPS-mAP Curve (Speed-Accuracy Trade-off)
    ax = axes[1, 0]
    if len(yolo_fps) > 0 and len(yolo_iou) > 0:
        ax.scatter(yolo_fps, yolo_iou, c='#2ecc71', s=100, alpha=0.7, 
                   label='YOLO+SAM (Ours)', marker='o', edgecolors='black', linewidth=1.5)
    if len(hier_fps) > 0 and len(hier_iou) > 0:
        ax.scatter(hier_fps, hier_iou, c='#e74c3c', s=100, alpha=0.7, 
                   label='Hierarchical Everything', marker='s', edgecolors='black', linewidth=1.5)
    ax.set_xlabel('FPS (Speed)', fontsize=12, weight='bold')
    ax.set_ylabel('Mean IoU (Quality)', fontsize=12, weight='bold')
    ax.set_title('Speed-Quality Trade-off', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Number of Masks
    ax = axes[1, 1]
    mask_means = [np.mean(yolo_masks), np.mean(hier_masks)]
    mask_stds = [np.std(yolo_masks), np.std(hier_masks)]
    bars = ax.bar(methods, mask_means, yerr=mask_stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Number of Masks', fontsize=12, weight='bold')
    ax.set_title('Mask Count Comparison', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, mask_means)):
        ax.text(bar.get_x() + bar.get_width()/2, val + mask_stds[i], 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Speedup Factor
    ax = axes[1, 2]
    if np.mean(hier_time) > 0:
        speedup = np.mean(hier_time) / np.mean(yolo_time)
        colors_speedup = ['#2ecc71' if speedup > 1 else '#e74c3c']
        bars = ax.bar(['Speedup Factor'], [speedup], color=colors_speedup, alpha=0.7)
        ax.set_ylabel('Speedup Factor', fontsize=12, weight='bold')
        ax.set_title('Speed Improvement', fontsize=14, weight='bold')
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Baseline')
        ax.grid(axis='y', alpha=0.3)
        ax.text(bars[0].get_x() + bars[0].get_width()/2, speedup, 
                f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=14)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
        ax.set_title('Speed Improvement', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('comparison_yolo_vs_hierarchical.png', dpi=150, bbox_inches='tight')
    print("Comparison plots saved to: comparison_yolo_vs_hierarchical.png")
    plt.close()

def main():
    """
    Main evaluation function
    """
    print("="*80)
    print("EVALUATION: YOLO+TinySAM vs Hierarchical Everything")
    print("="*80)
    
    # Initialize models
    print("\nLoading models...")
    yolo_model = YOLO('yolov8n.pt')
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint="./weights/tinysam_42.3.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Initialize hierarchical mask generator
    mask_generator = SamHierarchicalMaskGenerator(sam)
    print(f"Models loaded on {device}")
    
    # Test images - use COCO val2017 dataset
    # Get images from val2017 directory
    val2017_images = sorted(glob.glob('data/val2017/*.jpg'))
    
    # Configuration: adjust number of images to evaluate
    NUM_TEST_IMAGES = 50  # Change this to test more/fewer images
    
    if len(val2017_images) == 0:
        # Fallback to fig images if val2017 not available
        test_images = ['fig/picture2.jpg', 'fig/picture3.jpg']
        print(f"Warning: val2017 directory empty, using fallback images")
    else:
        # Randomly sample images for diverse evaluation
        np.random.seed(42)  # For reproducibility
        test_images = np.random.choice(val2017_images, 
                                       min(NUM_TEST_IMAGES, len(val2017_images)), 
                                       replace=False).tolist()
        print(f"Selected {len(test_images)} images from val2017 (total available: {len(val2017_images)})")
    
    yolo_results = []
    hierarchical_results = []
    
    print(f"\nEvaluating {len(test_images)} images...")
    print("Progress: ", end='', flush=True)
    
    for idx, img_path in enumerate(test_images):
        # Show progress
        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(test_images) - 1:
            print(f"{idx+1}/{len(test_images)}", end='...', flush=True)
        
        try:
            # YOLO method
            yolo_result = evaluate_yolo_method(img_path, yolo_model, predictor)
            yolo_results.append(yolo_result)
            
            # Hierarchical method (only evaluate every 5th image to save time)
            if idx % 5 == 0 or idx < 2:  # Evaluate first 2 and then every 5th
                hier_result = evaluate_hierarchical_method(img_path, mask_generator)
                hierarchical_results.append(hier_result)
                
                # Show detailed info for first few images
                if idx < 3:
                    print(f"\n  Image {idx+1}: {os.path.basename(img_path)}")
                    print(f"    YOLO+SAM: {yolo_result['total_time']*1000:.1f}ms, {yolo_result['num_masks']} masks, IoU={yolo_result['mean_iou']:.3f}")
                    print(f"    Hierarchical: {hier_result['total_time']*1000:.1f}ms, {hier_result['num_masks']} masks, IoU={hier_result['mean_iou']:.3f}")
                    if hier_result['total_time'] > 0:
                        speedup = hier_result['total_time'] / yolo_result['total_time']
                        print(f"    Speedup: {speedup:.1f}x")
            
        except Exception as e:
            if idx < 3:  # Only show errors for first few images
                print(f"\n  Error processing {os.path.basename(img_path)}: {e}")
    
    print(f"\n\nCompleted evaluation of {len(test_images)} images!")
    
    # Aggregate results
    print("\n" + "="*80)
    print("OVERALL COMPARISON")
    print("="*80)
    
    # Calculate statistics
    yolo_fps = [r['fps'] for r in yolo_results]
    yolo_iou = [r['mean_iou'] for r in yolo_results if r['mean_iou'] > 0]
    yolo_time = [r['total_time'] for r in yolo_results]
    yolo_masks = [r['num_masks'] for r in yolo_results]
    
    hier_fps = [r['fps'] for r in hierarchical_results if r['fps'] > 0]
    hier_iou = [r['mean_iou'] for r in hierarchical_results if r['mean_iou'] > 0]
    hier_time = [r['total_time'] for r in hierarchical_results if r['total_time'] > 0]
    hier_masks = [r['num_masks'] for r in hierarchical_results]
    
    print(f"\nDataset Statistics:")
    print(f"  - YOLO+SAM evaluated: {len(yolo_results)} images")
    print(f"  - Hierarchical evaluated: {len(hierarchical_results)} images (sampled)")
    print(f"  - Total objects detected (YOLO): {sum(yolo_masks)}")
    print(f"  - Total masks (Hierarchical): {sum(hier_masks)}")
    
    yolo_fps_mean = np.mean(yolo_fps)
    yolo_iou_mean = np.mean(yolo_iou)
    yolo_time_mean = np.mean(yolo_time) * 1000
    
    hier_fps_mean = np.mean(hier_fps) if hier_fps else 0
    hier_iou_mean = np.mean(hier_iou) if hier_iou else 0
    hier_time_mean = np.mean(hier_time) * 1000 if hier_time else 0
    
    print(f"\n{'Method':<40} {'FPS':<10} {'Time(ms)':<12} {'Mean IoU':<10} {'#Masks'}")
    print("-"*80)
    print(f"{'YOLO + TinySAM (Ours)':<40} {yolo_fps_mean:<10.2f} {yolo_time_mean:<12.1f} {yolo_iou_mean:<10.4f} {np.mean(yolo_masks):.1f}")
    print(f"{'Hierarchical Everything':<40} {hier_fps_mean:<10.2f} {hier_time_mean:<12.1f} {hier_iou_mean:<10.4f} {np.mean(hier_masks):.1f}")
    print("-"*80)
    
    if hier_time_mean > 0:
        speedup = hier_time_mean / yolo_time_mean
        iou_change = ((yolo_iou_mean - hier_iou_mean) / hier_iou_mean * 100) if hier_iou_mean > 0 else 0
        efficiency_gain = speedup * yolo_iou_mean / hier_iou_mean if hier_iou_mean > 0 else 0
        
        print(f"\n✅ IMPROVEMENTS:")
        print(f"  - Speed: {speedup:.2f}x faster ({yolo_time_mean:.1f}ms vs {hier_time_mean:.1f}ms)")
        print(f"  - Quality: {iou_change:+.1f}% IoU change ({yolo_iou_mean:.4f} vs {hier_iou_mean:.4f})")
        print(f"  - Efficiency (speed × quality): {efficiency_gain:.2f}x better")
        print(f"  - Masks: {np.mean(yolo_masks):.1f} vs {np.mean(hier_masks):.1f} (more focused on main objects)")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(yolo_results, hierarchical_results)
    
    # Save detailed results
    all_results = {
        'yolo_method': yolo_results,
        'hierarchical_method': hierarchical_results,
        'summary': {
            'yolo': {
                'mean_fps': float(yolo_fps_mean),
                'mean_time_ms': float(yolo_time_mean),
                'mean_iou': float(yolo_iou_mean),
            },
            'hierarchical': {
                'mean_fps': float(hier_fps_mean),
                'mean_time_ms': float(hier_time_mean),
                'mean_iou': float(hier_iou_mean),
            },
            'speedup': float(hier_time_mean / yolo_time_mean) if yolo_time_mean > 0 else 0,
        }
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Detailed results saved to: comparison_results.json")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()


