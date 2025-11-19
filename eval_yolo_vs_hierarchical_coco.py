"""
COCO 评估脚本：YOLO v12-turbo + Hierarchical TinySAM
- 高置信度：YOLO检测 + BOX prompts
- 低置信度：16×16 密集点采样（YOLO框外） + Point prompts
"""

import os
import sys
import json
import time
import numpy as np
import torch
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util

sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor
from ultralytics import YOLO

# ============== 配置 ==============
VAL_IMG_PATH = "data/val2017"
VAL_JSON_PATH = "eval/json_files/instances_val2017.json"
SAM_CHECKPOINT = "weights/tinysam_42.3.pth"
OUTPUT_JSON = "eval/yolo_hierarchical_coco_results.json"

# 模型配置
YOLO_MODEL = "yolo12-turbo.pt"  # YOLO v12-turbo
YOLO_CONF_HIGH = 0.25           # 高置信度阈值
POINTS_PER_SIDE = 16            # 16×16 密集点采样
OVERLAP_THRESHOLD = 0.5         # 与高置信度mask的重叠阈值

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ============== 工具函数 ==============
def generate_dense_points_outside_boxes(image_shape, boxes, points_per_side=16):
    """在YOLO检测框外生成密集采样点"""
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

# ============== 加载模型 ==============
print("\n📥 加载模型...")
print(f"  - YOLO: {YOLO_MODEL}")
print(f"  - TinySAM: vit_t")

yolo_model = YOLO(YOLO_MODEL)
model_type = "vit_t"
sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)
print("✅ 模型加载完成")

# ============== 加载 COCO Ground Truth ==============
print("\n📥 加载 COCO Ground Truth...")
coco_gt = COCO(VAL_JSON_PATH)
print(f"✅ 加载完成: {len(coco_gt.imgs)} 张图片, {len(coco_gt.anns)} 个标注")

# 获取所有图片 ID
img_ids = sorted(coco_gt.imgs.keys())
print(f"📊 将处理 {len(img_ids)} 张图片")

# ============== 运行评估 ==============
print("\n🚀 开始 YOLO v12-turbo + Hierarchical TinySAM 评估...")
print(f"配置: YOLO conf={YOLO_CONF_HIGH}, 密集点={POINTS_PER_SIDE}×{POINTS_PER_SIDE}")

results = []
total_time = 0
processed = 0
total_high_conf_masks = 0
total_low_conf_masks = 0

# YOLO 类别ID到COCO类别ID的映射
yolo_to_coco_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
                    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
                    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 
                    85, 86, 87, 88, 89, 90]

for idx, img_id in enumerate(img_ids):
    # 加载图片
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = os.path.join(VAL_IMG_PATH, img_info['file_name'])
    
    if not os.path.exists(img_path):
        continue
    
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    
    # ============== Step 1: YOLO 检测（高置信度区域）==============
    yolo_results = yolo_model(image_rgb, conf=YOLO_CONF_HIGH, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    confs = yolo_results[0].boxes.conf.cpu().numpy()
    clses = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
    
    # 设置图片
    predictor.set_image(image_rgb)
    
    # ============== Step 2a: 高置信度分割（BOX prompts）==============
    high_conf_masks_list = []
    for box, conf, cls_id in zip(boxes, confs, clses):
        # 使用 BOX prompt
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :]
        )
        
        # 取最佳 mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        high_conf_masks_list.append(best_mask)
        
        # 转换为 COCO RLE 格式
        mask_binary = best_mask.astype(np.uint8)
        rle = mask_util.encode(np.asfortranarray(mask_binary))
        rle['counts'] = rle['counts'].decode('utf-8')
        
        # 映射类别ID
        coco_cat_id = yolo_to_coco_map[int(cls_id)]
        
        # 计算综合分数
        combined_score = float(conf) * float(best_score)
        combined_score = max(0.0, min(1.0, combined_score))
        
        # 添加到结果
        results.append({
            'image_id': img_id,
            'category_id': coco_cat_id,
            'segmentation': rle,
            'score': combined_score,
            'source': 'high_conf'  # 标记来源
        })
    
    total_high_conf_masks += len(high_conf_masks_list)
    
    # 创建高置信度区域的联合mask
    combined_high_conf_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    for mask in high_conf_masks_list:
        combined_high_conf_mask = combined_high_conf_mask | mask
    
    # ============== Step 2b: 低置信度分割（密集点采样）==============
    # 在YOLO框外生成16×16密集点
    low_conf_points = generate_dense_points_outside_boxes(
        image_rgb.shape, boxes, points_per_side=POINTS_PER_SIDE
    )
    
    # 对每个点进行分割
    low_conf_count = 0
    for point in low_conf_points:
        try:
            masks, scores, _ = predictor.predict(
                point_coords=point.reshape(1, 2),
                point_labels=np.ones(1)
            )
            
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            
            # 检查与高置信度区域的重叠
            overlap = np.sum(best_mask & combined_high_conf_mask) / (np.sum(best_mask) + 1e-6)
            
            # 只保留重叠小于阈值且面积足够的mask
            if overlap < OVERLAP_THRESHOLD and np.sum(best_mask) > 100:
                # 转换为 COCO RLE 格式
                mask_binary = best_mask.astype(np.uint8)
                rle = mask_util.encode(np.asfortranarray(mask_binary))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # 低置信度区域：使用默认类别或基于位置估计
                # 简化处理：使用类别1 (person) 作为默认
                default_category = 1
                
                # 添加到结果（分数较低）
                results.append({
                    'image_id': img_id,
                    'category_id': default_category,
                    'segmentation': rle,
                    'score': float(best_score) * 0.5,  # 降低低置信度mask的分数
                    'source': 'low_conf'  # 标记来源
                })
                low_conf_count += 1
        except:
            continue
    
    total_low_conf_masks += low_conf_count
    
    elapsed = time.time() - start_time
    total_time += elapsed
    processed += 1
    
    # 打印进度
    if (idx + 1) % 100 == 0:
        avg_time = total_time / processed
        avg_high = total_high_conf_masks / processed
        avg_low = total_low_conf_masks / processed
        eta = avg_time * (len(img_ids) - processed)
        print(f"进度: {idx+1}/{len(img_ids)} | "
              f"平均时间: {avg_time:.2f}s/图 | "
              f"高/低conf: {avg_high:.1f}/{avg_low:.1f} | "
              f"预计剩余: {eta/60:.1f}分钟")

print(f"\n✅ 处理完成！")
print(f"   共处理: {processed} 张图片")
print(f"   高置信度mask: {total_high_conf_masks:,} 个 ({total_high_conf_masks/processed:.1f}/图)")
print(f"   低置信度mask: {total_low_conf_masks:,} 个 ({total_low_conf_masks/processed:.1f}/图)")
print(f"   总mask数: {len(results):,} 个")
print(f"⏱️  总时间: {total_time:.1f}秒")
print(f"📊 平均速度: {total_time/processed:.2f}秒/图")

# ============== 保存结果 ==============
print(f"\n💾 保存结果到: {OUTPUT_JSON}")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f)

# ============== COCO 官方评估 ==============
print("\n" + "="*70)
print("📊 COCO 官方评估指标 - YOLO v12-turbo + Hierarchical TinySAM")
print("="*70)

# 加载预测结果
coco_dt = coco_gt.loadRes(OUTPUT_JSON)

# 创建评估器
coco_eval = COCOeval(coco_gt, coco_dt, 'segm')

# 运行评估
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("\n" + "="*70)
print("📊 关键指标总结")
print("="*70)
print(f"AP @IoU=0.50:0.95 (主要指标): {coco_eval.stats[0]:.3f} ({coco_eval.stats[0]*100:.1f}%)")
print(f"AP @IoU=0.50:         {coco_eval.stats[1]:.3f} ({coco_eval.stats[1]*100:.1f}%)")
print(f"AP @IoU=0.75:         {coco_eval.stats[2]:.3f} ({coco_eval.stats[2]*100:.1f}%)")
print(f"AP (small):           {coco_eval.stats[3]:.3f} ({coco_eval.stats[3]*100:.1f}%)")
print(f"AP (medium):          {coco_eval.stats[4]:.3f} ({coco_eval.stats[4]*100:.1f}%)")
print(f"AP (large):           {coco_eval.stats[5]:.3f} ({coco_eval.stats[5]*100:.1f}%)")
print(f"AR @maxDets=100:      {coco_eval.stats[8]:.3f} ({coco_eval.stats[8]*100:.1f}%)")

print("\n" + "="*70)
print("📈 与论文中的 TinySAM 对比")
print("="*70)
print("| 方法                           | COCO AP (%) |")
print("|--------------------------------|-------------|")
print(f"| TinySAM (ViTDet + SAM)         | 42.3        |")
print(f"| YOLO v12 + Hierarchical SAM    | {coco_eval.stats[0]*100:.1f}       |")

ap_diff = (coco_eval.stats[0] - 0.423) * 100
if ap_diff > 0:
    print(f"\n✅ 新方法比原始方法高 {ap_diff:.1f}%")
else:
    print(f"\n⚠️  新方法比原始方法低 {abs(ap_diff):.1f}%")

print("\n💡 方法特点:")
print("- 高置信度: YOLO v12-turbo检测 + BOX prompts")
print("- 低置信度: 16×16密集点采样 + Point prompts")
print("- 更全面的场景覆盖（包含背景区域）")
print("- 速度: ~2-3秒/图（GPU）")

