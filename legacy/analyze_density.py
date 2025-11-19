"""
简单分析检测密度（无需额外依赖）
"""
import json
from collections import defaultdict

# 加载数据
with open('eval/json_files/instances_val2017.json') as f:
    gt_data = json.load(f)

with open('eval/json_files/coco_instances_results_vitdet.json') as f:
    vitdet_preds = json.load(f)

with open('eval/yolo_tinysam_coco_results.json') as f:
    yolo_preds = json.load(f)

print("="*70)
print("📊 检测密度分析：YOLO vs ViTDet")
print("="*70)

# 统计每张图的检测数
img_gt = defaultdict(int)
img_vitdet = defaultdict(int)
img_yolo = defaultdict(int)

for ann in gt_data['annotations']:
    img_gt[ann['image_id']] += 1

for p in vitdet_preds:
    img_vitdet[p['image_id']] += 1

for p in yolo_preds:
    img_yolo[p['image_id']] += 1

# 计算密度
densities_vitdet = []
densities_yolo = []
recall_vitdet = []
recall_yolo = []

for img_id in img_gt.keys():
    gt_count = img_gt[img_id]
    vitdet_count = img_vitdet.get(img_id, 0)
    yolo_count = img_yolo.get(img_id, 0)
    
    if gt_count > 0:
        # 密度 = 预测数 / GT数（每个检测框3个候选）
        densities_vitdet.append(vitdet_count / gt_count)
        densities_yolo.append(yolo_count / gt_count)
        
        # 召回 = 检测框数 / GT数
        recall_vitdet.append(min(1.0, (vitdet_count/3) / gt_count))
        recall_yolo.append(min(1.0, (yolo_count/3) / gt_count))

# 计算平均值
def calc_stats(data):
    if not data:
        return 0, 0, 0, 0, 0
    data_sorted = sorted(data)
    n = len(data)
    return {
        'mean': sum(data) / n,
        'median': data_sorted[n//2],
        'min': min(data),
        'max': max(data),
        'p25': data_sorted[n//4],
        'p75': data_sorted[3*n//4]
    }

vitdet_stats = calc_stats(densities_vitdet)
yolo_stats = calc_stats(densities_yolo)
recall_vitdet_stats = calc_stats(recall_vitdet)
recall_yolo_stats = calc_stats(recall_yolo)

print("\n1️⃣  检测密度统计 (预测数/GT数)")
print("-"*70)
print(f"{'指标':<15} {'ViTDet+TinySAM':<20} {'YOLO+TinySAM':<20} {'差异'}")
print("-"*70)
print(f"{'平均值':<15} {vitdet_stats['mean']:<20.2f} {yolo_stats['mean']:<20.2f} {vitdet_stats['mean']-yolo_stats['mean']:+.2f}")
print(f"{'中位数':<15} {vitdet_stats['median']:<20.2f} {yolo_stats['median']:<20.2f} {vitdet_stats['median']-yolo_stats['median']:+.2f}")
print(f"{'最小值':<15} {vitdet_stats['min']:<20.2f} {yolo_stats['min']:<20.2f} {vitdet_stats['min']-yolo_stats['min']:+.2f}")
print(f"{'最大值':<15} {vitdet_stats['max']:<20.2f} {yolo_stats['max']:<20.2f} {vitdet_stats['max']-yolo_stats['max']:+.2f}")
print(f"{'25%分位':<15} {vitdet_stats['p25']:<20.2f} {yolo_stats['p25']:<20.2f} {vitdet_stats['p25']-yolo_stats['p25']:+.2f}")
print(f"{'75%分位':<15} {vitdet_stats['p75']:<20.2f} {yolo_stats['p75']:<20.2f} {vitdet_stats['p75']-yolo_stats['p75']:+.2f}")

print("\n2️⃣  召回率统计 (检测框数/GT数)")
print("-"*70)
print(f"{'指标':<15} {'ViTDet':<20} {'YOLO':<20} {'差异'}")
print("-"*70)
print(f"{'平均召回率':<15} {recall_vitdet_stats['mean']*100:<19.1f}% {recall_yolo_stats['mean']*100:<19.1f}% {(recall_vitdet_stats['mean']-recall_yolo_stats['mean'])*100:+.1f}%")
print(f"{'中位召回率':<15} {recall_vitdet_stats['median']*100:<19.1f}% {recall_yolo_stats['median']*100:<19.1f}% {(recall_vitdet_stats['median']-recall_yolo_stats['median'])*100:+.1f}%")

# 分析密集程度
print("\n3️⃣  密度特征分析")
print("-"*70)

# 过度检测（密度>3，即检测框数>GT）
over_detect_vitdet = sum(1 for d in densities_vitdet if d > 3)
over_detect_yolo = sum(1 for d in densities_yolo if d > 3)
print(f"过度检测图片数 (密度>3):")
print(f"  ViTDet: {over_detect_vitdet} 张 ({over_detect_vitdet/len(densities_vitdet)*100:.1f}%)")
print(f"  YOLO:   {over_detect_yolo} 张 ({over_detect_yolo/len(densities_yolo)*100:.1f}%)")

# 欠检测（密度<2，即检测框数<2/3 GT）
under_detect_vitdet = sum(1 for d in densities_vitdet if d < 2)
under_detect_yolo = sum(1 for d in densities_yolo if d < 2)
print(f"\n欠检测图片数 (密度<2):")
print(f"  ViTDet: {under_detect_vitdet} 张 ({under_detect_vitdet/len(densities_vitdet)*100:.1f}%)")
print(f"  YOLO:   {under_detect_yolo} 张 ({under_detect_yolo/len(densities_yolo)*100:.1f}%)")

print("\n4️⃣  结论")
print("="*70)

if yolo_stats['mean'] < vitdet_stats['mean'] * 0.7:
    print("❌ YOLO检测太稀疏（相比ViTDet少30%以上）")
    print("   → 主要问题：召回率低，漏检太多")
    print("   → 建议：降低置信度阈值或使用更大模型")
elif yolo_stats['mean'] > vitdet_stats['mean'] * 1.3:
    print("⚠️  YOLO检测太密集（相比ViTDet多30%以上）")
    print("   → 主要问题：可能有过多背景噪声")
    print("   → 建议：提高置信度阈值以减少误检")
else:
    print("✓ YOLO和ViTDet的检测密度相当")
    print("  → 密度不是主要问题")
    
print(f"\n💡 当前情况：")
print(f"   YOLO密度 = {yolo_stats['mean']:.2f}x GT")
print(f"   ViTDet密度 = {vitdet_stats['mean']:.2f}x GT")
print(f"   相对差异 = {(yolo_stats['mean']/vitdet_stats['mean']-1)*100:+.1f}%")
print(f"   召回率差异 = {(recall_yolo_stats['mean']/recall_vitdet_stats['mean']-1)*100:+.1f}%")

print("\n" + "="*70)

