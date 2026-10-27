# sam3first.py
"""
SAM3 图像分割与水线提取核心模块
包含掩码后处理、形态学操作、水位线计算、水体提取流水线，
以及单张图片处理（文件版和 PIL 版）和批量图片处理函数。
"""

import os
import re
import json
import time
import cv2
import numpy as np
import pandas as pd
import torch
import traceback
from PIL import Image
from pathlib import Path
from datetime import datetime
from sam3.model.sam3_image_processor import Sam3Processor

# 标定基准尺寸（虚拟水尺 CSV 的 v_star 及各处固定像素阈值均基于此标定）
BASE_W = 100
BASE_H = 240


# ==================== 辅助函数（真实水位线读取、匹配等） ====================

def load_ground_truth_waterline(gt_file_path):
    try:
        df = pd.read_excel(gt_file_path)
        print(f"读取真实水位线文件: {gt_file_path}")

        gt_dict_exact, gt_dict_stem, gt_dict_numeric, gt_dict_with_zero = {}, {}, {}, {}

        for _, row in df.iterrows():
            img_name = str(row.iloc[0])
            waterline = row.iloc[1]
            try:
                waterline = float(waterline)
            except:
                continue

            gt_dict_exact[img_name] = waterline
            stem = Path(img_name).stem
            gt_dict_stem[stem] = waterline

            numbers = re.findall(r'\d+', img_name)
            if numbers:
                gt_dict_with_zero[numbers[0]] = waterline
                num_without_zero = str(int(numbers[0]))
                gt_dict_numeric[num_without_zero] = waterline

        return {'exact': gt_dict_exact, 'stem': gt_dict_stem, 'with_zero': gt_dict_with_zero,
                'numeric': gt_dict_numeric}
    except Exception as e:
        print(f"读取真实水位线文件失败: {e}")
        return {}


def find_ground_truth(filename, gt_dicts):
    filename_str = str(filename)
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        if filename_str + ext in gt_dicts.get('exact', {}):
            return gt_dicts['exact'][filename_str + ext]
    if filename_str in gt_dicts.get('stem', {}):
        return gt_dicts['stem'][filename_str]

    numbers = re.findall(r'\d+', filename_str)
    if numbers:
        num = numbers[0]
        if num in gt_dicts.get('with_zero', {}):
            return gt_dicts['with_zero'][num]
        num_int = str(int(num))
        if num_int in gt_dicts.get('numeric', {}):
            return gt_dicts['numeric'][num_int]
    return None


def save_results_to_excel_with_comparison(success_results, failed_images, output_folder, gt_dicts):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if success_results:
        data = []
        for filename, pred_waterline, mask_top_y, confidence, used_threshold in success_results:
            gt_waterline = find_ground_truth(filename, gt_dicts)

            if gt_waterline is not None:
                diff = pred_waterline - gt_waterline
                diff_abs = abs(diff)
                data.append([filename, pred_waterline, mask_top_y, gt_waterline, diff, diff_abs,
                             f"{confidence:.4f}" if confidence is not None else "N/A", f"{used_threshold:.2f}"])
            else:
                data.append([filename, pred_waterline, mask_top_y, "N/A", "N/A", "N/A",
                             f"{confidence:.4f}" if confidence is not None else "N/A", f"{used_threshold:.2f}"])

        success_df = pd.DataFrame(data,
                                  columns=['图片序号名', '测量水位线', '掩码最高位置', '真实水位线', '差值', '绝对差值',
                                           '置信度', '使用阈值'])
        success_file = Path(output_folder) / f"水位线结果_{timestamp}.xlsx"
        success_df.to_excel(success_file, index=False)
        print(f"\n对比结果已保存到: {success_file}")

    if failed_images:
        failed_df = pd.DataFrame(failed_images, columns=['图片序号名'])
        failed_file = Path(output_folder) / f"失败图片_{timestamp}.xlsx"
        failed_df.to_excel(failed_file, index=False)


# ==================== 掩码基本操作 ====================

def get_mask_bottom_y(mask):
    if mask is None:
        return -1
    mask_np = mask.cpu().float().numpy() if isinstance(mask, torch.Tensor) else mask
    mask_np = mask_np > 0.5 if mask_np.dtype in [np.float32, np.float64] else mask_np
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return -1
    y_coords = np.where(mask_np)[0]
    return int(np.max(y_coords)) if len(y_coords) > 0 else -1


def get_all_masks_sorted_by_confidence(masks, scores):
    if masks is None or len(masks) == 0:
        return [], [], []
    if scores is not None and len(scores) > 0:
        scores_np = scores.cpu().float().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
        sorted_indices = np.argsort(scores_np)[::-1]
        sorted_scores = scores_np[sorted_indices]
        sorted_masks = [masks[i] for i in sorted_indices]
        return sorted_masks, sorted_scores, sorted_indices.tolist()
    else:
        sorted_masks = [masks[i] for i in range(len(masks))]
        return sorted_masks, None, list(range(len(masks)))


def remove_top_connected_components(mask, top_rows=50):
    if mask is None or not np.any(mask):
        return mask
    mask_2d = np.squeeze(mask).copy()
    if mask_2d.ndim != 2:
        return mask
    h, w = mask_2d.shape
    top_rows = min(top_rows, h)
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
    top_labels = set(labels[:top_rows, :].flatten()) - {0}
    if not top_labels:
        return mask
    new_mask = np.zeros_like(mask_2d)
    for label in range(1, num_labels):
        if label not in top_labels:
            new_mask[labels == label] = 1
    return new_mask.reshape(mask.shape) if isinstance(mask, np.ndarray) and mask.ndim > 2 else new_mask


def get_connected_components(mask, kernel_size=3, opening_iterations=2):
    if mask is None or not np.any(mask):
        return []
    mask_2d = np.squeeze(mask).copy()
    if mask_2d.ndim != 2:
        return []
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    num_labels, labels = cv2.connectedComponents(mask_opened, connectivity=8)
    components = []
    for label in range(1, num_labels):
        component = (labels == label).astype(mask_2d.dtype)
        y_coords = np.where(component)[0]
        if len(y_coords) > 0:
            components.append((component, int(np.max(y_coords))))
    components.sort(key=lambda x: x[1], reverse=True)
    return components


def select_masks_by_bottom_accumulation(mask, threshold_percent=60, kernel_size=3, opening_iterations=2):
    if mask is None or not np.any(mask):
        return None, [], 0
    components = get_connected_components(mask, kernel_size=kernel_size, opening_iterations=opening_iterations)
    if not components:
        return None, [], 0
    total_pixels = sum(np.sum(comp) for comp, _ in components)
    if total_pixels == 0:
        return None, [], 0
    threshold_area = int(total_pixels * threshold_percent / 100)
    accumulated_mask = np.zeros_like(components[0][0], dtype=bool)
    accumulated_pixels = 0
    selected_components = []
    for comp, bottom_y in components:
        comp_pixels = np.sum(comp)
        accumulated_mask = accumulated_mask | comp
        accumulated_pixels += comp_pixels
        selected_components.append((comp, bottom_y))
        if accumulated_pixels >= threshold_area:
            break
    return accumulated_mask, selected_components, accumulated_pixels / total_pixels


def calculate_waterline_by_percentage(masks, image_height, threshold_percent=60, min_continuous_rows=3, floating_mask=None):
    if masks is None:
        return None, None
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if masks.dtype in [np.float32, np.float64]:
        masks = masks > 0.5
    combined_mask = np.squeeze(masks)
    if combined_mask.ndim != 2:
        return None, None
    if not np.any(combined_mask):
        return None, None

    h, w = combined_mask.shape
    # 每行漂浮物占的宽度（有效宽度 = 总宽 - 漂浮物宽度）
    if floating_mask is not None:
        fm = np.squeeze(floating_mask)
        if fm.shape == (h, w):
            float_row_width = fm.sum(axis=1).astype(int)
        else:
            float_row_width = np.zeros(h, dtype=int)
    else:
        float_row_width = np.zeros(h, dtype=int)

    found_y = None
    continuous_count = 0

    for y in range(h):
        effective_width = w - int(float_row_width[y])
        threshold_count = int(effective_width * threshold_percent / 100)
        if int(np.sum(combined_mask[y, :])) >= threshold_count:
            continuous_count += 1
            if continuous_count >= min_continuous_rows:
                found_y = y - continuous_count + 1
                break
        else:
            continuous_count = 0

    if found_y is None:
        for y in range(h):
            effective_width = w - int(float_row_width[y])
            threshold_count = int(effective_width * threshold_percent / 100)
            if int(np.sum(combined_mask[y, :])) >= threshold_count:
                found_y = y
                break

    if found_y is None:
        y_coords = np.where(combined_mask)[0]
        if len(y_coords) > 0:
            found_y = np.max(y_coords)

    all_y = np.where(combined_mask)[0]
    mask_top_y = int(np.min(all_y)) if len(all_y) > 0 else 0
    return (int(found_y) if found_y is not None else None), mask_top_y


def extract_2d_mask(mask_tensor):
    mask_np = mask_tensor.cpu().float().numpy() if torch.is_tensor(mask_tensor) else np.array(mask_tensor)
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    return mask_np > 0.5


def keep_largest_connected_component(mask_2d):
    if mask_2d is None or not np.any(mask_2d):
        return mask_2d
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask_2d
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    return labels == largest_label


def calculate_overlap_ratio(mask_test, mask_ref):
    if mask_ref is None:
        return 0.0
    # 尺寸不一致时（如 SAM3 掩码比原图少 1 像素宽）统一对齐
    if mask_test.shape != mask_ref.shape:
        mask_ref = cv2.resize(mask_ref.astype(np.uint8),
                              (mask_test.shape[1], mask_test.shape[0])).astype(bool)
    intersection = np.logical_and(mask_test, mask_ref).sum()
    area_test = mask_test.sum()
    return intersection / area_test if area_test > 0 else 0.0


def apply_morphology(mask_2d):
    mask_uint8 = (mask_2d.astype(np.uint8) * 255)
    kernel_close = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    return closed > 0


# ==================== 动态框提示：MobileNet-SSD 水体框 ====================

def is_no_slope_mode(ruler_csv):
    """是否为「无边坡」模式。

    control_first.get_ruler_csv_for_site() 在用户端选择「无边坡」时返回空字符串 ""，
    该空串一路透传到这里的 ruler_csv 参数，作为「无边坡」的唯一标志。
    无边坡模式下不加载、不调用 MobileNet-SSD，直接走纯语义提示（仅文本提示，无正框/负框）。
    注意：None 表示调用方未指定 CSV（如离线批处理脚本），保持原有「默认模型编号」行为不变。
    """
    return ruler_csv is not None and not str(ruler_csv).strip()


def compute_dynamic_crop_edge(curr_img, floating_mask=None, ruler_csv=None):
    """动态框提示：仅用 MobileNet-SSD 水体框。
    检测到水体框 -> crop_edge = 水体框上沿 - 20px；同时给出水体框下沿 box_y2（作正框提示下边界，不往上移）。
    未检测到水体框 -> crop_edge = None（走纯语义提示，无框）。
    无边坡（ruler_csv 为空）-> 直接跳过 SSD，crop_edge = None（纯语义提示，无框）。
    返回 (source, crop_edge, ssd_confidence, ssd_box_y1, ssd_box_y2, ssd_bank_box)。
    source 恒为 "ssd"（提示来源只有 MobileNet-SSD 一种），无边坡时为 None。"""
    if is_no_slope_mode(ruler_csv):
        print("  [动态框] 无边坡：跳过 MobileNet-SSD，直接语义提示分割（无框）")
        return None, None, None, None, None, None
    try:
        import model as ssd_model
        water_box, bank_box, confidence = ssd_model.detect_water_box(curr_img, ruler_csv)
        if water_box is not None:
            box_y1 = int(water_box[1])
            box_y2 = int(water_box[3])
            crop_edge = max(0, box_y1 - 20)
            print(f"  [SSD框提示] SSD 水体框 ymin={box_y1}px ymax={box_y2}px，裁剪上边缘: {crop_edge}px")
            return "ssd", crop_edge, confidence, box_y1, box_y2, bank_box
        print("  [SSD框提示] SSD 未检出水体框，退回纯语义提示（无框）")
        return "ssd", None, confidence, None, None, bank_box
    except Exception as e:
        print(f"  [SSD框提示] SSD 模型检测异常: {e}")
        return "ssd", None, None, None, None, None


# ==================== 核心水位掩码提取 ====================

def mask_satisfies_constraints(mask, img_width, img_height, crop_edge=None):
    """
    判断掩码是否满足选取约束：
      1) 掩码体积 > 图片体积 10%；
      （已去除"掩码顶部高于提示框半高线"的限制）
    """
    if mask is None or not np.any(mask):
        return False
    if np.sum(mask) <= img_width * img_height * 0.10:
        return False
    return True


def build_mask_b(mask_after_top_removal, img_width, img_height, threshold_percent=90):
    """
    掩码B（较宽松）：原始掩码 -> 顶部连通域删除 -> 10%面积门槛（剔除小目标误检）
    -> 剩余掩码自下而上累计超过 threshold_percent%（默认90%）的面积。
    返回 2D bool 掩码；不满足条件返回 None。
    """
    if mask_after_top_removal is None or not np.any(mask_after_top_removal):
        return None
    mask_2d = np.squeeze(mask_after_top_removal).astype(bool)
    if mask_2d.ndim != 2:
        return None
    if np.sum(mask_2d) <= img_width * img_height * 0.10:
        return None
    selected, _, _ = select_masks_by_bottom_accumulation(
        mask_2d, threshold_percent=threshold_percent, kernel_size=3, opening_iterations=2
    )
    if selected is None or not np.any(selected):
        return None
    return selected.astype(bool)


def fill_floating_with_water_below(image_np, floating_mask):
    """
    把漂浮物区域用其下方紧邻的水体颜色填充（逐列向下取最近的非漂浮物像素）。
    漂浮物贴底时，改用其上方最近的非漂浮物像素。
    """
    img = image_np.copy()
    h, w = img.shape[:2]
    fm = np.squeeze(floating_mask)
    if fm.ndim != 2:
        return img
    fm = fm.astype(bool)
    if fm.shape[:2] != (h, w):
        fm = cv2.resize(fm.astype(np.uint8), (w, h)).astype(bool)

    for x in range(w):
        col = np.where(fm[:, x])[0]
        if len(col) == 0:
            continue
        y_top, y_bot = int(col.min()), int(col.max())
        y_src = y_bot + 1
        while y_src < h and fm[y_src, x]:
            y_src += 1
        if y_src >= h:
            y_src = y_top - 1
            if y_src < 0:
                continue
        img[y_top:y_bot + 1, x] = img[y_src, x]
    return img


def _box_to_prompt(box, img_w, img_h):
    """把 [xmin, ymin, xmax, ymax] 像素框转成 [center_x, center_y, width, height] 归一化框。"""
    xmin, ymin, xmax, ymax = box
    cx = (xmin + xmax) / 2.0 / float(img_w)
    cy = (ymin + ymax) / 2.0 / float(img_h)
    w = max(0.0, (xmax - xmin) / float(img_w))
    h = max(0.0, (ymax - ymin) / float(img_h))
    return [cx, cy, w, h]


def mask_top_y(mask):
    """掩码最高值 = 掩码最上方像素的 y（最小 y），无掩码返回 None。"""
    if mask is None or not np.any(mask):
        return None
    ys = np.where(mask)[0]
    return int(ys.min()) if len(ys) > 0 else None


def waterline_by_width(mask, threshold_percent=30, min_continuous_rows=3):
    """从顶部向下，找第一处连续 min_continuous_rows 行、每行掩码宽度 > 阈值比例(默认30%图宽)的位置作为水位线。
    未找到/无掩码返回 None。"""
    if mask is None or not np.any(mask):
        return None
    m = np.squeeze(mask)
    if m.ndim != 2:
        return None
    h, w = m.shape
    threshold = max(1, int(w * threshold_percent / 100.0))
    continuous = 0
    for y in range(h):
        if int(np.sum(m[y, :])) > threshold:
            continuous += 1
            if continuous >= min_continuous_rows:
                return y - continuous + 1
        else:
            continuous = 0
    return None


def get_water_mask_pipeline(pil_image, model, device, prompt="black water", is_cropped=False, crop_edge=None, floating_mask=None, return_mask_b=False, bank_box=None, box_y2=None):
    # 用下方水体代替漂浮物区域，使分割时把漂浮物处当作水面处理
    if floating_mask is not None:
        image_np = np.array(pil_image)
        image_np = fill_floating_with_water_below(image_np, floating_mask)
        pil_image = Image.fromarray(image_np)

    img_width, img_height = pil_image.size
    thresholds_to_try = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    # 框提示：仅在 SSD 给出水体框上沿（crop_edge 不为 None）时使用动态框；无边坡/未检出则不加框提示
    bottom_box = None
    if crop_edge is not None:
        y0 = max(0, min(int(crop_edge), img_height - 1)) / float(img_height)
        if box_y2 is not None:
            y1 = max(y0, min(int(box_y2), img_height - 1)) / float(img_height)
        else:
            y1 = 1.0
        bottom_box = [0.5, (y0 + y1) / 2.0, 1.0, y1 - y0]
    bank_prompt = None
    if bank_box is not None:
        bank_prompt = _box_to_prompt(bank_box, img_width, img_height)
        print(f"  [负框提示] 岸框 [xmin,ymin,xmax,ymax]={[round(v,1) for v in bank_box]}，下边缘 ymax={bank_box[3]:.0f}")

    for threshold in thresholds_to_try:
        try:
            processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
        except:
            processor = Sam3Processor(model, confidence_threshold=threshold)

        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        # 结合框提示（仅当动态裁剪上边缘存在时）
        if bottom_box is not None:
            try:
                state = processor.add_geometric_prompt(bottom_box, True, state)
            except Exception:
                pass
        if bank_prompt is not None:
            try:
                state = processor.add_geometric_prompt(bank_prompt, False, state)
            except Exception:
                pass
        if 'masks' not in state or state['masks'] is None:
            continue
        masks = state['masks']
        scores = state.get('scores', [])
        if len(masks) == 0:
            continue

        sorted_masks, sorted_scores, _ = get_all_masks_sorted_by_confidence(masks, scores)

        for i, mask in enumerate(sorted_masks):
            score = sorted_scores[i] if sorted_scores is not None else None
            mask_np = mask.cpu().float().numpy() > 0.5 if isinstance(mask, torch.Tensor) else mask > 0.5
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim != 2:
                continue
            # SAM3 对奇数尺寸输出掩码宽度可能差 1，统一 resize 到输入图片尺寸
            if mask_np.shape[:2] != (img_height, img_width):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (img_width, img_height)).astype(bool)

            mask_after_top_removal = mask_np.copy()

            if not np.any(mask_after_top_removal):
                continue

            # 不再做自下而上累计筛选：直接保留初始水体提示框(crop_edge)以下的全部掩码
            selected_mask = mask_after_top_removal.copy()
            if crop_edge is not None:
                cut = max(0, min(int(crop_edge), img_height))
                selected_mask[:cut, :] = False

            if not np.any(selected_mask):
                continue
            # 约束：体积>图片10%
            if not mask_satisfies_constraints(selected_mask, img_width, img_height, crop_edge):
                continue
            mask_b = None
            if return_mask_b:
                # 可视化用掩码B：直接输出 SAM3 原始二值掩码
                mask_b = mask_np.copy()
                return selected_mask, score, threshold, mask_b
            return selected_mask, score, threshold

    if return_mask_b:
        return None, None, None, None
    return None, None, None


# ==================== 单张图片处理（文件版本，保留原始逻辑） ====================

def process_single_image(img_path, model, prompt, gt_dicts, output_folder, mask_output_folder, save_masks, ruler_csv=None):
    """
    处理单张图片文件，提取水位线，保存结果图片和掩码，并返回结果。
    此函数保留原始实现，包含文件读取、结果保存和与真实值比较等功能。
    """
    fname = img_path.name
    print(f"\n[高阶处理] >>> {fname}")

    device = next(model.parameters()).device if hasattr(model, 'parameters') else "cuda"

    try:
        pil_image = Image.open(str(img_path)).convert("RGB")
        image_np = np.array(pil_image)
        h_orig, w_orig = image_np.shape[:2]
    except Exception as e:
        print(f"  ❌ 无法加载图片: {e}")
        return False, fname

    # SSD 框提示：用 MobileNet-SSD 水体框定位水位线裁剪上边缘
    _src, crop_edge, _ssd_conf, _ssd_box_y1, _ssd_box_y2, _ssd_bank_box = compute_dynamic_crop_edge(image_np, ruler_csv=ruler_csv)
    if crop_edge is not None:
        print(f"  [SSD框提示] 裁剪上边缘: {crop_edge}px")

    ref_water_mask, score1, thresh1 = get_water_mask_pipeline(pil_image, model, device, prompt=prompt, is_cropped=False, crop_edge=crop_edge)
    ref_waterline_y, ref_mask_top_y = -1, -1

    if ref_water_mask is not None:
        cleaned_ref_mask = apply_morphology(ref_water_mask)
        calc_y, top_y = calculate_waterline_by_percentage(cleaned_ref_mask, h_orig)
        if calc_y is not None:
            ref_waterline_y, ref_mask_top_y = calc_y, top_y
            print(f"  [原水体] 置信度:{score1:.3f}, Y={ref_waterline_y}")

    bank_masks, bank_scores = [], []
    for bank_thresh in [0.4, 0.2, 0.0]:
        try:
            processor_bank = Sam3Processor(model, device=device, confidence_threshold=bank_thresh)
        except:
            processor_bank = Sam3Processor(model, confidence_threshold=bank_thresh)

        state_bank = processor_bank.set_image(pil_image)
        processor_bank.set_text_prompt("bank", state_bank)
        if state_bank.get("masks") is not None and len(state_bank.get("masks")) > 0:
            bank_masks, bank_scores = state_bank.get("masks"), state_bank.get("scores")
            break

    best_bank_mask, best_bank_score = None, -1.0
    if len(bank_masks) > 0:
        sorted_bank_masks, sorted_bank_scores, _ = get_all_masks_sorted_by_confidence(bank_masks, bank_scores)
        top_thresh = max(int(h_orig * 50 / BASE_H), int(h_orig * 0.1))

        for i, mask in enumerate(sorted_bank_masks):
            m_2d = extract_2d_mask(mask)
            m_2d = keep_largest_connected_component(m_2d)
            s_val = float(sorted_bank_scores[i]) if sorted_bank_scores is not None else 0.0

            y_coords, _ = np.where(m_2d)
            if len(y_coords) == 0:
                continue
            min_y = np.min(y_coords)
            overlap = calculate_overlap_ratio(m_2d, ref_water_mask)

            if min_y <= top_thresh and overlap <= 0:
                if s_val > best_bank_score:
                    best_bank_score, best_bank_mask = s_val, m_2d

    crop_y = 0
    if best_bank_mask is not None:
        lowest_y = np.max(np.where(best_bank_mask)[0])
        crop_y = max(0, lowest_y - max(3, int(round(h_orig * 10 / BASE_H))))

    cropped_img_np = image_np[crop_y:, :, :]
    cropped_pil = Image.fromarray(cropped_img_np)
    # 第二轮裁剪图也加框提示（crop_edge 换算到裁剪图坐标）
    crop_edge_cropped = None
    if crop_edge is not None:
        crop_edge_cropped = max(0, crop_edge - crop_y)
    best_crop_water_mask, score2, thresh2 = get_water_mask_pipeline(cropped_pil, model, device, prompt=prompt,
                                                                    is_cropped=True, crop_edge=crop_edge_cropped)

    final_full_mask = np.zeros((h_orig, w_orig), dtype=bool)
    new_waterline_y, new_mask_top_y = -1, -1

    if best_crop_water_mask is not None:
        cleaned_crop_mask = apply_morphology(best_crop_water_mask)
        final_full_mask[crop_y:, :] = cleaned_crop_mask
        calc_new_y, top_y = calculate_waterline_by_percentage(final_full_mask, h_orig)
        if calc_new_y is not None:
            new_waterline_y, new_mask_top_y = calc_new_y, top_y
            print(f"  [裁剪水体] 置信度:{score2:.3f}, Y={new_waterline_y}")

    final_y, final_top_y, final_score, final_thresh = -1, -1, None, None

    if ref_waterline_y != -1 and new_waterline_y != -1:
        if ref_waterline_y < new_waterline_y:
            print(f"  [决策] 原水体(Y={ref_waterline_y}) 更靠上，弃用裁剪。")
            final_y, final_top_y, final_score, final_thresh = ref_waterline_y, ref_mask_top_y, score1, thresh1
            final_full_mask = apply_morphology(ref_water_mask)
        else:
            print(f"  [决策] 裁剪后(Y={new_waterline_y}) 更靠上，采用裁剪。")
            final_y, final_top_y, final_score, final_thresh = new_waterline_y, new_mask_top_y, score2, thresh2
    elif new_waterline_y != -1:
        final_y, final_top_y, final_score, final_thresh = new_waterline_y, new_mask_top_y, score2, thresh2
    elif ref_waterline_y != -1:
        final_y, final_top_y, final_score, final_thresh = ref_waterline_y, ref_mask_top_y, score1, thresh1
        final_full_mask = apply_morphology(ref_water_mask)
    else:
        print("  ❌ 提取失败，未找到合法水体")
        if save_masks:
            empty_mask_path = Path(mask_output_folder) / f"{img_path.stem}_mask_empty.png"
            cv2.imwrite(str(empty_mask_path), np.zeros((h_orig, w_orig), dtype=np.uint8))
        return False, fname

    if save_masks and np.any(final_full_mask):
        mask_path = Path(mask_output_folder) / f"{img_path.stem}_mask_combined.png"
        cv2.imwrite(str(mask_path), (final_full_mask * 255).astype(np.uint8))

    out_img = image_np.copy()
    color = np.array([0, 0, 255])
    for c in range(3):
        out_img[:, :, c] = np.where(final_full_mask, out_img[:, :, c] * 0.6 + color[c] * 0.4, out_img[:, :, c])

    cv2.line(out_img, (0, final_y), (w_orig, final_y), (255, 0, 0), 2)
    cv2.putText(out_img, f"Waterline Y: {final_y}", (10, final_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    save_path = Path(output_folder) / f"{img_path.stem}_result.png"
    cv2.imwrite(str(save_path), out_bgr)

    return True, (img_path.stem, final_y, final_top_y, final_score, final_thresh)


# ==================== 关键帧掩码优化 ====================

def optimize_mask(mask, threshold_percent=30):
    """
    关键帧优化：先取水体主体（最大连通域），再从上往下找第一处宽度 >= 主体最大行宽
    threshold_percent% 的行为既定水位线，然后删除水位线上方掩码（清零），
    并将水位线以下（含该行）的掩码全部填满。
    这样掩码最上方坐标 == 水位线坐标，映射实际坐标时二者一致。
    以主体自身最大行宽为基准（而非整图全宽），可避免水面被岸体夹窄时水位线被推到底部。
    返回 (优化后掩码, 既定水位线 y)；无有效掩码返回 (None, None)。
    """
    if mask is None or not np.any(mask):
        return None, None
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return None, None
    h, w = mask_2d.shape

    main = keep_largest_connected_component(mask_2d.astype(bool))
    if main is None or not np.any(main):
        return mask_2d.copy(), None
    row_widths = main.sum(axis=1)
    max_row_width = int(row_widths.max())
    if max_row_width <= 0:
        return mask_2d.copy(), None
    threshold_count = max(1, int(round(max_row_width * threshold_percent / 100.0)))

    waterline_y = None
    for y in range(h):
        if int(row_widths[y]) >= threshold_count:
            waterline_y = y
            break
    if waterline_y is None:
        ys = np.where(main.any(axis=1))[0]
        if len(ys) == 0:
            return mask_2d.copy(), None
        waterline_y = int(ys.min())

    optimized = mask_2d.copy()
    optimized[:waterline_y, :] = False  # 删除水位线上方掩码
    optimized[waterline_y:, :] = True   # 水位线以下（含）填满
    return optimized, waterline_y


def sam3find(pil_image, model, device):
    """
    对图片做 'floating object' 语义分割，返回漂浮物掩码。
    在所有宽度 > 图片宽10% 且置信度 > 0.4 的物体中取置信度最大的一个。
    返回 (floating_mask, score)；未找到返回 (None, None)。
    """
    img_w = pil_image.size[0]
    img_h = pil_image.size[1]
    thresholds_to_try = [0.4, 0.3, 0.2, 0.1, 0.0]
    for threshold in thresholds_to_try:
        try:
            processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
        except:
            processor = Sam3Processor(model, confidence_threshold=threshold)
        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(state=state, prompt="floating object")
        if 'masks' not in state or state['masks'] is None:
            continue
        masks = state['masks']
        scores = state.get('scores', [])
        if len(masks) == 0:
            continue

        best_mask = None
        best_score = -1.0
        for i, mask in enumerate(masks):
            score = float(scores[i]) if scores is not None and len(scores) > 0 else 0.0
            if score <= 0.4:
                continue
            mask_np = mask.cpu().float().numpy() > 0.5 if isinstance(mask, torch.Tensor) else mask > 0.5
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim != 2:
                continue
            # SAM3 对奇数尺寸输出掩码宽度可能差 1，统一 resize 到输入图片尺寸
            if mask_np.shape[:2] != (img_h, img_w):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (img_w, img_h)).astype(bool)
            cols = np.where(mask_np.any(axis=0))[0]
            if len(cols) == 0:
                continue
            width = int(cols.max()) - int(cols.min()) + 1
            if width < img_w * 0.1:
                continue
            if score > best_score:
                best_score = score
                best_mask = mask_np
        if best_mask is not None:
            return best_mask, best_score
    return None, None


def get_water_mask_by_box_only(pil_image, model, device, box_y1, bank_box=None, box_y2=None):
    """纯几何框提示分割（不加文本提示）：用 MobileNet-SSD 检测框上沿 box_y1 与下沿 box_y2 构造水体正框，
    只靠框提示让 SAM3 分割框内水体，返回 (原始二值掩码, score)；失败返回 (None, None)。
    bank_box 为岸框 [xmin,ymin,xmax,ymax]（像素），作为负框提示排除岸体。
    掩码不做任何后处理（不删顶部连通域、不做形态学、不做底部累计），原样返回。"""
    img_w, img_h = pil_image.size
    y0 = max(0.0, min(float(box_y1), img_h - 1)) / float(img_h)
    if box_y2 is not None:
        y1 = max(y0, min(float(box_y2), img_h - 1)) / float(img_h)
    else:
        y1 = 1.0
    bottom_box = [0.5, (y0 + y1) / 2.0, 1.0, y1 - y0]
    bank_prompt = None
    if bank_box is not None:
        bank_prompt = _box_to_prompt(bank_box, img_w, img_h)
        print(f"  [纯框负框] 岸框 [xmin,ymin,xmax,ymax]={[round(v,1) for v in bank_box]}，下边缘 ymax={bank_box[3]:.0f}")

    for threshold in [0.5, 0.3, 0.0]:
        try:
            processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
        except Exception:
            processor = Sam3Processor(model, confidence_threshold=threshold)
        state = processor.set_image(pil_image)
        state = processor.add_geometric_prompt(bottom_box, True, state)
        if bank_prompt is not None:
            state = processor.add_geometric_prompt(bank_prompt, False, state)
        if 'masks' not in state or state['masks'] is None or len(state['masks']) == 0:
            continue

        masks = state['masks']
        scores = state.get('scores', [])
        sorted_masks, sorted_scores, _ = get_all_masks_sorted_by_confidence(masks, scores)
        for i, mask in enumerate(sorted_masks):
            mask_np = mask.cpu().float().numpy() > 0.5 if isinstance(mask, torch.Tensor) else mask > 0.5
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim != 2:
                continue
            if mask_np.shape[:2] != (img_h, img_w):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (img_w, img_h)).astype(bool)
            score = float(sorted_scores[i]) if sorted_scores is not None else None
            return mask_np, score
    return None, None


# ==================== 单张图片处理（PIL版本，用于视频帧） ====================

def process_single_image_video(pil_image, model, prompt, manual_crop_edge=None, ruler_csv=None):
    """
    处理单张 PIL 图像（视频帧），提取水位线，返回核心结果，不保存任何文件。
    manual_crop_edge: 人工指定的框提示上边缘（像素 y），非 None 时跳过自动动态框。
    ruler_csv: 当前场景的水尺 CSV，用于选择对应的 MobileNet-SSD 模型；
               为空字符串表示用户在用户端选择了「无边坡」——此时不加载/不调用 MobileNet-SSD，
               直接走纯语义提示分割（仅文本提示 "black water"，无正框、无负框）。
    返回：(success, final_y, final_mask, final_score, floating_mask, crop_edge, mask_b)

    蓝色掩码两路水位线对比，取最高（像素 y 最小）：
      1) 原图常规（语义"black water" + 正框水体 + 负框岸，完整后处理）；
      2) 原图纯框（仅当 SSD 水体框置信度 > 0.3：不加文本、只靠框提示分割，掩码原样不做后处理）。
      无边坡模式下只有第 1 路（且不带任何框），第 2 路与 SSD 兜底逻辑全部跳过。
    mask_b 为最终采用那一路的掩码B（用于红色叠加对比）。
    """
    device = next(model.parameters()).device if hasattr(model, 'parameters') else "cuda"
    crop_edge = None

    try:
        pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image)
        h_orig, w_orig = image_np.shape[:2]
    except Exception as e:
        print(f"  ❌ 无法转图片: {e}")
        return False, -1, None, -1.0, None, None, None

    # 漂浮物检测：'floating object' 语义分割
    floating_mask, floating_score = sam3find(pil_image, model, device)
    if floating_mask is not None:
        print(f"  [漂浮物] 检测到漂浮物，置信度: {floating_score:.3f}")
    else:
        print("  [漂浮物] 未检测到漂浮物")

    # SSD 框提示：用 MobileNet-SSD 水体框定位水位线裁剪上边缘；人工指定时优先使用
    ssd_confidence = None
    ssd_box_y1 = None
    ssd_box_y2 = None
    ssd_bank_box = None
    no_slope = is_no_slope_mode(ruler_csv)
    if manual_crop_edge is not None:
        crop_edge = max(0, int(manual_crop_edge))
        print(f"  [手动框] 人工框上边缘: {crop_edge}px")
    elif no_slope:
        # 无边坡：完全不加载/不调用 MobileNet-SSD，直接语义提示分割
        print("  [动态框] 无边坡模式：跳过 MobileNet-SSD，直接语义提示分割（无框）")
    else:
        _src, crop_edge, ssd_confidence, ssd_box_y1, ssd_box_y2, ssd_bank_box = compute_dynamic_crop_edge(image_np, floating_mask, ruler_csv)
        if crop_edge is not None:
            print(f"  [SSD框提示] 裁剪上边缘: {crop_edge}px")

    # 第一轮：原图水体提取
    ref_water_mask, score1, thresh1, ref_mask_b = get_water_mask_pipeline(pil_image, model, device, prompt=prompt, is_cropped=False, crop_edge=crop_edge, floating_mask=floating_mask, return_mask_b=True, bank_box=ssd_bank_box, box_y2=ssd_box_y2)
    ref_waterline_y, ref_mask_top_y = -1, -1

    if ref_water_mask is not None:
        cleaned_ref_mask = apply_morphology(ref_water_mask)
        y_top = waterline_by_width(cleaned_ref_mask, threshold_percent=30, min_continuous_rows=3)
        if y_top is not None:
            ref_waterline_y, ref_mask_top_y = y_top, y_top
            print(f"  [原水体] 置信度:{score1:.3f}, Y={ref_waterline_y}")

    # 第一轮第二次：SSD 模型框置信度 > 0.3 时，纯框提示分割（掩码原样，不做后处理）
    box_water_mask = None
    box_waterline_y, box_mask_top_y = -1, -1
    box_score = None
    if manual_crop_edge is not None:
        print("  [纯框水体] 未分割：使用了人工框，跳过纯框轮")
    elif no_slope:
        print("  [纯框水体] 跳过：无边坡模式只走语义提示分割，不做纯框轮")
    elif ssd_confidence is None or ssd_box_y1 is None:
        print("  [纯框水体] 未分割：SSD 未检出水体框")
    elif ssd_confidence <= 0.3:
        print(f"  [纯框水体] 未分割：SSD 水体框置信度 {ssd_confidence:.2f} ≤ 0.3")
    else:
        box_water_mask, box_score = get_water_mask_by_box_only(pil_image, model, device, ssd_box_y1, bank_box=ssd_bank_box, box_y2=ssd_box_y2)
        if box_water_mask is None or not np.any(box_water_mask):
            print("  [纯框水体] 未分割：纯框提示分割未产出掩码")
        else:
            box_top_edge = max(0, min(int(ssd_box_y1), h_orig - 1))
            below_mask = box_water_mask[box_top_edge:, :]
            y_rel = waterline_by_width(below_mask, threshold_percent=30, min_continuous_rows=3)
            if y_rel is None:
                print("  [纯框水体] 未分割：框下方无掩码")
            else:
                box_waterline_y = y_rel + box_top_edge
                box_mask_top_y = box_waterline_y
                print(f"  [纯框水体] SSD置信度:{ssd_confidence:.2f}, 水位线 Y={box_waterline_y}")

    # 两轮水位线对比：原图常规 / 原图纯框，取最高（像素 y 最小）者作为最终蓝色掩码
    candidates = []
    if ref_waterline_y != -1:
        candidates.append((ref_waterline_y, cleaned_ref_mask, score1, ref_mask_b, 'ref'))
    if box_waterline_y != -1:
        candidates.append((box_waterline_y, box_water_mask, box_score, box_water_mask, 'box'))

    if not candidates:
        print("  ❌ 提取失败，未找到合法水体")
        return False, -1, None, -1.0, None, crop_edge, ref_mask_b

    src_name = {'ref': '原水体', 'box': '纯框水体'}
    final_y, final_mask, final_score, final_mask_b, final_src = min(candidates, key=lambda t: t[0])
    print(f"  [决策] 两轮对比取最高水位线: {src_name[final_src]}(Y={final_y})")

    # 兜底：若最终水位线明显低于 SSD 水体框上沿（不合理），回退到 SSD 水体框上沿
    if ssd_box_y1 is not None and final_y > ssd_box_y1 + 30:
        print(f"  [决策] 水位线 {final_y} 明显低于 SSD 水体框上沿 {ssd_box_y1}，回退到框上沿")
        final_y = ssd_box_y1
        final_mask = np.zeros((h_orig, w_orig), dtype=bool)
        final_mask[int(ssd_box_y1):, :] = True
        final_src = 'ssd'

    if final_mask is None:
        print("  ❌ 提取失败，未找到合法mask")
        return False, -1, None, -1.0, None, crop_edge, ref_mask_b

    # 漂浮物裁剪：把漂浮物区域从最终掩码里去掉（纯框提示的原样掩码除外）
    if floating_mask is not None and final_src != 'box':
        final_mask = final_mask & ~floating_mask

    return True, final_y, final_mask, final_score, floating_mask, crop_edge, final_mask_b


# ==================== 批量图片处理函数 ====================

def get_image_list(input_folder):
    image_files = set()
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.GIF']
    for ext in extensions:
        image_files.update(Path(input_folder).glob(ext))
    return sorted(list(image_files))


def process_all_images(input_folder, model, prompt, gt_dicts, output_folder, mask_output_folder, save_masks):
    """
    批量处理输入文件夹中的所有图片，依次调用 process_single_image，
    汇总成功与失败结果，并返回图片列表、成功结果和失败列表。
    """
    image_files = get_image_list(input_folder)
    print(f"\n找到 {len(image_files)} 张图片")
    success_results, failed = [], []
    for img_path in image_files:
        try:
            success, res = process_single_image(img_path, model, prompt, gt_dicts, output_folder, mask_output_folder,
                                                save_masks)
            if success:
                success_results.append(res)
            else:
                failed.append(res)
        except Exception as e:
            print(f"  错误: {img_path.name} - {e}")
            traceback.print_exc()
            failed.append(img_path.name)

    return image_files, success_results, failed