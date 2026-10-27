# sam3video.py
"""
SAM3 视频分割与水线传播模块
封装视频读取、单帧水位线识别、关键帧筛选、视频传播、渲染导出等功能。
所有与视频处理相关的函数均在此模块中定义，便于复用与维护。
"""

import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image
import multiprocessing as mp

# 导入 sam3first 中的视频帧处理函数
from sam3first import process_single_image_video, mask_satisfies_constraints

# 导入 time_tracker 中的关键帧筛选函数
from time_tracker import filter_n_to_best_3

# 导入 SAM3 模型构建函数
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor


# ==================== 视频输出与可视化 ====================

def save_frames_as_mp4(frames, output_path, fps=30):
    """将帧列表保存为 MP4 视频文件"""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    return True


def get_waterline_pixel_from_mask(mask, threshold_percent=60):
    """从单帧掩码计算像素水位线高度：从上往下找第一个掩码宽度 >= 总宽60% 的行。"""
    if mask is None:
        return None
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2 or not np.any(mask_2d):
        return None
    h, w = mask_2d.shape
    threshold_count = int(w * threshold_percent / 100)
    for y in range(h):
        if int(np.sum(mask_2d[y, :])) >= threshold_count:
            return y
    return None


def compute_avg_waterline_pixel(frames, outputs_per_frame, n_measure=10, threshold_percent=60):
    """处理完视频后，等距抽取 n_measure 帧，计算各帧像素水位线高度并取平均。"""
    if not frames:
        return None
    n = len(frames)
    indices = np.linspace(0, n - 1, min(n_measure, n), dtype=int)

    heights = []
    for i in indices:
        out = outputs_per_frame.get(i)
        if not out:
            continue
        masks = out.get('out_binary_masks', [])
        if masks is None or len(masks) == 0:
            continue

        merged = None
        for m in masks:
            m = np.squeeze(m)
            if m.ndim != 2:
                continue
            m_bool = m > 0.5 if m.dtype in (np.float32, np.float64) else m > 0
            merged = m_bool if merged is None else (merged | m_bool)

        y = get_waterline_pixel_from_mask(merged, threshold_percent)
        if y is not None:
            heights.append(y)

    if not heights:
        return None
    return float(np.mean(heights))


def overlay_mask(frame, outputs, alpha=0.5):
    """
    将 SAM3 输出的掩码以半透明红色叠加到原始帧上
    :param frame: BGR 图像 (numpy array)
    :param outputs: SAM3 输出的字典，需包含 'out_binary_masks'
    :param alpha: 叠加透明度 (0~1)
    :return: 叠加后的 BGR 图像
    """
    if not outputs:
        return frame
    overlay = frame.copy()
    h, w = frame.shape[:2]
    masks = outputs.get('out_binary_masks', [])
    for mask in masks:
        if len(mask.shape) == 3:
            mask = mask.squeeze()
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[:, :, c][mask_bool] = (alpha * 255 * (c == 0) + (1 - alpha) * frame[:, :, c][mask_bool]).astype(
                np.uint8)
    return overlay


def save_per_target_overlays(frame, frame_idx, outputs, video_name, save_dir, alpha=0.5):
    """
    将单帧中识别到的每个目标分别保存为掩码叠加原帧的图片
    命名格式：video_name_f000_obj0.png
    """
    if not outputs:
        return
    os.makedirs(save_dir, exist_ok=True)
    h, w = frame.shape[:2]
    masks = outputs.get('out_binary_masks', [])
    out_obj_ids = outputs.get('out_obj_ids', [])
    if len(masks) == 0:
        print(f"    帧 {frame_idx}: 无目标，未保存图片")
        return

    for obj_id, mask in zip(out_obj_ids, masks):
        overlay = frame.copy()
        if len(mask.shape) == 3:
            mask = mask.squeeze()
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.uint8)
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[:, :, c][mask_bool] = (alpha * 255 * (c == 0) + (1 - alpha) * frame[:, :, c][mask_bool]).astype(
                np.uint8)
        filename = f"{video_name}_f{frame_idx:03d}_obj{int(obj_id)}.png"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, overlay)
        print(f"    已保存目标图片: {save_path}")


# ==================== SAM3 视频传播 ====================

def propagate_in_video(predictor, session_id):
    """
    使用 SAM3 视频预测器进行标准 detector 传播
    :param predictor: SAM3 video predictor 实例
    :param session_id: 会话 ID
    :return: 字典 {frame_index: outputs}
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id)):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def propagate_in_video_tracker_only(predictor, session_id):
    """
    使用 SAM3 视频预测器进行 tracker-only 双向传播
    :param predictor: SAM3 video predictor 实例
    :param session_id: 会话 ID
    :return: 字典 {frame_index: outputs}
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
            request=dict(type="propagate_in_video_tracker_only", session_id=session_id, propagation_direction="both")):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def log_gpu_memory(tag=""):
    """打印当前 GPU 显存占用（仅在 CUDA 可用时）"""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() // 1024**2
    reserved = torch.cuda.memory_reserved() // 1024**2
    prefix = f"[{tag}] " if tag else ""
    print(f"  {prefix}GPU 显存: allocated={allocated} MiB, reserved={reserved} MiB")


# ==================== 关键帧提取 ====================

def save_mask_overlay(frame, mask, out_path, color=(0, 0, 255), alpha=0.5):
    """把掩码半透明叠加到原图并保存。"""
    mask_bool = np.squeeze(mask) > 0.5
    if mask_bool.ndim != 2:
        return
    h, w = frame.shape[:2]
    if mask_bool.shape[:2] != (h, w):
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h)).astype(bool)
    overlay = frame.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask_bool,
                                    overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                                    overlay[:, :, c])
    cv2.imwrite(out_path, overlay)


def get_box_prompts_from_frames(frames, image_model, frame_indices, image_paths=None, all_output_dir=None, ruler_csv=None):
    """
    对指定帧索引列表分别调用单帧水位线识别，返回各帧像素Y坐标、框提示、mask及score。
    使用 sam3first.process_single_image_video 处理每一帧。
    image_paths 提供时，直接用原始图片（无损）做单帧识别。
    all_output_dir 提供时，输出关键帧/单帧分割/优化后关键帧到该目录。
    """
    waterline_ys = []
    box_prompts = []
    masks = []
    raw_masks = []
    bboxes = []
    scores = []
    kept_frame_indices = []
    fallback_pool = []  # 保存所有识别成功帧的结果，用于全部剔除时的回退
    for frame_idx in frame_indices:
        # 优先用原始图片（无损），否则回退到视频帧
        if image_paths is not None and frame_idx < len(image_paths):
            frame = cv2.imread(image_paths[frame_idx])
            if frame is None:
                frame = frames[frame_idx]
        else:
            frame = frames[frame_idx]

        # 输出抽样帧原图到 all 的 keyframes（无论识别成败）
        if all_output_dir is not None:
            try:
                kf_dir = os.path.join(all_output_dir, "keyframes")
                os.makedirs(kf_dir, exist_ok=True)
                cv2.imwrite(os.path.join(kf_dir, f"frame_{frame_idx:03d}.jpg"), frame)
            except Exception as e:
                print(f"  ⚠️ 保存关键帧原图失败: {e}")

        # 将 BGR 帧转换为 PIL 图像
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, final_y, final_mask, final_score, floating_mask, crop_edge, mask_b = process_single_image_video(
            pil_image=pil_frame,
            model=image_model,
            prompt="black water",
            ruler_csv=ruler_csv
        )

        if not success:
            print(f"  ⚠️ 帧 {frame_idx} 水位线自动识别失败，跳过该帧")
            continue

        # 漂浮物可视化：保存漂浮物掩码叠加图（青色）
        if all_output_dir is not None and floating_mask is not None:
            float_dir = os.path.join(all_output_dir, "floating")
            os.makedirs(float_dir, exist_ok=True)
            save_mask_overlay(frame, floating_mask, os.path.join(float_dir, f"frame_{frame_idx:03d}_floating.jpg"), color=(0, 255, 255))

        # 记录优化前掩码（需求4用）
        raw_mask_before_opt = final_mask

        # 计算水位线、归一化框、bbox（约束检查前，供 fallback 使用）
        waterline_y = final_y
        h = frame.shape[0]
        y_top = waterline_y
        y_bottom = h
        y_norm = (y_top + y_bottom) / 2.0 / h
        h_norm = (y_bottom - y_top) / h
        frame_box = [[0.5, y_norm, 1.0, h_norm]]

        # 从 mask 计算 bbox [x1, y1, x2, y2]
        ys, xs = np.where(final_mask)
        if len(ys) > 0:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            bbox = [0, waterline_y, frame.shape[1] - 1, h - 1]

        # 保存到 fallback_pool（无论是否满足约束）
        fallback_pool.append((frame_idx, waterline_y, frame_box, final_mask, bbox, final_score, raw_mask_before_opt))

        # 优化后仍须满足体积约束，否则剔除该帧
        if not mask_satisfies_constraints(final_mask, frame.shape[1], frame.shape[0], crop_edge):
            print(f"  ⚠️ 帧 {frame_idx} 优化后掩码违反约束(体积不足10%)，剔除该帧")
            continue

        raw_masks.append(raw_mask_before_opt)
        waterline_ys.append(waterline_y)
        box_prompts.append(frame_box)
        masks.append(final_mask)
        bboxes.append(bbox)
        scores.append(final_score)
        kept_frame_indices.append(frame_idx)
        print(f"  - 帧 {frame_idx} 水位线 Y 像素坐标: {waterline_y}, 归一化框: {frame_box}, bbox: {bbox}, score: {final_score:.3f}")

    if not kept_frame_indices:
        if not fallback_pool:
            raise ValueError("所有抽样帧识别均失败，无法生成关键帧 Prompt")
        print("所有抽样帧均不满足约束，回退到第一个/中间/最后一个关键帧")
        n = len(fallback_pool)
        if n == 1:
            picks = [fallback_pool[0]]
        elif n == 2:
            picks = [fallback_pool[0], fallback_pool[-1]]
        else:
            picks = [fallback_pool[0], fallback_pool[n // 2], fallback_pool[-1]]
        for fi, wy, fb, m, b, sc, rm in picks:
            kept_frame_indices.append(fi)
            waterline_ys.append(wy)
            box_prompts.append(fb)
            masks.append(m)
            raw_masks.append(rm)
            bboxes.append(b)
            scores.append(sc)
            print(f"  [回退关键帧] 帧 {fi} 水位线 Y: {wy}")
    return kept_frame_indices, waterline_ys, box_prompts, masks, raw_masks, bboxes, scores


# ==================== 单个视频处理 ====================

def process_single_video(v_path, out_path, v_name, checkpoint_path, save_target_folder, n_sample_frames=5, ruler_csv=None, image_model=None, video_predictor=None, image_paths=None, all_output_dir=None):
    print(f"  [sam3video] process_single_video 收到 ruler_csv = {ruler_csv!r}")
    """
    处理单个视频。传入 image_model / video_predictor 时复用常驻模型（不加载也不释放），
    否则自行加载并在结束时释放。
    """
    own_image_model = (image_model is None)
    own_video_predictor = (video_predictor is None)

    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"  ❌ 视频 {v_name} 无有效帧，跳过。")
        return False, None

    last_idx = len(frames) - 1
    n_actual = min(n_sample_frames, len(frames))
    sample_indices = sorted(set(np.linspace(0, last_idx, n_actual, dtype=int).tolist()))
    print(f"  - 视频共 {len(frames)} 帧，等间距抽取 {len(sample_indices)} 帧: {sample_indices}")

    session_id = None

    waterline_ys = []
    box_prompts = []
    masks = []
    bboxes = []
    single_frame_scores = []
    outputs_per_frame = {}
    result_frames = []

    try:
        if own_image_model:
            print("  正在加载 SAM3 Image 模型进行单帧识别...")
            log_gpu_memory("加载 Image 模型前")
            image_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            log_gpu_memory("加载 Image 模型后")
        else:
            print("  复用常驻 SAM3 Image 模型进行单帧识别...")

        frame_indices, waterline_ys, box_prompts, masks, raw_masks, bboxes, single_frame_scores = get_box_prompts_from_frames(
            frames, image_model, sample_indices, image_paths=image_paths, all_output_dir=all_output_dir, ruler_csv=ruler_csv
        )
        print(f"  - 已获取各帧水位线 Y 坐标: {waterline_ys}")

        # 使用 time_tracker 中的均值聚类筛选，保留最优 3 帧
        frame_indices, waterline_ys, box_prompts, masks, raw_masks, bboxes, single_frame_scores = filter_n_to_best_3(
            frame_indices, waterline_ys, box_prompts, masks, raw_masks, bboxes, single_frame_scores
        )

        # 需求 2 + 4 + 5：输出最终 3 个关键帧原图、单帧分割（优化前）、优化后关键帧
        if all_output_dir is not None:
            kf_dir = os.path.join(all_output_dir, "keyframes")
            seg_dir = os.path.join(all_output_dir, "single_frame_seg")
            opt_dir = os.path.join(all_output_dir, "optimized_keyframes")
            os.makedirs(kf_dir, exist_ok=True)
            os.makedirs(seg_dir, exist_ok=True)
            os.makedirs(opt_dir, exist_ok=True)
            for fi, mask, raw_mask in zip(frame_indices, masks, raw_masks):
                frame = frames[fi]
                cv2.imwrite(os.path.join(kf_dir, f"frame_{fi:03d}.jpg"), frame)
                save_mask_overlay(frame, raw_mask, os.path.join(seg_dir, f"frame_{fi:03d}_seg.jpg"))
                save_mask_overlay(frame, mask, os.path.join(opt_dir, f"frame_{fi:03d}_optimized.jpg"))

        if own_image_model:
            print("  正在释放 SAM3 Image 模型以释放内存...")
            del image_model
            image_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            log_gpu_memory("释放 Image 模型后")

        if own_video_predictor:
            print("  正在加载 SAM3 Video 预测模型...")
            log_gpu_memory("加载 Video 模型前")
            video_predictor = build_sam3_video_predictor(checkpoint_path)
            print("  SAM3 Video 模型加载完毕。")
            log_gpu_memory("加载 Video 模型后")
        else:
            print("  复用常驻 SAM3 Video 预测模型...")

        session_id = video_predictor.handle_request(
            dict(type="start_session", resource_path=v_path)
        )["session_id"]

        target_save_dir = save_target_folder
        tracker_only_failed = False

        try:
            # 使用单帧分割 mask 初始化 tracker masklet
            seeds = [
                {
                    "frame_idx": fi,
                    "mask": mask,
                    "bbox": bbox,
                    "score": float(score),
                }
                for fi, mask, bbox, score in zip(frame_indices, masks, bboxes, single_frame_scores)
            ]
            video_predictor.handle_request(dict(
                type="init_video_from_single_frame_seeds",
                session_id=session_id,
                seeds=seeds,
            ))
            print(f"  - 已用单帧分割 mask 初始化 {len(seeds)} 个关键帧的 tracker masklet")

            print("  - 关键帧分割结果（来自单帧分割）：")
            for fi, mask, bbox in zip(frame_indices, masks, bboxes):
                num_targets = 1 if mask.any() else 0
                print(f"    帧 {fi}: bbox {bbox}, 识别到目标数: {num_targets}")
                if num_targets > 0:
                    seed_outputs = {
                        "out_obj_ids": np.array([0], dtype=np.int64),
                        "out_binary_masks": mask.astype(np.uint8)[None, ...],
                    }
                    save_per_target_overlays(frames[fi], fi, seed_outputs, v_name, target_save_dir)

            print("  - 开始 tracker-only 时序传播...")
            outputs_per_frame = propagate_in_video_tracker_only(video_predictor, session_id)

        except Exception as e:
            tracker_only_failed = True
            print(f"  ⚠️ tracker-only 流失败，回退到原 detector 流: {e}")
            # 关闭旧会话，重开新会话
            try:
                video_predictor.handle_request(dict(type="close_session", session_id=session_id))
            except Exception:
                pass
            # 彻底清空所有 inference states，释放显存
            try:
                video_predictor._ALL_INFERENCE_STATES.clear()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            log_gpu_memory("重置 Session 后")
            session_id = video_predictor.handle_request(
                dict(type="start_session", resource_path=v_path)
            )["session_id"]

        if tracker_only_failed:
            # 回退到 detector 流，使用修正后的 box_prompts
            prompts = [
                {
                    "frame_idx": fi,
                    "text_str": "water",
                    "boxes_xywh": box,
                    "box_labels": [1],
                }
                for fi, box in zip(frame_indices, box_prompts)
            ]
            add_prompts_response = video_predictor.handle_request(dict(
                type="add_prompts", session_id=session_id, prompts=prompts
            ))
            print(f"  - 已一次性注入 {len(prompts)} 帧提示: water + {box_prompts}")

            outputs_per_frame = add_prompts_response["outputs_per_frame"]
            print("  - 关键帧分割结果：")
            for fi, box in zip(frame_indices, box_prompts):
                outputs = outputs_per_frame.get(fi, {})
                out_obj_ids = outputs.get("out_obj_ids", [])
                num_targets = len(out_obj_ids)
                print(f"    帧 {fi}: 提示框 {box}，识别到目标数: {num_targets}")
                save_per_target_overlays(frames[fi], fi, outputs, v_name, target_save_dir)

            print("  - 开始原 detector 时序传播...")
            outputs_per_frame = propagate_in_video(video_predictor, session_id)

        # 渲染与导出
        result_frames = []
        fallback_output = outputs_per_frame.get(0, None)
        for i, frame in enumerate(frames):
            out = outputs_per_frame.get(i, fallback_output)
            result_frames.append(overlay_mask(frame, out))

        save_frames_as_mp4(result_frames, out_path, fps)
        print(f"  ✅ 视频导出成功: {out_path}")

        real_waterline = None
        avg_pixel = compute_avg_waterline_pixel(frames, outputs_per_frame)
        if avg_pixel is not None:
            print(f"  视频平均像素水位线高为：{avg_pixel}")
            print(f"  [sam3video] 准备计算真实水位线，ruler_csv = {ruler_csv!r}")
            if ruler_csv:
                try:
                    import ruler
                    print("  [sam3video] import ruler 成功")
                    frame_h = frames[0].shape[0]
                    real_waterline = ruler.pixel_to_real_waterline(
                        avg_pixel, ruler_csv, image_height=frame_h, original_image_height=240
                    )
                    print(f"  视频真实水位线：{real_waterline}")
                except Exception as e:
                    print(f"  视频真实水位线计算失败: {e}")
            else:
                print("  [sam3video] ruler_csv 为空，跳过真实水位线计算")
        else:
            print("  视频平均像素水位线高为：N/A")

        return True, real_waterline

    except Exception as e:
        print(f"  ❌ 视频处理失败: {e}")
        # 分割失败也输出原视频，保证有结果文件
        try:
            if frames:
                out_fps = fps if fps and fps > 0 else 25
                save_frames_as_mp4(frames, out_path, out_fps)
                print(f"  ⚠️ 分割失败，已输出原视频: {out_path}")
        except Exception as e2:
            print(f"  输出原视频也失败: {e2}")
        return False, None

    finally:
        # 清理 session（每次都要，复用模型也要关 session）
        if session_id is not None and video_predictor is not None:
            try:
                video_predictor.handle_request(dict(type="close_session", session_id=session_id))
            except Exception:
                pass

        if video_predictor is not None:
            try:
                video_predictor._ALL_INFERENCE_STATES.clear()
            except Exception:
                pass

        # 只有自己加载的模型才移到 CPU 释放（复用的不释放）
        if own_video_predictor and video_predictor is not None:
            try:
                video_predictor.model.cpu()
            except Exception:
                pass
        if own_image_model and image_model is not None:
            try:
                image_model.cpu()
            except Exception:
                pass

        print("  正在清理资源（帧缓存、显存）...")
        log_gpu_memory("清理前")

        # del 局部引用；复用模型时 worker 仍持有引用，实际不释放
        try:
            del video_predictor, image_model, frames, outputs_per_frame
            del result_frames, masks, bboxes, box_prompts, waterline_ys
            del single_frame_scores
        except NameError:
            pass
        try:
            del seeds, prompts, add_prompts_response, fallback_output
        except NameError:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if own_image_model and own_video_predictor:
                torch.cuda.reset_peak_memory_stats()
        log_gpu_memory("清理后")


# ==================== 多进程辅助函数 ====================

def _video_worker(v_path, out_path, v_name, checkpoint_path, save_target_folder, n_sample_frames, result_queue, ruler_csv=None):
    """子进程入口：在独立进程中处理单个视频，进程退出后由 OS 彻底回收全部显存。"""
    try:
        success, avg_elevation = process_single_video(
            v_path, out_path, v_name, checkpoint_path, save_target_folder,
            n_sample_frames=n_sample_frames, ruler_csv=ruler_csv
        )
        result_queue.put((success, avg_elevation))
    except Exception:
        import traceback
        traceback.print_exc()
        result_queue.put((False, None))


# ==================== 批量处理入口 ====================

def batch_process_videos_auto(input_dir, output_dir, checkpoint_path, save_target_folder, n_sample_frames=7):
    """
    视频批量自动化分割流水线（子进程隔离模式）。
    每个视频在独立子进程中处理，进程退出后由操作系统强制回收全部 GPU 显存。
    """
    print("\n" + "=" * 60)
    print("🚀 启动视频自动化分割流水线 (子进程隔离模式)")
    print(f"   等间距抽取 {n_sample_frames} 帧，均值聚类筛选最优3帧")
    print("   每个视频在独立子进程中运行，退出后显存由 OS 彻底回收")
    print("=" * 60)
    log_gpu_memory("主进程启动时")

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    if not video_files:
        print("❌ 未在输入目录找到 .mp4 文件！")
        return

    for idx, v_name in enumerate(video_files, 1):
        print(f"\n▶️ 正在处理视频 ({idx}/{len(video_files)}): {v_name}")
        v_path = os.path.join(input_dir, v_name)
        out_path = os.path.join(output_dir, v_name)

        log_gpu_memory(f"视频 {v_name} 处理前(主进程)")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        p = ctx.Process(
            target=_video_worker,
            args=(v_path, out_path, v_name, checkpoint_path, save_target_folder,
                  n_sample_frames, result_queue)
        )
        p.start()
        p.join()

        if p.exitcode == 0:
            try:
                res = result_queue.get(timeout=10)
                success = res[0] if isinstance(res, tuple) else bool(res)
            except Exception:
                success = False
            if not success:
                print(f"  ⚠️ 视频 {v_name} 处理未成功")
        else:
            print(f"  ⚠️ 子进程异常退出 (exitcode={p.exitcode})")

        p.close()
        log_gpu_memory(f"视频 {v_name} 处理后(主进程)")

    print("\n🎉 所有视频处理完毕！")
    log_gpu_memory("全部结束时(主进程)")