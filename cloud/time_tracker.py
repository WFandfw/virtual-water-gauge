import pandas as pd
import numpy as np
import re


def detect_temporal_anomalies(success_results, poly_degree=3, threshold_pixels=15):
    """
    基于迭代贪心剔除的全局曲线拟合，并用最终纯净函数预测异常点坐标。
    修改点：将时间序号提取改为提取文件名中的最后一个数字（通常为帧号）。
    """
    if not success_results:
        return {}

    df = pd.DataFrame(success_results,
                      columns=['filename', 'waterline_y', 'mask_top_y', 'confidence', 'used_threshold'])

    # 1. 提取时间序号并排序
    def extract_num(name):
        nums = re.findall(r'\d+', str(name))
        # 取最后一个数字作为时间序号（帧号），避免视频名中的数字干扰
        return int(nums[-1]) if nums else 0

    df['time_idx'] = df['filename'].apply(extract_num)
    df = df.sort_values('time_idx').reset_index(drop=True)

    X_all = df['time_idx'].values
    Y_all = df['waterline_y'].values
    filenames = df['filename'].values

    # 维护一个布尔数组，记录哪些数据是“干净”的
    active_mask = np.ones(len(df), dtype=bool)
    anomalies_info = []  # 记录被剔除的数据

    iteration = 1

    # ========================================================
    # 核心算法一：循环逐次剔除误差最大点 (不预测，只记录)
    # ========================================================
    while True:
        X_active = X_all[active_mask]
        Y_active = Y_all[active_mask]
        active_indices = np.where(active_mask)[0]

        n_samples = len(X_active)

        if n_samples < 3:
            break

        degree = poly_degree if n_samples > poly_degree + 1 else max(1, n_samples - 1)

        coeffs = np.polyfit(X_active, Y_active, degree)
        Y_fit = np.polyval(coeffs, X_active)

        deviations = np.abs(Y_active - Y_fit)

        max_dev_idx = np.argmax(deviations)
        max_dev = deviations[max_dev_idx]

        if max_dev > threshold_pixels:
            real_idx = active_indices[max_dev_idx]
            active_mask[real_idx] = False  # 永久移出干净数据池

            anomalies_info.append({
                'iteration': iteration,
                'filename': filenames[real_idx],
                'time_idx': X_all[real_idx],
                'deviation': max_dev,
                'waterline': Y_all[real_idx]
            })
            iteration += 1
        else:
            break

    # ========================================================
    # 核心算法二：使用绝对纯净数据拟合终局函数，并预测坐标
    # ========================================================
    X_clean = X_all[active_mask]
    Y_clean = Y_all[active_mask]
    final_degree = poly_degree if len(X_clean) > poly_degree + 1 else max(1, len(X_clean) - 1)

    if len(X_clean) >= 2:
        final_coeffs = np.polyfit(X_clean, Y_clean, final_degree)
    else:
        final_coeffs = np.polyfit(X_all, Y_all, 1)  # 极端情况保底

    print("\n" + "=" * 65)
    print("📈 【时序追踪：迭代式曲线拟合诊断报告】")

    predicted_anomalies = {}
    if anomalies_info:
        print(f"⚠️ 经过 {iteration - 1} 轮逐次筛查，共抓出 {len(anomalies_info)} 张异常图片：")
        for info in anomalies_info:
            # 【关键点】用剔除所有杂质后的 final_coeffs 来预测期望坐标
            expected_y = int(np.polyval(final_coeffs, info['time_idx']))
            predicted_anomalies[info['filename']] = expected_y

            print(
                f"   [剔除] {info['filename']:<15} | 错误测值: {info['waterline']:<4} | 终局曲线基准: {expected_y:<4} | 偏离: {info['deviation']:.1f} 像素")

        print("\n🔎 (注：预测坐标均由剔除噪点后的纯净物理曲线方程得出)")
    else:
        print("✅ 一次性通过！所有测量值均严丝合缝地贴合整体水位演变曲线。")
    print("=" * 65 + "\n")

    return predicted_anomalies


def filter_n_to_best_3(frame_indices, waterline_ys, box_prompts, masks, raw_masks, bboxes, scores, poly_degree=3):
    """
    基于水位线均值的动态聚类筛选，最终保留离均值最近的 3 帧。
    此函数与原 main 中的实现完全一致，用于从 N 个抽样帧中选出最佳 3 帧。

    参数：
        frame_indices: 原始帧索引列表
        waterline_ys:  对应水位线 Y 坐标列表
        box_prompts:   对应框提示列表
        masks:         对应掩码列表
        bboxes:        对应边界框列表
        scores:        对应置信度列表
        poly_degree:   未使用，仅为兼容保留

    返回：
        6 个列表：最佳 3 帧的帧索引、水位线、框提示、掩码、bbox、score
    """
    if len(frame_indices) <= 3:
        return frame_indices, waterline_ys, box_prompts, masks, raw_masks, bboxes, scores

    # 转为 numpy 便于计算
    fi_arr = np.array(frame_indices)
    y_arr = np.array(waterline_ys, dtype=float)

    # 保留原始各数据列表，方便按索引筛选
    all_fi = list(frame_indices)
    all_y = list(waterline_ys)
    all_bp = list(box_prompts)
    all_masks = list(masks)
    all_raw_masks = list(raw_masks)
    all_bboxes = list(bboxes)
    all_scores = list(scores)

    # 当前候选帧的索引（在原始列表中的位置）
    active_idx = list(range(len(all_fi)))
    round_num = 0
    removed_log = []

    while True:
        round_num += 1
        current_y = [all_y[i] for i in active_idx]
        if len(current_y) <= 3:
            break

        mean_y = np.mean(current_y)
        left_limit = mean_y - 20
        right_limit = mean_y + 20

        # 划分三区
        left_idx = [i for i in active_idx if all_y[i] < left_limit]
        center_idx = [i for i in active_idx if left_limit <= all_y[i] <= right_limit]
        right_idx = [i for i in active_idx if all_y[i] > right_limit]

        # 各区帧数
        cnt_left = len(left_idx)
        cnt_center = len(center_idx)
        cnt_right = len(right_idx)

        # 选择保留区域（帧数最多，并列时优先级：center > right > left）
        max_cnt = max(cnt_left, cnt_center, cnt_right)
        if cnt_center == max_cnt:
            keep_idx = center_idx
            region_name = "中间区"
        elif cnt_right == max_cnt:
            keep_idx = right_idx
            region_name = "右侧区"
        else:
            keep_idx = left_idx
            region_name = "左侧区"

        # 记录本轮剔除的帧（用于日志）
        removed_this_round = [i for i in active_idx if i not in keep_idx]
        for i in removed_this_round:
            removed_log.append((i, all_y[i], region_name))

        # 更新 active_idx
        active_idx = keep_idx

        # 若保留区域帧数等于本轮总帧数，说明其他两区为空，停止迭代
        if len(keep_idx) == len(active_idx):
            break

        # 防止无限循环（通常不会发生）
        if round_num > 50:
            break

    # 最终候选
    final_indices = active_idx
    final_y = [all_y[i] for i in final_indices]
    final_fi = [all_fi[i] for i in final_indices]

    # 打印筛选日志
    print("\n" + "=" * 55)
    print(f"📈 [聚类筛选] 原始 {len(all_fi)} 帧，经 {round_num} 轮迭代后保留 {len(final_indices)} 帧")
    if removed_log:
        print("  逐轮剔除详情：")
        for i, y_val, region in removed_log:
            print(f"    剔除 帧{all_fi[i]:>5}  Y={y_val:>4.0f}  (属于{region})")
    print("=" * 55 + "\n")

    # 若最终帧数 > 3，取离当前均值最近的 3 帧
    if len(final_indices) > 3:
        mean_final = np.mean(final_y)
        # 按与均值的绝对距离排序
        sorted_pairs = sorted(zip(final_indices, final_y), key=lambda x: abs(x[1] - mean_final))
        best3_idx = [p[0] for p in sorted_pairs[:3]]
    else:
        best3_idx = final_indices

    # 按原始帧顺序排序输出
    best3_idx.sort(key=lambda i: all_fi[i])
    best3_fi = [all_fi[i] for i in best3_idx]
    best3_wy = [all_y[i] for i in best3_idx]
    best3_bp = [all_bp[i] for i in best3_idx]
    best3_masks = [all_masks[i] for i in best3_idx]
    best3_raw_masks = [all_raw_masks[i] for i in best3_idx]
    best3_bboxes = [all_bboxes[i] for i in best3_idx]
    best3_scores = [all_scores[i] for i in best3_idx]

    print(f"   ✅ 最终选择 3 帧: {best3_fi}  水位线Y: {[round(y,1) for y in best3_wy]}")
    return best3_fi, best3_wy, best3_bp, best3_masks, best3_raw_masks, best3_bboxes, best3_scores