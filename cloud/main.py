import sys
import os
import shutil
import time
from pathlib import Path

# ==========================================
# 环境变量与依赖配置
# ==========================================
#sys.path.append(r"D:\sam3model")
from sam3.model_builder import build_sam3_image_model

# 导入你的模块
from sam3first import load_ground_truth_waterline, process_all_images, save_results_to_excel_with_comparison
from state_tracker import create_tracker, update_waterline, display_tracker

# 阶段 2/3 的算法模块已移除：缺失时自动跳过对应阶段，避免启动即 ImportError
try:
    from secondlight import batch_process_binary_flood
except ImportError:
    batch_process_binary_flood = None

try:
    from secondwenli import batch_process_texture
except ImportError:
    batch_process_texture = None

try:
    from thirdlight import process_third_stage
except ImportError:
    process_third_stage = None

# 导入时序追踪逻辑库
from time_tracker import detect_temporal_anomalies


def setup_paths():
    paths = { 
        "checkpoint_path": "/root/sam/sam3.pt",
        "input_folder": "/root/sam/input",
        "output_folder": "/root/sam/output_first",
        "gt_file": "",
        "mask_output_folder": "/root/sam/output_mask",
        "flare_output_folder": "/root/sam/output_flare",
        "texture_output_folder": "/root/sam/output_texture",
        "third_output_folder": "/root/sam/output_third"
    }

    print("\n" + "=" * 60)
    print("【当前系统默认路径配置】")
    for key, val in paths.items():
        print(f" - {key}:\n   {val}")
    print("=" * 60 + "\n")

    modify_all = input("是否要手动更改上述路径配置？(y/n): ").lower().strip()
    if modify_all in ['y', 'yes']:
        print("\n>>> 进入手动路径配置模式 <<<")
        for key, default_path in paths.items():
            change = input(
                f"\n[?] 路径名称: {key}\n    当前设定值: {default_path}\n    是否需要修改？(y/n): ").lower().strip()
            if change in ['y', 'yes']:
                new_path = input("    请输入新路径: ").strip()
                if new_path: paths[key] = new_path
        print("\n✅ 路径配置完成！\n")
    else:
        print("✅ 使用系统默认配置继续执行...\n")

    return (paths["checkpoint_path"], paths["input_folder"], paths["output_folder"],
            paths["gt_file"], paths["mask_output_folder"], paths["flare_output_folder"],
            paths["texture_output_folder"], paths["third_output_folder"])


def clear_target_directories(directories):
    for d in directories:
        if os.path.exists(d):
            print(f"  🗑️ 正在清理: '{d}' ...", end="", flush=True)
            try:
                shutil.rmtree(d)
                time.sleep(0.1)
                os.makedirs(d, exist_ok=True)
                print(" ✅ 秒清完成")
            except Exception:
                for filename in os.listdir(d):
                    file_path = os.path.join(d, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    else:
                        shutil.rmtree(file_path)
                print(f"  ✅ 尽力清理完毕: '{d}'")
        else:
            print(f"  ⏩ 文件夹不存在: '{d}'")


def main():
    (checkpoint_path, input_folder, output_folder, gt_file,
     mask_output_folder, flare_output_folder, texture_output_folder, third_output_folder) = setup_paths()

    print("\n" + "=" * 60)
    do_clear = input("是否要清空以上所有输出文件夹中的历史结果与表格？(y/n): ").lower().strip()
    if do_clear in ['y', 'yes']:
        print("\n>>> 开始清理历史输出文件 <<<")
        clear_target_directories([
            output_folder, mask_output_folder, flare_output_folder,
            texture_output_folder, third_output_folder
        ])
        print(">>> 清理完成！<<<\n")
    else:
        print("\n✅ 保留历史文件，直接进行处理。\n")

    for p in [output_folder, mask_output_folder, flare_output_folder, texture_output_folder, third_output_folder]:
        Path(p).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    enable_temporal = input(
        "此次输入的图片是否为连续的一段时间序列？\n是否启用【时序曲线追踪】对被遮挡的异常值进行拟合替换？(y/n): ").lower().strip()
    use_temporal = enable_temporal in ['y', 'yes']

    if use_temporal:
        print("✅ 时序曲线追踪已开启，将直接使用全局拟合趋势代替离群的错误坐标。")
    else:
        print("⏩ 时序追踪已关闭。")
    print("=" * 60 + "\n")

    # ==========================================
    # 阶段 1: SAM3 分割与水位线计算
    # ==========================================
    print("正在加载 SAM3 模型 (请稍候)...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        bpe_path="/root/sam/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    )

    save_masks = True
    print("\n加载真实水位线数据...")
    gt_dicts = load_ground_truth_waterline(gt_file)

    image_files, success_results, failed = process_all_images(
        input_folder, model, "black water", gt_dicts, output_folder, mask_output_folder, save_masks
    )

    # ------------------------------------------
    # 核心：使用时序曲线预估值直接覆盖原数据
    # ------------------------------------------
    if use_temporal:
        anomalies_dict = detect_temporal_anomalies(success_results, poly_degree=3, threshold_pixels=15)

        if anomalies_dict:
            print("\n>>> 正在将表格与内存中的异常测值替换为物理拟合坐标 <<<")
            new_success_results = []

            for res in success_results:
                filename, pred_waterline, mask_top_y, confidence, used_threshold = res

                if filename in anomalies_dict:
                    estimated_y = anomalies_dict[filename]
                    print(f"  [坐标覆写] {filename} | 原错误值: {pred_waterline} -> 曲线真理值: {estimated_y}")
                    # 将测量水位和掩码顶端都替换为拟合值，保留原本的置信度和阈值记录
                    new_success_results.append((filename, estimated_y, estimated_y, confidence, used_threshold))
                else:
                    new_success_results.append(res)

            success_results = new_success_results

    # 导出替换后的纯净表格
    save_results_to_excel_with_comparison(success_results, failed, output_folder, gt_dicts)

    image_names = [Path(f).stem for f in image_files]
    global_tracker = create_tracker(image_names)

    # 毫无顾忌地将清洗后的数据灌入跟踪器
    for res in success_results:
        filename, pred_waterline, mask_top_y, confidence, used_threshold = res
        update_waterline(global_tracker, filename, mask_top_y)

    print("\n" + "=" * 50)
    print("SAM3 第一阶段 (水体分割与水位线) 处理完成！")
    print("=" * 50)

    # ==========================================
    # 阶段 2: 耀斑特征提取 + 水面纹理特征提取
    # ==========================================
    do_phase2 = input("\n是否进行第二阶段处理 (耀斑路径框选 & 纹理特征提取)？(y/n): ").lower().strip()

    if do_phase2 in ['y', 'yes']:
        if batch_process_binary_flood is None:
            print("\n⏭️ [2.1] 耀斑路径框选模块 (secondlight) 已移除，跳过。")
        else:
            print("\n>>> [2.1] 开始耀斑路径框选 (secondlight) <<<")
            batch_process_binary_flood(
                tracker_array=global_tracker,
                input_folder=input_folder,
                output_folder=flare_output_folder,
                tlbp_threshold=4,
                min_brightness=200
            )

        if batch_process_texture is None:
            print("\n⏭️ [2.2] 水面纹理特征提取模块 (secondwenli) 已移除，跳过。")
        else:
            print("\n>>> [2.2] 开始水面纹理特征提取 (secondwenli) <<<")
            batch_process_texture(
                tracker_array=global_tracker,
                input_folder=input_folder,
                output_folder=texture_output_folder,
                tlbp_threshold=4
            )
    else:
        print("\n⏭️ 用户选择跳过第二阶段。")

    # ==========================================
    # 阶段 3: 基于边界框的精准分割
    # ==========================================
    do_third = input("\n是否进行第三阶段处理 (基于边界框的SAM3精确分割)？(y/n): ").lower().strip()

    if do_third in ['y', 'yes']:
        if process_third_stage is None:
            print("\n⏭️ 第三阶段模块 (thirdlight) 已移除，跳过。")
        else:
            print("\n>>> 开始第三阶段处理 <<<")
            process_third_stage(
                tracker_array=global_tracker,
                model=model,
                input_dir=input_folder,
                box_dir=flare_output_folder,
                output_dir=third_output_folder,
                confidence_threshold=0.5
            )
    else:
        print("\n⏭️ 用户选择跳过第三阶段。")

    print("\n🚀 程序已全部运行完毕。")


if __name__ == "__main__":
    main()