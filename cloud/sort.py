import os
import shutil
import pandas as pd

# ================= 路径配置 =================
EXCEL_PATH = r"C:\Users\a1595\Desktop\水位线高度统计.xlsx"
SOURCE_DIR = r"C:\Users\a1595\Desktop\extracted_images"
GOOD_DIR = r"C:\Users\a1595\Desktop\good"
BAD_DIR = r"C:\Users\a1595\Desktop\bad"

def classify_images():
    # 1. 检查输入文件和文件夹是否存在
    if not os.path.exists(EXCEL_PATH):
        print(f"❌ 找不到 Excel 文件: {EXCEL_PATH}")
        return
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 找不到源图片文件夹: {SOURCE_DIR}")
        return

    # 2. 自动创建输出文件夹（如果不存在）
    os.makedirs(GOOD_DIR, exist_ok=True)
    os.makedirs(BAD_DIR, exist_ok=True)

    # 3. 读取 Excel 数据
    try:
        # pandas 默认会将第一行作为表头
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 检查列数是否足够（G列意味着至少需要有7列数据）
    if df.shape[1] < 7:
        print(f"❌ Excel 文件的列数不足，当前只有 {df.shape[1]} 列，无法读取第G列！")
        return

    count_good = 0
    count_bad = 0
    count_miss = 0

    print("🚀 开始分类图片...\n")

    # 4. 遍历 Excel 中的每一行数据
    for index, row in df.iterrows():
        # iloc[0] 表示第一列 (A列)，iloc[6] 表示第七列 (G列)
        img_name = str(row.iloc[0]).strip()
        g_value_raw = row.iloc[6]

        # 尝试将 G 列的值转换为浮点数（跳过 N/A、空白或文字格式）
        try:
            g_value = float(g_value_raw)
        except (ValueError, TypeError):
            continue  # 如果该行数据无法转为数字，则跳过

        # 补全图片后缀 (如果 Excel 里的名字没有带 .png)
        if not (img_name.lower().endswith('.png') or img_name.lower().endswith('.jpg')):
            img_name += '.png'

        src_path = os.path.join(SOURCE_DIR, img_name)

        # 如果图片在源文件夹中不存在，给出提示并跳过
        if not os.path.exists(src_path):
            print(f"⚠️ 找不到图片: {img_name}，已跳过。")
            count_miss += 1
            continue

        # 5. 根据 G 列的数值大小进行分发复制
        if g_value < -5:
            dst_path = os.path.join(GOOD_DIR, img_name)
            shutil.copy2(src_path, dst_path)
            print(f"✅ [GOOD] {img_name} (G值: {g_value}) -> 复制到 good 文件夹")
            count_good += 1

        elif g_value > 5:
            dst_path = os.path.join(BAD_DIR, img_name)
            shutil.copy2(src_path, dst_path)
            print(f"❌ [BAD]  {img_name} (G值: {g_value}) -> 复制到 bad 文件夹")
            count_bad += 1

    # 打印最终统计结果
    print("\n" + "="*40)
    print("🎉 分类任务完成！")
    print(f"📂 输出到 Good 文件夹: {count_good} 张")
    print(f"📂 输出到 Bad  文件夹: {count_bad} 张")
    if count_miss > 0:
        print(f"⚠️ 找不到的源图片: {count_miss} 张")
    print("="*40)

if __name__ == "__main__":
    classify_images()