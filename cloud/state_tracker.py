# state_tracker.py

def create_tracker(image_names):
    """
    初始化状态数组
    格式: [图片序号, 掩码最高位置(水位线), 是否有绿色框]
    默认值: [name, 0, 0]
    """
    return [[name, 0, 0] for name in image_names]

def update_waterline(tracker, img_name, waterline_val):
    """将 SAM3 计算出的掩码最高位置更新到数组的第一项"""
    for row in tracker:
        if str(row[0]) == str(img_name):
            row[1] = waterline_val if waterline_val is not None else 0
            break

def display_tracker(tracker, top_n=10):
    """展示前 N 张图片的状态数组"""
    print(f"\n{'='*55}")
    print(f"前 {top_n} 张图片状态数组结果展示:")
    print(f"{'图片序号':<15} | {'掩码最高位置(项1)':<15} | {'是否有绿框(项2)'}")
    print("-" * 55)
    for row in tracker[:top_n]:
        print(f"{str(row[0]):<15} | {str(row[1]):<19} | {row[2]}")
    print(f"{'='*55}\n")