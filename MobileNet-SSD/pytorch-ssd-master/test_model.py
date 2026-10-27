import argparse
import os
import sys
import time

import cv2
import torch

# 保证无论从哪个目录运行，都能 import 到 vision 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 脚本所在目录，用于将默认的相对路径解析为绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 全局参数：在 VSCode 中直接修改下面的值即可调试运行（无需命令行传参）
# 也可通过命令行参数覆盖这些默认值，例如：
#   python test_model.py --test_image_file D:/imgs --result_file D:/out
# =====================================================================

# 测试图片所在文件夹路径
test_image_file = r"E:\VS code\MobileNet-SSD\pytorch-ssd-master\test_images"

# 结果保存的文件夹路径
result_file = os.path.join(SCRIPT_DIR, "results")

# 网络类型
net_type = "mb1-ssd"

# 训练好的模型权重路径
model_path = os.path.join(SCRIPT_DIR, "models", "mb1-ssd-Epoch-199-Loss-0.6272794902324677.pth")

# 标签文件路径
label_file = os.path.join(SCRIPT_DIR, "models", "voc-model-labels.txt")

# 置信度阈值
threshold = 0.4

# 每张图最多检测的目标数
top_k = 3
# =====================================================================


def build_net(net_type, num_classes):
    if net_type == 'mb1-ssd':
        return create_mobilenetv1_ssd(num_classes, is_test=True), create_mobilenetv1_ssd_predictor
    if net_type == 'mb1-ssd-lite':
        return create_mobilenetv1_ssd_lite(num_classes, is_test=True), create_mobilenetv1_ssd_lite_predictor
    if net_type == 'mb2-ssd-lite':
        return create_mobilenetv2_ssd_lite(num_classes, is_test=True), create_mobilenetv2_ssd_lite_predictor
    if net_type == 'mb3-large-ssd-lite':
        return create_mobilenetv3_large_ssd_lite(num_classes, is_test=True), create_mobilenetv2_ssd_lite_predictor
    if net_type == 'mb3-small-ssd-lite':
        return create_mobilenetv3_small_ssd_lite(num_classes, is_test=True), create_mobilenetv2_ssd_lite_predictor
    if net_type == 'vgg16-ssd':
        return create_vgg_ssd(num_classes, is_test=True), create_vgg_ssd_predictor
    if net_type == 'sq-ssd-lite':
        return create_squeezenet_ssd_lite(num_classes, is_test=True), create_squeezenet_ssd_lite_predictor
    raise ValueError(f"Unsupported net type: {net_type}")


def list_images(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            files.append(os.path.join(folder, name))
    return files


def main():
    parser = argparse.ArgumentParser(description='Run and test the trained SSD detection model.')
    parser.add_argument('--test_image_file', default=test_image_file,
                        help='Folder containing the test images.')
    parser.add_argument('--result_file', default=result_file,
                        help='Folder where the annotated images and the result record will be saved.')
    parser.add_argument('--net', default=net_type, help='Network type.')
    parser.add_argument('--model', default=model_path, help='Path to the trained model weights.')
    parser.add_argument('--label_file', default=label_file, help='Path to the label file.')
    parser.add_argument('--threshold', type=float, default=threshold,
                        help='Confidence threshold for detections.')
    parser.add_argument('--top_k', type=int, default=top_k,
                        help='Maximum detections per image.')
    args = parser.parse_args()

    if not os.path.isdir(args.test_image_file):
        print(f"[ERROR] test image folder does not exist: {args.test_image_file}")
        sys.exit(1)

    os.makedirs(args.result_file, exist_ok=True)

    class_names = [name.strip() for name in open(args.label_file, encoding='utf-8').readlines()]
    num_classes = len(class_names)
    print(f"Labels: {class_names}")

    net, predictor_fn = build_net(args.net, num_classes)
    net.load(args.model)
    predictor = predictor_fn(net, candidate_size=200)

    image_paths = list_images(args.test_image_file)
    if not image_paths:
        print(f"[ERROR] no images found in: {args.test_image_file}")
        sys.exit(1)
    print(f"Found {len(image_paths)} images to test.\n")

    result_txt = os.path.join(args.result_file, 'results.txt')
    total_objects = 0
    detected_images = 0
    start_all = time.time()

    with open(result_txt, 'w', encoding='utf-8') as f:
        f.write(f"# model: {args.model}\n")
        f.write(f"# threshold: {args.threshold}\n")
        f.write(f"# test folder: {args.test_image_file}\n\n")

        for image_path in image_paths:
            name = os.path.basename(image_path)
            orig = cv2.imread(image_path)
            if orig is None:
                print(f"[WARN] cannot read image, skip: {name}")
                f.write(f"{name}: cannot read\n")
                continue

            image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            boxes, labels, probs = predictor.predict(image, args.top_k, args.threshold)

            line = f"{name}: {len(probs)} object(s)"
            f.write(line + "\n")
            print(line)

            for i in range(len(probs)):
                x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
                cls = class_names[labels[i]]
                conf = float(probs[i])
                total_objects += 1
                detail = f"    {cls}: {conf:.3f}  box=({x1},{y1},{x2},{y2})"
                print(detail)
                f.write(detail + "\n")

                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(orig, f"{cls}: {conf:.2f}", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if len(probs) > 0:
                detected_images += 1

            out_path = os.path.join(args.result_file, os.path.splitext(name)[0] + '_result.jpg')
            cv2.imwrite(out_path, orig)

    elapsed = time.time() - start_all
    summary = (
        f"\nDone. {len(image_paths)} images tested in {elapsed:.2f}s, "
        f"{detected_images} images with detections, {total_objects} objects in total.\n"
        f"Annotated images and the record are saved in: {args.result_file}"
    )
    print(summary)
    with open(result_txt, 'a', encoding='utf-8') as f:
        f.write("\n" + summary + "\n")


if __name__ == '__main__':
    main()
