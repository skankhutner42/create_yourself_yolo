from ultralytics import YOLO

# 加载 COCO 预训练的 YOLOv8n 模型
model = YOLO("./runs/detect/train2/weights/best.pt")

# 在 'test.jpg' 图片上运行推理
results = model("test.jpg")
print(results)
# 遍历结果并保存每张带有预测结果的图片
for result in results:
    result.save()  # 保存到当前目录，或指定其他路径
