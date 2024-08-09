from ultralytics import YOLO

model = YOLO("yolov8m.yaml")
result = model.train(data='/home/kimi/project/yolo_coco/coco.yaml', epochs=100, imgsz=640)
