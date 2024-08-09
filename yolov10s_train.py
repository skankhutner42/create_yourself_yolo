from ultralytics import YOLO


model = YOLO("yolov10s.yaml")
model.info()
# result = model.train(data='/home/kimi/project/yolo_coco/coco.yaml', epochs=250, imgsz=640, resume=True)
result = model.train(data='coco.yaml', epochs=250, imgsz=640, resume=True)
