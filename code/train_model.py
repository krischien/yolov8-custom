from ultralytics import YOLO

model = YOLO('./best.py')  # Load model

results = model.train(data='data_custom.yaml', epochs=100, imgsz=640, workers=1, batch=3)  # Train for 100 epochs