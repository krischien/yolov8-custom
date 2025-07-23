from ultralytics import YOLO

model = YOLO('./best.pt')  # Load model

# results = model.predict('data/images/image.jpg', save=True, show=True, conf=0.7, save_txt=True)  # Predict images in 'data/images'
#results = model.predict('data/images/test.mp4', save=True, show=True, conf=0.7, save_txt=True)  # Predict video in 'data/images'
# results = model.predict('0', save=True, show=True, conf=0.7, save_txt=True)  # Predict camera'