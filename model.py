from ultralytics import YOLO

# Load a YOLOv8 model (Nano here)
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' etc.

# Train the model
model.train(data='dataset/data.yaml', epochs=50, imgsz=640, batch=16)
results = model('img.jpg', conf=0.5)
results[0].save(filename='output.jpg')  # Save to file
results[0].plot()