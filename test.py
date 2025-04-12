from ultralytics import YOLO

# Load your custom trained model
model = YOLO('runs/detect/train4/weights/best.pt')  # Update path if needed

# Run inference on an image
results = model('img.jpg', conf=0.1)  # Replace with your image path

# Show result (opens the image in a window)
results[0].show()

# Save result to file
results[0].save(filename='output.jpg')
