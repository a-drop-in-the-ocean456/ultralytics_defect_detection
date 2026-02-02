from ultralytics import YOLO
import os
import cv2

# Load a model
model = YOLO("runs/detect/train4/weights/best.pt")  # load trained model

# Define input and output directories
input_dir = "./NG1000"  # directory containing validation images
output_dir = "predictions1000"  # directory to save predicted images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image files from input directory
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]

print(f"Found {len(image_files)} images to predict on")

# Process each image
for i, filename in enumerate(image_files):
    # Construct full image path
    img_path = os.path.join(input_dir, filename)
    
    # Predict on image
    results = model.predict(source=img_path, conf=0.2, iou=0.6)
    
    # Plot results with bounding boxes
    for result in results:
        annotated_img = result.plot()  # plot bounding boxes and labels
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"predicted_{filename}")
        cv2.imwrite(output_path, annotated_img)
    
    print(f"Processed {i+1}/{len(image_files)}: {filename}")

print("Prediction complete! Results saved in 'predictions' directory")