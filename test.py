from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Khởi tạo model
model = YOLO(model='yolov8_1000.pt')

# Initialize TrOCR processor and model
model_dir = "./content/sample_data/vit-ocr"  # Đường dẫn đến thư mục chứa mô hình
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model_ocr = VisionEncoderDecoderModel.from_pretrained(model_dir)

# Hình ảnh
# img = cv2.imread("./15415dba-1bfe-4a5e-a79b-3cfc0f5c1d5f_jpg.rf.1d7c45e26c7a2e361e2a17136598faff.jpg")
# img = Image.open(
#     "dongho/15415dba-1bfe-4a5e-a79b-3cfc0f5c1d5f_jpg.rf.1d7c45e26c7a2e361e2a17136598faff.jpg").convert("RGB")

# # Kết quả nhận dạng
# results = model.predict(source=img, conf=0.2, iou=0.5)
# result = results[0]

# # Tính tọa độ của bounding box trong tệp label
# for box in result.boxes:
#     class_id = result.names[box.cls[0].item()]
#     print("Object type:", class_id)
#     cords = box.xyxy[0].tolist()
#     cords = [round(x) for x in cords]
#     print("Coordinates:", cords)
#     print("---")

#     # Cắt ảnh con từ ảnh gốc
#     cropped_image = img.crop((cords[0], cords[1], cords[2], cords[3]))

#     # Lưu ảnh con
#     # cropped_image.save(f"class_{class_id}_cropped.jpg")

#     # Convert
#     pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     pixel_values = pixel_values.to(device)

#     # # Inference
#     generated_ids = model_ocr.generate(pixel_values)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(generated_text)
