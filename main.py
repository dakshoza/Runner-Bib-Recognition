import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
import os
from tqdm import tqdm

def load_models():
    person_model = YOLO('./models/yolov8n.pt')
    bib_model = YOLO('./models/bib_detector_700.pt')
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    return person_model, bib_model, ocr

def detect_persons(image, model):
    results = model(image)
    person_boxes = []
    for r in results:
        for box in r.boxes:
            if box.cls == 0:  # 0 is the class index for person
                person_boxes.append(box.xyxy.cpu().numpy()[0])
    return np.array(person_boxes)

def detect_bibs(person_crop, model):
    results = model(person_crop)
    return results[0].boxes.xyxy.cpu().numpy()

def perform_ocr(bib_crop, ocr):
    bib_crop = cv2.resize(bib_crop, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    bib_crop_gray = cv2.cvtColor(bib_crop, cv2.COLOR_BGR2GRAY)
    _, bib_crop_binary = cv2.threshold(bib_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    ocr_result = ocr.ocr(bib_crop_binary, cls=False)
    return ocr_result[0] if ocr_result else []

def perform_ocr_extra(bib_crop, ocr):
    bib_crop = cv2.resize(bib_crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    bib_crop_gray = cv2.cvtColor(bib_crop, cv2.COLOR_BGR2GRAY)
    _, bib_crop_binary = cv2.threshold(bib_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to enhance the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bib_crop_binary = cv2.morphologyEx(bib_crop_binary, cv2.MORPH_CLOSE, kernel)
    
    # Apply adaptive thresholding to further enhance the text
    bib_crop_binary = cv2.adaptiveThreshold(bib_crop_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    denoised = cv2.fastNlMeansDenoising(bib_crop_binary, None, 10, 7, 21)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations = 1)
    eroded = cv2.erode(dilated, kernel, iterations = 1)

    ocr_result = ocr.ocr(eroded, cls=False)
    return ocr_result[0] if ocr_result else []

def draw_boxes_and_labels(image, boxes, labels):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Default color for person bounding boxes (green)
        if label.startswith("Bib") or label.startswith("NA"):
            color = (0, 0, 255)  # Red color for bib bounding boxes
            text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 255), -1)  # Yellow background
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)  # Black text
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou

def distance_to_center(box, person_center):
    bib_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return np.sqrt((bib_center[0] - person_center[0])**2 + (bib_center[1] - person_center[1])**2)

def process_image(image_path, person_model, bib_model, ocr, output_dir):
    image = cv2.imread(image_path)
    person_boxes = detect_persons(image, person_model)
    
    all_boxes = []
    all_labels = []
    processed_bibs = []

    for i, person_box in enumerate(person_boxes):
        x1, y1, x2, y2 = map(int, person_box)
        person_crop = image[y1:y2, x1:x2]
        person_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        bib_boxes = detect_bibs(person_crop, bib_model)
        
        best_bib = None
        best_bib_number = None
        best_score = float('inf')  # Lower is better
        
        for bib_box in bib_boxes:
            bib_box_global = bib_box.copy()
            bib_box_global[0] += x1
            bib_box_global[1] += y1
            bib_box_global[2] += x1
            bib_box_global[3] += y1
            
            if any(box_iou(bib_box_global, proc_bib) > 0.5 for proc_bib in processed_bibs):
                continue
            
            bx1, by1, bx2, by2 = map(int, bib_box_global)
            bib_crop = image[by1:by2, bx1:bx2]
            
            ocr_result = perform_ocr(bib_crop, ocr)
            bib_number = ''.join(re.findall(r'\d+', ocr_result[0][1][0])) if ocr_result else ""

            # If bib number is longer than 4 characters or no bib number detected, try perform_ocr_extra
            if len(bib_number) > 5 or not bib_number:
                ocr_result = perform_ocr_extra(bib_crop, ocr)
                bib_number = ''.join(re.findall(r'\d+', ocr_result[0][1][0])) if ocr_result else ""
            
            # Calculate score based on OCR result length and distance to person center
            ocr_score = 1 / (len(bib_number) + 1)  # +1 to avoid division by zero
            distance_score = distance_to_center(bib_box_global, person_center)
            score = ocr_score * distance_score
            
            if score < best_score:
                best_bib = bib_box_global
                best_bib_number = bib_number
                best_score = score
        
        all_boxes.append(person_box)
        all_labels.append("Person")
        
        if best_bib is not None:
            all_boxes.append(best_bib)
            processed_bibs.append(best_bib)
            if best_bib_number:
                all_labels.append(f"Bib: {best_bib_number}")
            else:
                all_labels.append("NA")
        elif len(bib_boxes) == 1:
            bib_box_global = bib_boxes[0].copy()
            bib_box_global[0] += x1
            bib_box_global[1] += y1
            bib_box_global[2] += x1
            bib_box_global[3] += y1
            if not any(box_iou(bib_box_global, proc_bib) > 0.5 for proc_bib in processed_bibs):
                all_boxes.append(bib_box_global)
                processed_bibs.append(bib_box_global)
                all_labels.append("NA")
    
    final_image = draw_boxes_and_labels(image, all_boxes, all_labels)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, final_image)
def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    person_model, bib_model, ocr = load_models()
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        process_image(image_path, person_model, bib_model, ocr, output_dir)

# Usage
input_directory = './test data'
output_directory = './test output 700'
process_directory(input_directory, output_directory)
# process_image('./test data/BIB_Test_Image11.jpeg', *load_models(), './')