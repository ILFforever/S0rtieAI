import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F

class HumanDetector:
    def __init__(self, confidence_threshold=0.7):
        # Initialize SSD model with new weights parameter
        self.ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        self.ssd_model.eval()
        self.confidence_threshold = confidence_threshold
        
        # Initialize face and body part detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        
        # Filter for human-related classes only
        self.human_classes = {
            1: 'person',
        }
        
        # Adjusted size thresholds for high-res images
        self.min_detection_size = 150  # Minimum width or height for any detection
        self.max_image_dimension = 1600  # Maximum dimension for processing
        
    def resize_image(self, image):
        """
        Resize image while maintaining aspect ratio if it exceeds max dimension
        """
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > self.max_image_dimension:
            scale = self.max_image_dimension / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height))
            return resized, scale
        return image, 1.0
        
    def preprocess_image(self, image):
        """
        Preprocess image to reduce noise and improve detection
        """
        # Resize image if too large
        processed, scale = self.resize_image(image)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(processed, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced, scale
    
    def scale_bbox_to_original(self, bbox, scale):
        """
        Scale detection bbox back to original image size
        """
        x, y, w, h = bbox
        return (int(x/scale), int(y/scale), int(w/scale), int(h/scale))
    
    def is_valid_detection(self, bbox, image_shape):
        """
        Check if detection is valid based on size and position
        """
        x, y, w, h = bbox
        image_height, image_width = image_shape[:2]
        
        # Size check (adjusted for high-res)
        if w < self.min_detection_size or h < self.min_detection_size:
            return False
            
        # Position check
        if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
            return False
            
        # Aspect ratio check
        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 3:
            return False
            
        return True
    
    def detect_roi(self, image, roi=None):
        """
        Detect faces and body parts within a region of interest
        """
        # Preprocess the image
        processed_image, scale = self.preprocess_image(image)
        
        if roi is not None:
            x, y, w, h = roi
            roi_image = processed_image[y:y+h, x:x+w]
        else:
            roi_image = processed_image
            x, y = 0, 0
        
        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        detected_objects = []
        
        # Detect with adjusted parameters for high-res
        faces = self.face_cascade.detectMultiScale(gray, 
                                                 scaleFactor=1.2,
                                                 minNeighbors=6,
                                                 minSize=(100, 100))
        
        for (fx, fy, fw, fh) in faces:
            bbox = (x + fx, y + fy, fw, fh)
            # Scale bbox back to original size
            orig_bbox = self.scale_bbox_to_original(bbox, scale)
            if self.is_valid_detection(orig_bbox, image.shape):
                detected_objects.append({
                    'bbox': orig_bbox,
                    'confidence': 0.9,
                    'class': 'face_frontal'
                })
        
        profile_faces = self.profile_face_cascade.detectMultiScale(gray,
                                                                 scaleFactor=1.2,
                                                                 minNeighbors=6,
                                                                 minSize=(100, 100))
        
        upper_bodies = self.upper_body_cascade.detectMultiScale(gray,
                                                              scaleFactor=1.2,
                                                              minNeighbors=5,
                                                              minSize=(150, 150))
        
        for (fx, fy, fw, fh) in profile_faces:
            bbox = (x + fx, y + fy, fw, fh)
            orig_bbox = self.scale_bbox_to_original(bbox, scale)
            if self.is_valid_detection(orig_bbox, image.shape):
                detected_objects.append({
                    'bbox': orig_bbox,
                    'confidence': 0.85,
                    'class': 'face_profile'
                })
                
        for (ux, uy, uw, uh) in upper_bodies:
            bbox = (x + ux, y + uy, uw, uh)
            orig_bbox = self.scale_bbox_to_original(bbox, scale)
            if self.is_valid_detection(orig_bbox, image.shape):
                detected_objects.append({
                    'bbox': orig_bbox,
                    'confidence': 0.8,
                    'class': 'upper_body'
                })
        
        return detected_objects
    
    def detect_ssd(self, image):
        """
        Detect full human bodies using SSD model
        """
        # Preprocess the image
        processed_image, scale = self.preprocess_image(image)
        
        # Prepare image for model
        image_tensor = F.to_tensor(processed_image)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            predictions = self.ssd_model(image_tensor)
        
        detected_objects = []
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score > self.confidence_threshold and label in self.human_classes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                bbox = (int(x1), int(y1), int(w), int(h))
                # Scale bbox back to original size
                orig_bbox = self.scale_bbox_to_original(bbox, scale)
                if self.is_valid_detection(orig_bbox, image.shape):
                    detected_objects.append({
                        'bbox': orig_bbox,
                        'confidence': float(score),
                        'class': self.human_classes[label]
                    })
        
        return detected_objects
    
    def draw_detections(self, image, detections):
        """
        Draw detection boxes on the image
        """
        image_copy = image.copy()
        color_map = {
            'face_frontal': (0, 255, 0),    # Green
            'face_profile': (0, 255, 255),   # Yellow
            'upper_body': (255, 0, 0),       # Blue
            'person': (0, 0, 255)            # Red
        }
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection.get('confidence', 0)
            label = detection.get('class', 'object')
            
            color = color_map.get(label, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, 3)  # Thicker lines for high-res
            
            # Draw label with larger font
            text = f"{label}: {confidence:.2f}"
            font_scale = image_copy.shape[1] / 2000  # Scale font based on image width
            thickness = max(1, int(image_copy.shape[1] / 1000))  # Scale thickness
            cv2.putText(image_copy, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return image_copy

def main():
    detector = HumanDetector(confidence_threshold=0.7)
    
    # Load image
    image = cv2.imread('img/back.jpg')
    if image is None:
        print("Error loading image")
        return
    
    # Detect using both methods
    roi_detections = detector.detect_roi(image)
    ssd_detections = detector.detect_ssd(image)
    
    # Combine detections
    all_detections = roi_detections + ssd_detections
    
    # Draw all detections
    result = detector.draw_detections(image, all_detections)
    
    # Display results (resize for display if needed)
    height, width = result.shape[:2]
    max_display_dim = 1200
    if max(height, width) > max_display_dim:
        scale = max_display_dim / max(height, width)
        display_size = (int(width * scale), int(height * scale))
        display_result = cv2.resize(result, display_size)
    else:
        display_result = result
    
    cv2.imshow('Human Detections', display_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()