#!/usr/bin/env python3
"""
YOLOv8 Desktop Application
A desktop GUI for YOLOv8 object detection with camera, video, and image support
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
import threading
import time
import os
import sys
from PIL import Image, ImageTk
import numpy as np
import torch

# Add the parent directory to path to import the API
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code', 'API'))
from api import get_model, live_detection_counts

class YOLODesktopApp:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("YOLOv8 Object Detection Desktop App")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Initialize variables
        self.camera_active = False
        self.camera_thread = None
        self.cap = None
        self.current_detections = []
        self.person_count = 0
        self.car_count = 0
        
        # Load models
        self.person_model = None
        self.car_model = None
        self.load_models()
        
        # Create GUI
        self.create_widgets()
        
        # Start update loop
        self.update_display()
    
    def load_models(self):
        """Load YOLO models"""
        try:
            # Get the path to the API directory
            api_dir = os.path.join(os.path.dirname(__file__), '..', 'code', 'API')
            
            # Load person detection model
            person_model_path = os.path.join(api_dir, 'yolo11m.pt')
            if os.path.exists(person_model_path):
                self.person_model = get_model(person_model_path)
                print(f"✓ Person model loaded: {person_model_path}")
            else:
                print(f"✗ Person model not found: {person_model_path}")
            
            # Load car detection model
            car_model_path = os.path.join(api_dir, 'models', 'best.pt')
            if os.path.exists(car_model_path):
                self.car_model = get_model(car_model_path)
                print(f"✓ Car model loaded: {car_model_path}")
            else:
                print(f"✗ Car model not found: {car_model_path}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            messagebox.showerror("Model Loading Error", f"Failed to load models: {e}")
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Create main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="YOLOv8 Object Detection", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Create content frame
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ctk.CTkFrame(content_frame, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - Video display
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # === LEFT PANEL - CONTROLS ===
        
        # Detection Type
        detection_frame = ctk.CTkFrame(left_panel)
        detection_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(detection_frame, text="Detection Type", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.detection_type = ctk.StringVar(value="person")
        person_radio = ctk.CTkRadioButton(detection_frame, text="Person Detection", variable=self.detection_type, value="person")
        person_radio.pack(pady=2)
        car_radio = ctk.CTkRadioButton(detection_frame, text="Car Detection", variable=self.detection_type, value="car")
        car_radio.pack(pady=2)
        
        # Input Source
        source_frame = ctk.CTkFrame(left_panel)
        source_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(source_frame, text="Input Source", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.source_type = ctk.StringVar(value="camera")
        camera_radio = ctk.CTkRadioButton(source_frame, text="Camera", variable=self.source_type, value="camera", command=self.on_source_change)
        camera_radio.pack(pady=2)
        ip_camera_radio = ctk.CTkRadioButton(source_frame, text="IP Camera", variable=self.source_type, value="ip_camera", command=self.on_source_change)
        ip_camera_radio.pack(pady=2)
        video_radio = ctk.CTkRadioButton(source_frame, text="Video File", variable=self.source_type, value="video", command=self.on_source_change)
        video_radio.pack(pady=2)
        image_radio = ctk.CTkRadioButton(source_frame, text="Image File", variable=self.source_type, value="image", command=self.on_source_change)
        image_radio.pack(pady=2)
        
        # Camera Settings
        self.camera_frame = ctk.CTkFrame(left_panel)
        self.camera_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(self.camera_frame, text="Camera Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        ctk.CTkLabel(self.camera_frame, text="Camera Index:").pack()
        self.camera_index = ctk.CTkEntry(self.camera_frame, placeholder_text="0")
        self.camera_index.pack(pady=5)
        self.camera_index.insert(0, "0")
        
        # IP Camera Settings (initially hidden)
        self.ip_camera_frame = ctk.CTkFrame(left_panel)
        
        ctk.CTkLabel(self.ip_camera_frame, text="IP Camera Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        ctk.CTkLabel(self.ip_camera_frame, text="Stream URL:").pack()
        self.ip_camera_url_var = ctk.StringVar()
        self.ip_camera_url_entry = ctk.CTkEntry(self.ip_camera_frame, textvariable=self.ip_camera_url_var, placeholder_text="rtsp://192.168.1.100:554/stream")
        self.ip_camera_url_entry.pack(pady=5, padx=10, fill="x")
        
        # Add example URLs
        examples_frame = ctk.CTkFrame(self.ip_camera_frame)
        examples_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(examples_frame, text="Examples:", font=ctk.CTkFont(size=12, weight="bold")).pack()
        
        example_urls = [
            "rtsp://192.168.1.100:554/stream",
            "http://192.168.1.100:8080/video",
            "rtmp://192.168.1.100:1935/live/stream"
        ]
        
        for url in example_urls:
            example_btn = ctk.CTkButton(examples_frame, text=url, command=lambda u=url: self.ip_camera_url_var.set(u), height=25)
            example_btn.pack(pady=1, fill="x")
        
        # Test connection button
        self.test_ip_camera_button = ctk.CTkButton(self.ip_camera_frame, text="Test Connection", command=self.test_ip_camera)
        self.test_ip_camera_button.pack(pady=5)
        
        # File Settings (initially hidden)
        self.file_frame = ctk.CTkFrame(left_panel)
        
        ctk.CTkLabel(self.file_frame, text="File Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.file_path_var = ctk.StringVar()
        self.file_path_entry = ctk.CTkEntry(self.file_frame, textvariable=self.file_path_var, placeholder_text="Select file...")
        self.file_path_entry.pack(pady=5, padx=10, fill="x")
        
        self.browse_button = ctk.CTkButton(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)
        
        # Detection Settings
        settings_frame = ctk.CTkFrame(left_panel)
        settings_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(settings_frame, text="Detection Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        ctk.CTkLabel(settings_frame, text="Confidence Threshold:").pack()
        self.confidence_var = ctk.DoubleVar(value=0.6)
        self.confidence_slider = ctk.CTkSlider(settings_frame, from_=0.1, to=1.0, variable=self.confidence_var, number_of_steps=90)
        self.confidence_slider.pack(pady=5, padx=10, fill="x")
        
        self.confidence_label = ctk.CTkLabel(settings_frame, text="0.6")
        self.confidence_label.pack()
        self.confidence_slider.configure(command=self.update_confidence_label)
        
        # Control Buttons
        control_frame = ctk.CTkFrame(left_panel)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_button = ctk.CTkButton(control_frame, text="Start Detection", command=self.start_detection, fg_color="green")
        self.start_button.pack(pady=5, fill="x")
        
        self.stop_button = ctk.CTkButton(control_frame, text="Stop Detection", command=self.stop_detection, fg_color="red", state="disabled")
        self.stop_button.pack(pady=5, fill="x")
        
        # Results
        results_frame = ctk.CTkFrame(left_panel)
        results_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(results_frame, text="Detection Results", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.person_count_label = ctk.CTkLabel(results_frame, text="Persons: 0")
        self.person_count_label.pack(pady=2)
        
        self.car_count_label = ctk.CTkLabel(results_frame, text="Cars: 0")
        self.car_count_label.pack(pady=2)
        
        # Status
        self.status_label = ctk.CTkLabel(left_panel, text="Ready", font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=10)
        
        # === RIGHT PANEL - VIDEO DISPLAY ===
        
        # Video display
        self.video_label = ctk.CTkLabel(right_panel, text="No video source", font=ctk.CTkFont(size=18))
        self.video_label.pack(expand=True)
        
        # Initialize source change
        self.on_source_change()
    
    def on_source_change(self):
        """Handle source type change"""
        source = self.source_type.get()
        
        # Hide all frames first
        self.camera_frame.pack_forget()
        self.ip_camera_frame.pack_forget()
        self.file_frame.pack_forget()
        
        # Show appropriate frame
        if source == "camera":
            self.camera_frame.pack(fill="x", padx=10, pady=10)
        elif source == "ip_camera":
            self.ip_camera_frame.pack(fill="x", padx=10, pady=10)
        else:  # video or image
            self.file_frame.pack(fill="x", padx=10, pady=10)
    
    def update_confidence_label(self, value):
        """Update confidence label when slider changes"""
        self.confidence_label.configure(text=f"{value:.2f}")
    
    def browse_file(self):
        """Browse for video or image file"""
        source = self.source_type.get()
        
        if source == "video":
            filetypes = [
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                ("All files", "*.*")
            ]
        else:  # image
            filetypes = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        
        filename = filedialog.askopenfilename(
            title=f"Select {source.title()} File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
    
    def start_detection(self):
        """Start detection based on selected source and type"""
        source = self.source_type.get()
        detection_type = self.detection_type.get()
        
        if source == "camera":
            self.start_camera_detection()
        elif source == "ip_camera":
            self.start_ip_camera_detection()
        elif source == "video":
            self.start_video_detection()
        elif source == "image":
            self.start_image_detection()
    
    def start_camera_detection(self):
        """Start live camera detection"""
        try:
            camera_index = int(self.camera_index.get())
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", f"Cannot open camera {camera_index}")
                return
            
            # Test frame read
            ret, test_frame = self.cap.read()
            if not ret:
                messagebox.showerror("Camera Error", f"Camera {camera_index} cannot read frames")
                self.cap.release()
                return
            
            self.camera_active = True
            self.camera_thread = threading.Thread(target=self.camera_detection_loop, daemon=True)
            self.camera_thread.start()
            
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_label.configure(text="Camera detection active")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera detection: {e}")
    
    def validate_ip_camera_url(self, url):
        """Validate IP camera URL format"""
        supported_protocols = ['rtsp://', 'http://', 'https://', 'rtmp://', 'udp://']
        return any(url.lower().startswith(protocol) for protocol in supported_protocols)
    
    def test_ip_camera(self):
        """Test IP camera connection"""
        url = self.ip_camera_url_var.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter an IP camera URL")
            return
        
        if not self.validate_ip_camera_url(url):
            messagebox.showerror("Error", f"Unsupported protocol. Supported protocols: {', '.join(['rtsp://', 'http://', 'https://', 'rtmp://', 'udp://'])}")
            return
        
        try:
            self.status_label.configure(text="Testing IP camera connection...")
            
            # Test connection
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                messagebox.showerror("Connection Error", f"Cannot connect to IP camera: {url}")
                return
            
            # Try to read a test frame
            ret, test_frame = cap.read()
            if not ret:
                messagebox.showerror("Connection Error", f"IP camera connected but cannot read frames: {url}")
                cap.release()
                return
            
            cap.release()
            messagebox.showinfo("Success", f"IP camera connection successful!\nURL: {url}\nFrame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
            self.status_label.configure(text="IP camera connection test successful")
            
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to test IP camera: {e}")
            self.status_label.configure(text="IP camera connection test failed")
    
    def start_ip_camera_detection(self):
        """Start IP camera detection"""
        url = self.ip_camera_url_var.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter an IP camera URL")
            return
        
        if not self.validate_ip_camera_url(url):
            messagebox.showerror("Error", f"Unsupported protocol. Supported protocols: {', '.join(['rtsp://', 'http://', 'https://', 'rtmp://', 'udp://'])}")
            return
        
        try:
            self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                messagebox.showerror("IP Camera Error", f"Cannot connect to IP camera: {url}")
                return
            
            # Test frame read
            ret, test_frame = self.cap.read()
            if not ret:
                messagebox.showerror("IP Camera Error", f"IP camera connected but cannot read frames: {url}")
                self.cap.release()
                return
            
            self.camera_active = True
            self.camera_thread = threading.Thread(target=self.camera_detection_loop, daemon=True)
            self.camera_thread.start()
            
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_label.configure(text=f"IP camera detection active: {url}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start IP camera detection: {e}")
    
    def camera_detection_loop(self):
        """Main camera detection loop"""
        detection_type = self.detection_type.get()
        confidence = self.confidence_var.get()
        
        # Select model based on detection type
        model = self.person_model if detection_type == "person" else self.car_model
        
        if model is None:
            self.status_label.configure(text="Model not loaded")
            return
        
        # Track detections across frames for persistence
        tracked_detections = []  # List of [x1, y1, x2, y2, conf, age]
        frame_count = 0
        
        while self.camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 2nd frame for better performance (reduced from 3rd)
            if frame_count % 2 == 0:
                try:
                    # Resize frame for faster detection (optional optimization)
                    detection_frame = cv2.resize(frame, (640, 480))
                    
                    # Run detection on smaller frame for speed
                    results = model.predict(detection_frame, conf=confidence, verbose=False, device='0' if torch.cuda.is_available() else 'cpu')
                    
                    # Process detections
                    current_detections = []
                    # Get original frame dimensions for coordinate scaling
                    orig_h, orig_w = frame.shape[:2]
                    det_h, det_w = detection_frame.shape[:2]
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                
                                # Scale coordinates back to original frame size
                                x1 = x1 * orig_w / det_w
                                y1 = y1 * orig_h / det_h
                                x2 = x2 * orig_w / det_w
                                y2 = y2 * orig_h / det_h
                                
                                current_detections.append([x1, y1, x2, y2, conf, 0])  # Add age=0
                    
                    # Update tracked detections (simple persistence)
                    tracked_detections = current_detections.copy()
                    
                    # Update counts
                    if detection_type == "person":
                        self.person_count = len(current_detections)
                    else:
                        self.car_count = len(current_detections)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
            
            # Draw all tracked detections on frame (every frame)
            for detection in tracked_detections:
                x1, y1, x2, y2, conf, age = detection
                
                # Draw bounding box with persistence effect
                color = (0, 255, 0) if detection_type == "person" else (255, 0, 0)
                thickness = 2
                
                # Make boxes more visible with thicker lines for persistent detections
                if age > 0:
                    thickness = 3
                    # Add slight glow effect for persistent detections
                    cv2.rectangle(frame, (int(x1-1), int(y1-1)), (int(x2+1), int(y2+1)), (0, 200, 0) if detection_type == "person" else (200, 0, 0), thickness+1)
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
                # Add label with confidence
                label = f"{detection_type.title()}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Age tracked detections (for persistence across frames)
            for detection in tracked_detections:
                detection[5] += 1  # Increment age
            
            # Remove old detections (keep for 10 frames = ~0.3 seconds at 30fps)
            tracked_detections = [d for d in tracked_detections if d[5] < 10]
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_resized)
            self.current_frame = ImageTk.PhotoImage(pil_image)
            
            # Reduced sleep for smoother video
            time.sleep(0.01)  # ~100 FPS target
    
    def start_video_detection(self):
        """Start video file detection"""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("File Error", "Please select a valid video file")
            return
        
        # Run video detection in a separate thread
        threading.Thread(target=self.video_detection_worker, daemon=True).start()
    
    def video_detection_worker(self):
        """Worker thread for video detection"""
        try:
            file_path = self.file_path_var.get()
            detection_type = self.detection_type.get()
            confidence = self.confidence_var.get()
            
            model = self.person_model if detection_type == "person" else self.car_model
            if model is None:
                self.status_label.configure(text="Model not loaded")
                return
            
            self.status_label.configure(text="Processing video...")
            
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                messagebox.showerror("Video Error", "Cannot open video file")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            total_detections = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 10th frame for performance
                if frame_count % 10 == 0:
                    results = model.predict(frame, conf=confidence, verbose=False)
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            total_detections += len(boxes)
                
                # Update progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.status_label.configure(text=f"Processing video... {progress:.1f}%")
            
            cap.release()
            
            # Update results
            if detection_type == "person":
                self.person_count = total_detections
            else:
                self.car_count = total_detections
            
            self.status_label.configure(text="Video processing complete")
            messagebox.showinfo("Complete", f"Video processing complete. Found {total_detections} {detection_type}s.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Video processing failed: {e}")
            self.status_label.configure(text="Video processing failed")
    
    def start_image_detection(self):
        """Start image file detection"""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("File Error", "Please select a valid image file")
            return
        
        try:
            detection_type = self.detection_type.get()
            confidence = self.confidence_var.get()
            
            model = self.person_model if detection_type == "person" else self.car_model
            if model is None:
                messagebox.showerror("Model Error", "Model not loaded")
                return
            
            self.status_label.configure(text="Processing image...")
            
            # Load and process image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Image Error", "Cannot load image file")
                return
            
            # Run detection
            results = model.predict(image, conf=confidence, verbose=False)
            
            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf])
                        
                        # Draw bounding box
                        color = (0, 255, 0) if detection_type == "person" else (255, 0, 0)
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        label = f"{detection_type.title()}: {conf:.2f}"
                        cv2.putText(image, label, (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update counts
            if detection_type == "person":
                self.person_count = len(detections)
            else:
                self.car_count = len(detections)
            
            # Display result
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (640, 480))
            
            pil_image = Image.fromarray(image_resized)
            self.current_frame = ImageTk.PhotoImage(pil_image)
            
            self.status_label.configure(text=f"Image processed. Found {len(detections)} {detection_type}s.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Image processing failed: {e}")
            self.status_label.configure(text="Image processing failed")
    
    def stop_detection(self):
        """Stop detection"""
        self.camera_active = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="Detection stopped")
    
    def update_display(self):
        """Update the display"""
        # Update video display
        if hasattr(self, 'current_frame'):
            self.video_label.configure(image=self.current_frame, text="")
        else:
            self.video_label.configure(image=None, text="No video source")
        
        # Update count labels
        self.person_count_label.configure(text=f"Persons: {self.person_count}")
        self.car_count_label.configure(text=f"Cars: {self.car_count}")
        
        # Schedule next update - increased frequency for smoother video
        self.root.after(16, self.update_display)  # ~60 FPS
    
    def run(self):
        """Run the application"""
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup"""
        self.stop_detection()

if __name__ == "__main__":
    app = YOLODesktopApp()
    app.run() 