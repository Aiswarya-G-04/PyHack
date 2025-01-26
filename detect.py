import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button

# Global variables for file paths
template_path = ""
video_path = ""

def upload_template():
    global template_path
    template_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if template_path:
        lbl_template_path.config(text=f"Template: {template_path}")

def upload_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if video_path:
        lbl_video_path.config(text=f"Video: {video_path}")

def detect_object():
    if not template_path or not video_path:
        lbl_status.config(text="Please upload both template and video!", fg="red")
        return
    
    # Load template image
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        lbl_status.config(text="Failed to load template image!", fg="red")
        return
    w, h = template.shape[::-1]

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        lbl_status.config(text="Failed to open video file!", fg="red")
        return

    detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # Define threshold for detection
        threshold = 0.8
        if max_val >= threshold:
            detected = True
            break  # Stop processing further as detection is confirmed

    cap.release()

    # Display results
    if detected:
        lbl_status.config(text="Detected", fg="green")
    else:
        lbl_status.config(text="Not Detected", fg="red")

# Create Tkinter GUI
root = tk.Tk()
root.title("Object Detection in Video")

# Upload template button
lbl_template_path = Label(root, text="Template: None")
lbl_template_path.pack()
btn_upload_template = Button(root, text="Upload Template Image", command=upload_template)
btn_upload_template.pack()

# Upload video button
lbl_video_path = Label(root, text="Video: None")
lbl_video_path.pack()
btn_upload_video = Button(root, text="Upload Video", command=upload_video)
btn_upload_video.pack()

# Detect button
btn_detect = Button(root, text="Start Detection", command=detect_object)
btn_detect.pack()

# Status label
lbl_status = Label(root, text="", fg="blue")
lbl_status.pack()

# Run Tkinter main loop 
root.mainloop()
