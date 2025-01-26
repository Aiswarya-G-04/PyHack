import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame

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
root.geometry("700x450")
root.config(bg="#e8f1f2")

# Create a main frame
main_frame = Frame(root, bg="#ffffff", padx=30, pady=30, relief="raised", borderwidth=3)
main_frame.pack(pady=30, padx=30, fill="both", expand=True)

# Title Label
lbl_title = Label(
    main_frame, 
    text="Object Detection Tool", 
    font=("Segoe UI", 24, "bold"), 
    bg="#ffffff", 
    fg="#4a4a4a"
)
lbl_title.pack(pady=10)

# Upload template button
lbl_template_path = Label(
    main_frame, 
    text="Template: None", 
    font=("Segoe UI", 12), 
    bg="#ffffff", 
    fg="#555555", 
    anchor="w"
)
lbl_template_path.pack(fill="x", pady=5)

btn_upload_template = Button(
    main_frame, 
    text="Upload Template Image", 
    font=("Segoe UI", 14), 
    command=upload_template, 
    bg="#007BFF", 
    fg="white", 
    relief="groove", 
    padx=10
)
btn_upload_template.pack(pady=5)

# Upload video button
lbl_video_path = Label(
    main_frame, 
    text="Video: None", 
    font=("Segoe UI", 12), 
    bg="#ffffff", 
    fg="#555555", 
    anchor="w"
)
lbl_video_path.pack(fill="x", pady=5)

btn_upload_video = Button(
    main_frame, 
    text="Upload Video", 
    font=("Segoe UI", 14), 
    command=upload_video, 
    bg="#28A745", 
    fg="white", 
    relief="groove", 
    padx=10
)
btn_upload_video.pack(pady=5)

# Detect button
btn_detect = Button(
    main_frame, 
    text="Start Detection", 
    font=("Segoe UI", 16, "bold"), 
    command=detect_object, 
    bg="#FFC107", 
    fg="black", 
    relief="raised", 
    padx=20
)
btn_detect.pack(pady=20)

# Status label
lbl_status = Label(
    main_frame, 
    text="", 
    font=("Segoe UI", 14), 
    bg="#ffffff", 
    fg="blue"
)
lbl_status.pack(pady=10)

# Run Tkinter main loop
root.mainloop()
