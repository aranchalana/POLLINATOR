#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv5 detection script that processes video frames and saves them
to separate folders based on whether objects are detected.
"""

# Usage:
# python detect_and_save_script.py --video ./videos/rotate_video2024-02-19_15-30-44.mp4 --model ./runs/train/merged_ALL/weights/best.pt --conf-thres 0.8
# python detect_and_save_script.py --video 0 --model ./runs/train/merged_ALL/weights/best.pt --conf-thres 0.8
# python detect_and_save_script.py --video ./videos/rotate_video2024-02-19_15-30-44.mp4 --model ./runs/train/merged_ALL/weights/best.pt --conf-thres 0.8 --frame-interval 0.5 --output-dir ./output

import cv2
import torch
import os
import argparse
from datetime import datetime
import sys

def parse_args():

    parser = argparse.ArgumentParser(description='YOLOv5 detection and frame saving')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file or camera index (e.g., 0)')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv5 model weights')
    parser.add_argument('--output-dir', type=str, default='output', help='Base output directory')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--frame-interval', type=float, default=1.0, 
                        help='Interval in seconds between frames to process')
    parser.add_argument('--no-display', action='store_true', help='Disable frame display')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle the video source - either a file or camera index
    try:
        video_source = args.video
        if video_source.isdigit():
            video_source = int(video_source)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {args.video}")
            return
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    # Load YOLOv5 model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model)
        print(f"Model loaded from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(args.video) if isinstance(args.video, str) else f"camera_{args.video}"
    
    # Calculate frame interval in frames
    frame_interval = int(args.frame_interval * fps)
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"Video: {video_name}")
    print(f"FPS: {fps}")
    print(f"Processing every {frame_interval} frames ({args.frame_interval} seconds)")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(args.output_dir, f"{video_name.split('.')[0]}_{timestamp}")
    output_dir_with_object = os.path.join(base_output_dir, "with_object")
    output_dir_without_object = os.path.join(base_output_dir, "without_object")
    
    os.makedirs(output_dir_with_object, exist_ok=True)
    os.makedirs(output_dir_without_object, exist_ok=True)
    
    print(f"Saving detected objects to: {output_dir_with_object}")
    print(f"Saving frames without objects to: {output_dir_without_object}")
    
    # Process frames
    frame_count = 0
    processed_count = 0
    detected_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Show progress for video files (not for camera streams)
        if total_frames > 0 and frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Only analyze every nth frame
        if frame_count % frame_interval != 0:
            continue
        
        processed_count += 1
        
        # Calculate the current timestamp in the video
        current_time = frame_count / fps
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        time_str = f"{hours:02d}-{minutes:02d}-{seconds:02d}"
        
        # Convert frame to RGB for YOLOv5
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = model(img_rgb)
        
        # Check if any objects are detected above the confidence threshold
        detected = False
        for *box, conf, cls in results.xyxy[0]:  # xyxy format
            if conf > args.conf_thres:
                detected = True
                detected_count += 1
                x1, y1, x2, y2 = map(int, box)
                class_name = results.names[int(cls)]
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the frame with or without object detection
        if detected:
            image_path = os.path.join(output_dir_with_object, 
                                    f"frame_{frame_count:06d}_time_{time_str}.jpg")
        else:
            image_path = os.path.join(output_dir_without_object, 
                                    f"frame_{frame_count:06d}_time_{time_str}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Display the frame (optional)
        if not args.no_display:
            # Add text showing current time in video
            cv2.putText(frame, f"Time: {time_str}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Frame', frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Report statistics
    print(f"Processed {processed_count} frames out of {frame_count} total frames")
    print(f"Detected objects in {detected_count} frames")
    print(f"Images with objects saved to: {output_dir_with_object}")
    print(f"Images without objects saved to: {output_dir_without_object}")
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
