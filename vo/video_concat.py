import cv2
import numpy as np
import sys

def concatenate_videos_vertically(video_path1, video_path2, output_path):
    """
    Reads two videos, concatenates them vertically frame by frame,
    and saves the result to a new video file.

    Args:
        video_path1 (str): Path to the first input video file.
        video_path2 (str): Path to the second input video file.
        output_path (str): Path to save the concatenated output video file.
    """
    # Open the two video files
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Check if video captures were opened successfully
    if not cap1.isOpened():
        print(f"Error: Could not open video 1: {video_path1}")
        return
    if not cap2.isOpened():
        print(f"Error: Could not open video 2: {video_path2}")
        cap1.release()
        return

    # Get properties from the first video (assuming both have same properties like fps, width)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Ensure widths are the same for vertical concatenation
    if width1 != width2:
        print(f"Error: Video widths must be the same for vertical concatenation ({width1} != {width2})")
        # Optional: Add resizing logic here if needed
        cap1.release()
        cap2.release()
        return

    # Define the codec and create VideoWriter object
    # Use 'mp4v' for .mp4 files. Adjust if using a different container.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_width = width1 # Width remains the same
    output_height = height1 + height2 # Height is the sum
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    if not writer.isOpened():
        print(f"Error: Could not open video writer for path: {output_path}")
        cap1.release()
        cap2.release()
        return

    print(f"Processing videos... Output size: {output_width}x{output_height} @ {fps:.2f} FPS")

    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # If either video ends, stop processing
        if not ret1 or not ret2:
            print("Reached end of one or both videos.")
            break

        # Ensure frame heights match the initial check (optional, but good practice)
        if frame1.shape[0] != height1 or frame2.shape[0] != height2 or frame1.shape[1] != width1 or frame2.shape[1] != width2:
             print("Warning: Frame dimensions mismatch during processing. Skipping frame.")
             # Or resize frames here if dynamic resizing is desired
             # Example: frame1 = cv2.resize(frame1, (width1, height1))
             #          frame2 = cv2.resize(frame2, (width2, height2))
             # If resizing, ensure width1 == width2 still holds.
             continue # Skip this frame pair if dimensions are wrong


        # Concatenate frames vertically
        combined_frame = cv2.vconcat([frame1, frame2])

        # Write the concatenated frame to the output video
        writer.write(combined_frame)

        # Optional: Display the resulting frame
        # cv2.imshow('Concatenated Video', combined_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release everything when done
    print("Finished processing. Releasing resources.")
    cap1.release()
    cap2.release()
    writer.release()
    # cv2.destroyAllWindows() # Only if cv2.imshow was used

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the actual paths to your video files
    input_video1 = "./0414_traj.mp4"
    input_video2 = "./output_video_1280_480.mp4"
    output_video = "./output_concatenated_video.mp4"

    # Basic command-line argument handling
    if len(sys.argv) == 4:
        input_video1 = sys.argv[1]
        input_video2 = sys.argv[2]
        output_video = sys.argv[3]
        print(f"Using command line arguments:")
        print(f"  Video 1: {input_video1}")
        print(f"  Video 2: {input_video2}")
        print(f"  Output:  {output_video}")
        concatenate_videos_vertically(input_video1, input_video2, output_video)
    elif len(sys.argv) == 1:
        print("No command line arguments provided. Using default paths in script.")
        print("You can run this script with arguments: python video_concat.py <video1_path> <video2_path> <output_path>")
        # Check if default paths exist before running
        # import os
        # if os.path.exists(input_video1) and os.path.exists(input_video2):
        #    concatenate_videos_vertically(input_video1, input_video2, output_video)
        # else:
        #    print(f"Error: Default video paths not found. Please edit the script or provide command line arguments.")
        print("Please edit the script with your video paths or provide them as command line arguments.")

    else:
        print("Usage: python video_concat.py <video1_path> <video2_path> <output_path>")
