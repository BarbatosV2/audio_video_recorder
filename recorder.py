import cv2
import pyaudio
import wave
import threading
import os
import time
import re
import sys
import numpy as np # Added for audio visualization
import argparse # Added for command-line argument parsing

# --- Configuration ---
RECORD_FOLDER = "record"  # Name of the subfolder for recordings

# IP Camera settings
IP_CAM_URL = "http://172.20.10.2:81/stream"  # Your IP camera stream URL
                                             # Set to None or "" to disable IP camera recording

# USB Webcam settings
USB_CAM_INDEX = 0  # Index of the USB webcam (usually 0 for the default)
                   # Set to None to disable USB camera recording

# Audio settings
AUDIO_ENABLED = True  # Set to False to disable audio recording
AUDIO_RATE = 44100    # Sample rate for audio
AUDIO_CHUNK_SIZE = 1024 # Number of frames per buffer
AUDIO_CHANNELS = 1      # 1 for mono, 2 for stereo
AUDIO_FORMAT = pyaudio.paInt16 # Sample format and size
DEFAULT_AUDIO_DEVICE_INDEX = None # Use None for system's default input device, or specify an index

# Video settings
VIDEO_FPS_DEFAULT = 20.0  # Default FPS if camera doesn't provide it or for consistency
VIDEO_FOURCC = 'mp4v'     # Codec for MP4. Common options: 'mp4v', 'XVID', 'MJPG'.
                          # 'H264' might require a GStreamer backend for OpenCV.


def get_next_session_index(folder_path: str) -> int:
    """
    Determines the next session index by looking for files matching the pattern
    (video_ip_N, video_usb_N, audio_N) in the given folder.
    The next session index will be max_found_index + 1.
    """
    if not os.path.isdir(folder_path):
        return 1
    
    max_index = 0
    # Regex to find _N part in filenames like video_ip_N.mp4, video_usb_N.mp4, audio_N.wav
    pattern = re.compile(r"(?:video_ip|video_usb|audio)_(\d+)\.(?:mp4|wav)")
    
    for f_name in os.listdir(folder_path):
        match = pattern.match(f_name)
        if match:
            try:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
            except ValueError:
                print(f"Warning: Could not parse index from filename {f_name}")
                continue
    return max_index + 1


def video_recorder(source, output_filename: str, stop_event: threading.Event, 
                   desired_fps: float, fourcc_str: str):
    """
    Records video from the given source to the output_filename.
    Stops when stop_event is set.
    'source' can be an IP camera URL (string) or a device index (int).
    """
    print(f"Attempting to open video source: {source}")
    if isinstance(source, str) and source.lower().startswith("http"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG) # CAP_FFMPEG can sometimes help with IP streams
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'. This recording thread will exit.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps_from_source = cap.get(cv2.CAP_PROP_FPS)
    if fps_from_source > 0:
        effective_fps = fps_from_source
        print(f"Source '{source}' reported FPS: {effective_fps:.2f}")
    else:
        effective_fps = desired_fps
        print(f"Warning: FPS not available from source '{source}'. Using configured default FPS: {effective_fps:.2f}")

    if width == 0 or height == 0:
        print(f"Error: Could not get valid frame dimensions from source '{source}' (Width: {width}, Height: {height}). This recording thread will exit.")
        cap.release()
        return

    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    except Exception as e:
        print(f"Error: Could not create FOURCC '{fourcc_str}': {e}. Ensure OpenCV has codec support. This recording thread will exit.")
        cap.release()
        return
        
    out = cv2.VideoWriter(output_filename, fourcc, effective_fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for '{output_filename}'. Check codec, permissions, and disk space. This recording thread will exit.")
        cap.release()
        return

    print(f"Recording video from '{source}' to '{output_filename}' ({width}x{height} @ {effective_fps:.2f} FPS using {fourcc_str})")
    
    frame_count = 0
    start_time = time.time()
    
    window_name = f"Video Feed: {source}"
    cv2.namedWindow(window_name)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame from source '{source}'. Stream may have ended or an error occurred.")
            break
        out.write(frame)
        frame_count += 1

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"'q' pressed in window {window_name}. Signaling stop for all recordings.")
            stop_event.set()
            break
        # A small sleep can be added here if CPU usage is too high,
        # e.g., time.sleep(0.001), but cap.read() is often blocking.

    end_time = time.time()
    duration = end_time - start_time
    actual_avg_fps = frame_count / duration if duration > 0 else 0
    print(f"Finished recording from '{source}'. Saved {frame_count} frames in {duration:.2f} seconds (Avg FPS: {actual_avg_fps:.2f}). Releasing resources.")
    
    cap.release()
    out.release()
    cv2.destroyWindow(window_name)


def audio_recorder(output_filename: str, stop_event: threading.Event, 
                   rate: int, chunk_size: int, channels: int, audio_format, 
                   device_index: int = None):
    """
    Records audio from the specified or default microphone to the output_filename.
    Stops when stop_event is set.
    """
    audio_interface = pyaudio.PyAudio()
    
    try:
        stream = audio_interface.open(format=audio_format,
                                      channels=channels,
                                      rate=rate,
                                      input=True,
                                      frames_per_buffer=chunk_size,
                                      input_device_index=device_index)
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        print("Please ensure a microphone is connected and configured correctly.")
        print("Available audio input devices (if enumeration is possible):")
        try:
            for i in range(audio_interface.get_device_count()):
                dev_info = audio_interface.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels', 0) > 0: # Check if it's an input device
                    print(f"  Index {i}: {dev_info['name']} (Input Channels: {dev_info['maxInputChannels']})")
        except Exception as list_e:
            print(f"  Could not list audio devices: {list_e}")
        audio_interface.terminate()
        return

    print(f"Recording audio to '{output_filename}' (Rate: {rate}, Channels: {channels})")
    frames = []

    audio_window_name = "Audio Level"
    cv2.namedWindow(audio_window_name)
    vis_frame_height = 100
    vis_frame_width = 400 # Increased width for better waveform display
    
    while not stop_event.is_set():
        try:
            data = stream.read(chunk_size, exception_on_overflow=False) # Avoid crashing on overflow
            frames.append(data)

            # Audio visualization
            try:
                audio_data_np = np.frombuffer(data, dtype=np.int16)
                
                vis_frame = np.zeros((vis_frame_height, vis_frame_width, 3), dtype=np.uint8)
                
                # Max amplitude for int16
                max_amplitude = 32767.0 
                
                num_samples = len(audio_data_np)
                
                if num_samples > 0:
                    # Create an array of x-coordinates, scaled to frame width
                    x_coords = np.linspace(0, vis_frame_width - 1, num_samples, dtype=int)
                    
                    # Scale y-coordinates (amplitude)
                    # y_normalized = audio_data_np / max_amplitude  (range -1 to 1)
                    # y_pixels = (vis_frame_height / 2) * (1 - y_normalized)
                    y_coords = (vis_frame_height / 2) * (1 - audio_data_np.astype(np.float32) / max_amplitude)
                    y_coords = np.clip(y_coords, 0, vis_frame_height - 1).astype(int)

                    points = np.column_stack((x_coords, y_coords))
                    cv2.polylines(vis_frame, [points], isClosed=False, color=(0, 255, 0), thickness=1)
                cv2.line(vis_frame, (0, vis_frame_height // 2), (vis_frame_width, vis_frame_height // 2), (100, 100, 100), 1) # Center line
                cv2.imshow(audio_window_name, vis_frame)
            except Exception as e_vis:
                print(f"Warning: Error during audio visualization: {e_vis}")

        except IOError as e:
            # pyaudio.paInputOverflowed is a common one if system is busy
            if hasattr(pyaudio, 'paInputOverflowed') and e.errno == pyaudio.paInputOverflowed:
                print("Warning: Audio input overflowed. Some audio frames may have been lost.")
            else:
                print(f"Audio recording IOError: {e}. Stopping audio recording.")
                break 
        except Exception as e:
            print(f"Unexpected error during audio read: {e}. Stopping audio recording.")
            break
        
        # Check for 'q' press in audio window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"'q' pressed in window {audio_window_name}. Signaling stop for all recordings.")
            stop_event.set()
            break

    print("Finished audio recording. Processing and saving to file...")
    
    try:
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Error stopping/closing audio stream: {e}")
    
    audio_interface.terminate()
    cv2.destroyWindow(audio_window_name)

    if not frames:
        print(f"No audio frames recorded for '{output_filename}'. File will not be created.")
        return

    try:
        with wave.open(output_filename, 'wb') as wave_file:
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(audio_interface.get_sample_size(audio_format))
            wave_file.setframerate(rate)
            wave_file.writeframes(b''.join(frames))
        print(f"Audio saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving WAV file '{output_filename}': {e}")


def main():
    # Createing the recording folder if it doesn't exist
    record_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RECORD_FOLDER)
    if not os.path.exists(record_folder_path):
        try:
            os.makedirs(record_folder_path)
            print(f"Created directory: {record_folder_path}")
        except OSError as e:
            print(f"Error creating directory {record_folder_path}: {e}")
            return

    parser = argparse.ArgumentParser(description="Record video and audio from specified sources.")
    parser.add_argument(
        '--device',
        type=str,
        help="Specify video recording source: 'web' or 'ip' for IP camera, or a numeric index for USB camera (e.g., '0'). If not provided, uses settings from script configuration."
    )
    args = parser.parse_args()

    session_index = get_next_session_index(record_folder_path)
    print(f"Starting recording session {session_index}")

    stop_event = threading.Event()
    active_threads = []

    # Determine which video sources to use based on command-line args or config
    should_record_ip_cam = False
    should_record_usb_cam = False
    usb_cam_index_to_use = USB_CAM_INDEX # Default

    if args.device:
        device_arg = args.device.lower()
        if device_arg in ["web", "ip"]:
            if IP_CAM_URL:
                should_record_ip_cam = True
                print(f"Command-line override: Will attempt to record from IP camera ({IP_CAM_URL}).")
            else:
                print("Warning: --device specified IP camera, but IP_CAM_URL is not configured in the script. IP camera will not be recorded.")
        elif device_arg.isdigit():
            usb_cam_index_to_use = int(device_arg)
            should_record_usb_cam = True
            print(f"Command-line override: Will attempt to record from USB camera index {usb_cam_index_to_use}.")
        else:
            print(f"Error: Invalid --device specifier: '{args.device}'. Must be 'web', 'ip', or a numeric USB index.")
            print("No video sources will be recorded due to invalid argument. Audio might still record if enabled.")
    else:
        print("No --device argument provided. Using script configuration for video sources.")
        if IP_CAM_URL:
            should_record_ip_cam = True
        if USB_CAM_INDEX is not None:
            should_record_usb_cam = True
            usb_cam_index_to_use = USB_CAM_INDEX

    # --- Setup IP Camera Recording Thread ---
    if should_record_ip_cam and IP_CAM_URL: # Double check IP_CAM_URL in case it was None but should_record_ip_cam was true from bad logic path (though current logic prevents this)
        video_ip_filename = os.path.join(record_folder_path, f"video_ip_{session_index}.mp4")
        thread_ip_cam = threading.Thread(target=video_recorder, 
                                         args=(IP_CAM_URL, video_ip_filename, stop_event, VIDEO_FPS_DEFAULT, VIDEO_FOURCC),
                                         name=f"IPCamRecorder-{session_index}")
        active_threads.append(thread_ip_cam)
    else:
        if not (args.device and args.device.lower() in ["web", "ip"] and not IP_CAM_URL): # Avoid redundant message if warning already printed
            print("IP Camera recording is not active for this session.")

    # Setup USB Camera Recording Thread
    if should_record_usb_cam: # usb_cam_index_to_use will be set if this is true
        video_usb_filename = os.path.join(record_folder_path, f"video_usb_{session_index}.mp4")
        thread_usb_cam = threading.Thread(target=video_recorder, 
                                          args=(usb_cam_index_to_use, video_usb_filename, stop_event, VIDEO_FPS_DEFAULT, VIDEO_FOURCC),
                                          name=f"USBCamRecorder-{session_index}")
        active_threads.append(thread_usb_cam)
    else:
        print("USB Camera recording is not active for this session.")
        
    # Setup Audio Recording Thread 
    if AUDIO_ENABLED:
        audio_filename = os.path.join(record_folder_path, f"audio_{session_index}.wav")
        thread_audio = threading.Thread(target=audio_recorder, 
                                        args=(audio_filename, stop_event, AUDIO_RATE, AUDIO_CHUNK_SIZE, AUDIO_CHANNELS, AUDIO_FORMAT, DEFAULT_AUDIO_DEVICE_INDEX),
                                        name=f"AudioRecorder-{session_index}")
        active_threads.append(thread_audio)
    else:
        print("Audio recording is disabled by configuration.")

    if not active_threads:
        print("No recording sources (video or audio) are active for this session. Exiting.")
        return

    for thread in active_threads:
        thread.start()

    print("\nRecording started for configured sources.")
    is_interactive = sys.stdin.isatty()
    if is_interactive:
        print("Press 'q' and then Enter to stop all recordings gracefully.")
    print("Alternatively, press Ctrl+C to interrupt.")

    try:
        if is_interactive:
            while not stop_event.is_set() and any(t.is_alive() for t in active_threads):
                try:
                    user_input = input() # Blocking call
                    if user_input.strip().lower() == 'q':
                        print("User requested stop ('q'). Signaling threads...")
                        stop_event.set()
                        break 
                except EOFError: # Handle if stdin is unexpectedly closed
                    print("\nEOF on stdin. Assuming non-interactive mode now. Send SIGINT (Ctrl+C) to stop.")
                    is_interactive = False # Switch to non interactive wait
                    break # Break from input loop to non interactive wait loop
        
        # Non interactive wait loop (or if switched from interactive)
        while not stop_event.is_set() and any(t.is_alive() for t in active_threads):
            time.sleep(0.5) # Keep main thread alive, periodically check stop_event

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Signaling all threads to stop...")
        stop_event.set()
    except Exception as e:
        print(f"\nUnexpected error in main control loop: {e}. Signaling stop...")
        stop_event.set()
    finally:
        if not stop_event.is_set(): # If loop exited for other reasons (e.g. all threads died)
            print("Main loop ended; ensuring stop signal is sent to any remaining threads.")
            stop_event.set()

    print("Waiting for all recording threads to finish gracefully...")
    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=10) # Wait for up to 10 seconds per thread
            if thread.is_alive():
                print(f"Warning: Thread {thread.name} did not terminate gracefully after 10s.")
        else:
            print(f"Thread {thread.name} had already finished.")

    print(f"\nAll recording operations for session {session_index} concluded.")
    print(f"Files should be in the '{record_folder_path}' directory.")
    # A final call to destroy all OpenCV windows, in case some were missed or main loop exited abruptly.
    # However, individual destroyWindow calls in threads are generally preferred for cleaner shutdown.
    # If threads ensure their windows are closed, this might not be strictly necessary.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
