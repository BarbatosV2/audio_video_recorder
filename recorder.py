import cv2
import pyaudio
import wave
import threading
import os
import time
import re
import sys
import asyncio # Added for WebSocket audio
import websockets # Added for WebSocket audio
import argparse # Added for command-line argument parsing

# --- Configuration ---
RECORD_FOLDER = "record"  # Name of the subfolder for recordings

# IP Camera settings
IP_CAM_URL = "http://172.20.10.2:81/stream"  # Your IP camera stream URL
                                             # Set to None or "" to disable IP camera recording

# WebSocket Audio settings (for IP camera audio source)
WS_AUDIO_URL = "ws://172.20.10.2:82"       # Your WebSocket audio stream URL (e.g., from an IP camera)
                                           # Set to None or "" to disable WebSocket audio recording
# USB Webcam settings
USB_CAM_INDEX = 0  # Index of the USB webcam (usually 0 for the default)
                   # Set to None to disable USB camera recording

# Audio settings
AUDIO_ENABLED = True  # Set to False to disable audio recording
AUDIO_RATE = 16000    # Sample rate for audio
AUDIO_CHUNK_SIZE = 1024 # Number of frames per buffer
AUDIO_CHANNELS = 1      # 1 for mono, 2 for stereo
AUDIO_FORMAT = pyaudio.paInt16 # Sample format and size
DEFAULT_AUDIO_DEVICE_INDEX = None # Use None for system's default input device, or specify an index

# Video settings
VIDEO_FPS_DEFAULT = 20.0  # Default FPS if camera doesn't provide it or for consistency
VIDEO_FOURCC = 'mp4v'     # Codec for MP4. Common options: 'mp4v', 'XVID', 'MJPG'.
                          # 'H264' might require a GStreamer backend for OpenCV.
# Determine audio sample width in bytes based on AUDIO_FORMAT
try:
    # Temporary PyAudio instance to get sample width
    _pyaudio_instance_for_format = pyaudio.PyAudio()
    AUDIO_SAMPLE_WIDTH_BYTES = _pyaudio_instance_for_format.get_sample_size(AUDIO_FORMAT)
    _pyaudio_instance_for_format.terminate()
except Exception as e:
    print(f"Warning: Could not determine audio sample width from PyAudio format {AUDIO_FORMAT}: {e}. Defaulting to 2 bytes.")
    AUDIO_SAMPLE_WIDTH_BYTES = 2 # Default for paInt16

def get_next_session_index(folder_path: str) -> int:
    """
    Determines the next session index by looking for files matching the pattern
    (video_ip_N, video_usb_N, audio_mic_N, audio_ws_N) in the given folder.
    The next session index will be max_found_index + 1.
    """
    if not os.path.isdir(folder_path):
        return 1
    
    max_index = 0
    # Regex to find _N part in filenames like:
    # video_ip_N.mp4, video_usb_N.mp4, audio_mic_N.wav, audio_ws_N.wav
    pattern = re.compile(
        r"(?:video_ip|video_usb|audio_mic|audio_ws)_(\d+)\.(?:mp4|wav)"
    )
    
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
    window_title = f"LIVE Preview: {source}" # Unique window title for this source

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # Window will be destroyed after the loop
            print(f"Warning: Could not read frame from source '{source}'. Stream may have ended or an error occurred.")
            break
        out.write(frame)
        frame_count += 1
        # A small sleep can be added here if CPU usage is too high,
        # e.g., time.sleep(0.001), but cap.read() is often blocking.

        # Display the current frame
        cv2.imshow(window_title, frame)
        # cv2.waitKey(1) is crucial for imshow to work and update the window.
        # It also processes window events. A 1ms delay is generally negligible.
        cv2.waitKey(1)

    end_time = time.time()
    duration = end_time - start_time
    actual_avg_fps = frame_count / duration if duration > 0 else 0
    print(f"Finished recording from '{source}'. Saved {frame_count} frames in {duration:.2f} seconds (Avg FPS: {actual_avg_fps:.2f}). Releasing resources.")
    
    cap.release()
    out.release()
    cv2.destroyWindow(window_title) # Close the specific preview window for this thread


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
    
    while not stop_event.is_set():
        try:
            data = stream.read(chunk_size, exception_on_overflow=False) # Avoid crashing on overflow
            frames.append(data)
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

    print("Finished audio recording. Processing and saving to file...")
    
    try:
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Error stopping/closing audio stream: {e}")
    
    audio_interface.terminate()

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


async def websocket_audio_consumer(uri: str, stop_event: threading.Event, frames_list: list, output_filename_debug: str):
    """
    Connects to a WebSocket audio stream and appends received binary data to frames_list.
    Stops when stop_event is set.
    """
    print(f"Attempting to connect to WebSocket audio stream: {uri} for {output_filename_debug}")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Successfully connected to WebSocket audio stream: {uri}")
            while not stop_event.is_set():
                try:
                    # Timeout allows checking stop_event periodically
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    if isinstance(message, bytes):
                        frames_list.append(message)
                    # else:
                        # print(f"WS Audio ({output_filename_debug}): Received non-bytes message: {type(message)}")
                except asyncio.TimeoutError:
                    continue  # No message received, loop to check stop_event
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"WS Audio ({output_filename_debug}): Connection closed by server ({uri}): {e}")
                    break
                except Exception as e:
                    print(f"WS Audio ({output_filename_debug}): Error receiving from WebSocket ({uri}): {e}")
                    break
    except websockets.exceptions.InvalidURI:
        print(f"Error: Invalid WebSocket URI: {uri}. WebSocket audio recording for {output_filename_debug} will not start.")
    except ConnectionRefusedError:
        print(f"Error: Connection refused for WebSocket: {uri}. Ensure the server is running. WebSocket audio recording for {output_filename_debug} will not start.")
    except Exception as e:
        print(f"Error connecting to WebSocket {uri} for {output_filename_debug}: {e}. WebSocket audio recording will not start.")
    finally:
        print(f"WebSocket audio consumer for {output_filename_debug} stopping.")


def websocket_audio_recorder_thread_target(uri: str, output_filename: str, stop_event: threading.Event,
                                           rate: int, channels: int, sample_width_bytes: int):
    """
    Thread target for recording audio from a WebSocket stream.
    Manages an asyncio event loop for websocket_audio_consumer and saves the audio to a WAV file.
    """
    frames = []
    print(f"Starting WebSocket audio recording to '{output_filename}' from '{uri}' (Rate: {rate}, Channels: {channels}, SampleWidth: {sample_width_bytes} bytes)")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(websocket_audio_consumer(uri, stop_event, frames, output_filename))
    except Exception as e:
        print(f"Error in WebSocket audio consumer event loop for {output_filename}: {e}")
    finally:
        loop.close()

    print(f"Finished WebSocket audio data collection for '{output_filename}'. Processing and saving to file...")
    if not frames:
        print(f"No audio frames received from WebSocket for '{output_filename}'. File will not be created.")
        return
    try:
        with wave.open(output_filename, 'wb') as wave_file:
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(sample_width_bytes)
            wave_file.setframerate(rate)
            wave_file.writeframes(b''.join(frames))
        print(f"WebSocket audio saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving WAV file from WebSocket stream '{output_filename}': {e}")

def main():
    # Create record folder if it doesn't exist
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

    # --- Setup USB Camera Recording Thread ---
    if should_record_usb_cam: # usb_cam_index_to_use will be set if this is true
        video_usb_filename = os.path.join(record_folder_path, f"video_usb_{session_index}.mp4")
        thread_usb_cam = threading.Thread(target=video_recorder, 
                                          args=(usb_cam_index_to_use, video_usb_filename, stop_event, VIDEO_FPS_DEFAULT, VIDEO_FOURCC),
                                          name=f"USBCamRecorder-{session_index}")
        active_threads.append(thread_usb_cam)
    else:
        print("USB Camera recording is not active for this session.")
        
    # --- Setup Audio Recording Thread(s) ---
    audio_thread_started_this_session = False
    # Prioritize WebSocket audio if --device ip is used and WS_AUDIO_URL is configured
    if should_record_ip_cam and WS_AUDIO_URL: # IP Cam video is active, and WS Audio URL is set
        audio_ws_filename = os.path.join(record_folder_path, f"audio_ws_{session_index}.wav")
        thread_ws_audio = threading.Thread(
            target=websocket_audio_recorder_thread_target,
            args=(WS_AUDIO_URL, audio_ws_filename, stop_event, AUDIO_RATE, AUDIO_CHANNELS, AUDIO_SAMPLE_WIDTH_BYTES),
            name=f"WSAudioRecorder-{session_index}"
        )
        active_threads.append(thread_ws_audio)
        audio_thread_started_this_session = True
        print(f"WebSocket audio recording will be attempted from {WS_AUDIO_URL} to {audio_ws_filename}.")
    elif should_record_ip_cam and not WS_AUDIO_URL:
        print("IP Camera recording is active, but WS_AUDIO_URL is not configured. No IP-based audio will be recorded.")

    # Fallback to local microphone audio if not doing IP audio or if IP audio isn't configured/wanted
    if not audio_thread_started_this_session and AUDIO_ENABLED:
        # This block runs if:
        # 1. Not an IP camera session (e.g., USB cam only, or no video args) AND AUDIO_ENABLED.
        # 2. It IS an IP camera session, WS_AUDIO_URL is NOT set, AND AUDIO_ENABLED (for local mic fallback).
        audio_mic_filename = os.path.join(record_folder_path, f"audio_mic_{session_index}.wav")
        thread_mic_audio = threading.Thread(target=audio_recorder,
                                        args=(audio_mic_filename, stop_event, AUDIO_RATE, AUDIO_CHUNK_SIZE, AUDIO_CHANNELS, AUDIO_FORMAT, DEFAULT_AUDIO_DEVICE_INDEX),
                                        name=f"MicAudioRecorder-{session_index}")
        active_threads.append(thread_mic_audio)
        audio_thread_started_this_session = True
        print(f"Local microphone audio recording is active, saving to {audio_mic_filename}.")
    
    if not audio_thread_started_this_session:
        print("No audio recording sources (WebSocket or local microphone) are active for this session.")

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
                    is_interactive = False # Switch to non-interactive wait
                    break # Break from input loop to non-interactive wait loop
        
        # Non-interactive wait loop (or if switched from interactive)
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

if __name__ == "__main__":
    main()