import schedule
import time
import subprocess
import sys
import os # We need this to check if the file exists

# --- CONFIGURATION ---
# Set the path to your mp3 file.
# If it's in the same folder as the script, "morning.mp3" is fine.
LOCAL_FILE_PATH = "morning.mp3" 
TARGET_TIME = "01:30"  # 4:30 AM in 24-hour format
# --- END CONFIGURATION ---

def play_local_audio():
    """
    This function is called by the scheduler at the target time.
    It uses 'cvlc' (command-line VLC) to play the local mp3 file.
    """
    print(f"[{time.ctime()}] It's {TARGET_TIME}! Playing {LOCAL_FILE_PATH} with VLC...")
    
    try:
        # Use cvlc (command-line VLC) with:
        # vlc://quit: A special command to make VLC exit after playing
        subprocess.run(
            ["cvlc", LOCAL_FILE_PATH, "vlc://quit"], 
            check=True
        )
        print(f"[{time.ctime()}] Audio playback finished.")
    
    except FileNotFoundError:
        print("\n*** ERROR: 'cvlc' command not found. ***")
        print("Please make sure VLC is installed correctly.")
        sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print(f"\n*** ERROR: VLC failed to play the file. ***")
        print(f"Error: {e}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """
    Main function to set up the schedule and run the loop.
    """
    # --- Check if the file exists BEFORE starting ---
    if not os.path.isfile(LOCAL_FILE_PATH):
        print(f"*** ERROR: File not found! ***")
        # os.path.abspath shows the full path it was looking for
        print(f"Path: {os.path.abspath(LOCAL_FILE_PATH)}")
        print("Please check the 'LOCAL_FILE_PATH' variable in the script,")
        print("or make sure the file is in the same folder.")
        sys.exit(1)
    # --- End check ---

    print(f"Script started. Waiting to play '{LOCAL_FILE_PATH}' at {TARGET_TIME}...")
    
    # Schedule the job
    schedule.every().day.at(TARGET_TIME).do(play_local_audio)

    # Main loop to keep the script alive and check the schedule
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScript stopped by user.")

if __name__ == "__main__":
    main()