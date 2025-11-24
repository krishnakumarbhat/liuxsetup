# Import the AudioSegment class from the pydub library
from pydub import AudioSegment
import os

# --- Configuration ---
# Specify the path to your input M4A file
input_file = "record.m4a" 
# Specify the desired name for the output file
output_file = "first_30_minutes.m4a" 

# --- Main Logic ---
# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' was not found.")
else:
    try:
        print(f"Loading audio file: {input_file}...")
        original_audio = AudioSegment.from_file(input_file, format="m4a")
        
        # pydub works in milliseconds (ms)
        thirty_minutes_in_ms = 30 * 60 * 1000
        
        print("Slicing the first 30 minutes...")
        first_30_minutes = original_audio[:thirty_minutes_in_ms]
        
        print(f"Exporting the sliced audio to: {output_file}...")
        # Use format="mp4" because M4A is an audio-only MP4 container
        first_30_minutes.export(output_file, format="mp4")
        
        print("\n✅ Success! The first 30 minutes have been saved.")

    except Exception as e:
        print(f"An error occurred: {e}")