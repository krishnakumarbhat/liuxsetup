#!/usr/bin/env python3
import os
import sys
import subprocess

def check_dependencies():
    """Check if yt-dlp is installed."""
    if shutil.which("yt-dlp") is None:
        print("❌ Error: 'yt-dlp' is not installed.")
        print("   Please install it using: sudo apt install yt-dlp  OR  pip install yt-dlp")
        sys.exit(1)

import shutil

def download_media(url, mode):
    """
    Download media using yt-dlp.
    mode: 'video' or 'audio'
    """
    if mode == 'video':
        print(f"🎬 Downloading Video: {url}")
        # Download best video+best audio, merge to mp4
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            url
        ]
    elif mode == 'audio':
        print(f"🎵 Downloading Audio: {url}")
        # Extract audio, convert to mp3
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "0", # Best quality
            url
        ]
    else:
        print("❌ Invalid mode.")
        return

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")
    except KeyboardInterrupt:
        print("\n⚠️ Download cancelled.")

def main():
    check_dependencies()

    print("--- YouTube Downloader (via yt-dlp) ---")
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("🔗 Enter YouTube URL: ").strip()

    if not url:
        print("❌ No URL provided.")
        return

    print("\nChoose format:")
    print("1. Video (Best Quality MP4)")
    print("2. Audio Only (MP3)")
    
    choice = input("Enter choice (1/2): ").strip()

    if choice == '1':
        download_media(url, 'video')
    elif choice == '2':
        download_media(url, 'audio')
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
