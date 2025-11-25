#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# Configuration
DOWNLOADS_DIR = Path.home() / "Downloads"

# Category Mappings
CATEGORIES = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".tiff"],
    "Documents": [".pdf", ".doc", ".docx", ".txt", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".csv", ".md"],
    "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
    "Video": [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"],
    "Archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".iso"],
    "Installers": [".deb", ".sh", ".run", ".AppImage", ".exe", ".msi"],
    "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h", ".json", ".xml", ".sql"],
}

def organize_downloads():
    if not DOWNLOADS_DIR.exists():
        print(f"❌ Downloads directory not found: {DOWNLOADS_DIR}")
        return

    print(f"📂 Organizing {DOWNLOADS_DIR}...")

    # Create category directories
    for category in CATEGORIES:
        (DOWNLOADS_DIR / category).mkdir(exist_ok=True)
    
    # Create 'Others' directory
    (DOWNLOADS_DIR / "Others").mkdir(exist_ok=True)

    moved_count = 0

    for item in DOWNLOADS_DIR.iterdir():
        if item.is_dir():
            continue
        
        # Skip hidden files
        if item.name.startswith('.'):
            continue

        file_ext = item.suffix.lower()
        destination = None

        for category, extensions in CATEGORIES.items():
            if file_ext in extensions:
                destination = DOWNLOADS_DIR / category
                break
        
        if destination is None:
            destination = DOWNLOADS_DIR / "Others"

        try:
            shutil.move(str(item), str(destination / item.name))
            print(f"✅ Moved: {item.name} -> {destination.name}/")
            moved_count += 1
        except Exception as e:
            print(f"❌ Error moving {item.name}: {e}")

    print(f"✨ Done! Organized {moved_count} files.")

if __name__ == "__main__":
    organize_downloads()
