import yt_dlp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 8 parallel downloads for speed
MAX_WORKERS = 8

def download_single_video(video_info, save_path, index, total):
    """Download a single video with retries"""
    video_url = video_info.get('url') or video_info.get('webpage_url') or f"https://www.youtube.com/watch?v={video_info.get('id')}"
    video_title = video_info.get('title', 'Unknown')
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print(f"⬇️  [{index}/{total}] Downloading: {video_title}")
            else:
                print(f"🔄 [{index}/{total}] Retry {attempt+1}/{max_retries}: {video_title}")
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': os.path.join(save_path, f'{index:03d} - %(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'retries': 10,
                'fragment_retries': 10,
                'socket_timeout': 60,
                'http_chunk_size': 1048576,  # 1MB chunks
                'skip_unavailable_fragments': True,
                'continuedl': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            print(f"✅ [{index}/{total}] Done: {video_title}")
            return {'success': True, 'title': video_title}
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retry
            else:
                print(f"❌ [{index}/{total}] Failed: {video_title} - {str(e)[:50]}")
                return {'success': False, 'title': video_title, 'error': str(e)}

def download_playlist(playlist_url, save_path, skip_first=0):
    """Download playlist, optionally skip first N videos"""
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Playlist: {playlist_url}")
    print(f"Save to: {save_path}")
    if skip_first > 0:
        print(f"Skipping first {skip_first} videos (already downloaded)")
    print(f"{'='*60}")
    
    # Get playlist info
    ydl_opts = {'extract_flat': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(playlist_url, download=False)
    
    entries = [e for e in playlist_info['entries'] if e is not None]
    playlist_name = playlist_info.get('title', 'Unknown Playlist')
    
    # Skip first N videos if specified
    entries = entries[skip_first:]
    total = len(entries)
    
    print(f"📁 '{playlist_name}' - Downloading {total} videos with {MAX_WORKERS} parallel downloads\n")
    
    # Download in parallel
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_single_video, entry, save_path, skip_first + idx + 1, skip_first + total): entry
            for idx, entry in enumerate(entries)
        }
        for future in as_completed(futures):
            result = future.result()
            if result['success']:
                completed += 1
    
    print(f"\n✅ Playlist complete! {completed}/{total} videos downloaded.\n")

if __name__ == '__main__':
    # Playlist 1: ML - Skip first 8 (already downloaded)
    download_playlist(
        "https://www.youtube.com/playlist?list=PLxCzCOWd7aiEXg5BV10k9THtjnS48yI-T",
        "/media/pope/projecteo/ml2",
        skip_first=8
    )
    
    # Playlist 2: DAA - Download all
    download_playlist(
        "https://www.youtube.com/playlist?list=PLxCzCOWd7aiHcmS4i14bI0VrMbZTUvlTa",
        "/media/pope/projecteo/daa",
        skip_first=0
    )