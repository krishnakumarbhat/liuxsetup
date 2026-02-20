from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager
import time

app = Flask(__name__)

# Number of parallel downloads (adjust based on your internet speed)
MAX_WORKERS = 4

# Use a manager for thread-safe status updates
manager = None
download_status = None

def init_status():
    """Initialize thread-safe download status"""
    global manager, download_status
    manager = Manager()
    download_status = manager.dict({
        'is_downloading': False,
        'current_videos': manager.list(),
        'total_videos': 0,
        'completed_videos': 0,
        'errors': manager.list(),
        'completed': False,
        'video_progress': manager.dict(),
        # New fields for multiple playlists
        'current_playlist': 0,
        'total_playlists': 0,
        'playlist_name': '',
        'playlists_completed': manager.list()
    })

# Initialize on import
init_status()

def reset_status():
    """Reset download status"""
    global download_status
    download_status['is_downloading'] = False
    download_status['current_videos'] = manager.list()
    download_status['total_videos'] = 0
    download_status['completed_videos'] = 0
    download_status['errors'] = manager.list()
    download_status['completed'] = False
    download_status['video_progress'] = manager.dict()
    download_status['current_playlist'] = 0
    download_status['total_playlists'] = 0
    download_status['playlist_name'] = ''
    download_status['playlists_completed'] = manager.list()

def reset_playlist_status():
    """Reset status for a new playlist while keeping overall progress"""
    global download_status
    download_status['current_videos'] = manager.list()
    download_status['total_videos'] = 0
    download_status['completed_videos'] = 0
    download_status['video_progress'] = manager.dict()

def download_single_video(video_info, save_path, index, total):
    """Download a single video from the playlist with retries"""
    video_url = video_info.get('url') or video_info.get('webpage_url') or f"https://www.youtube.com/watch?v={video_info.get('id')}"
    video_title = video_info.get('title', 'Unknown')
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Update current video status
            current_list = list(download_status['current_videos'])
            status_msg = f"[{index}/{total}] {video_title}" if attempt == 0 else f"[{index}/{total}] {video_title} (retry {attempt + 1})"
            if status_msg not in current_list:
                current_list.append(status_msg)
                download_status['current_videos'] = manager.list(current_list)
            
            # Update progress dict
            progress_dict = dict(download_status['video_progress'])
            progress_dict[video_title] = {'status': 'downloading', 'progress': '0%', 'attempt': attempt + 1}
            download_status['video_progress'] = manager.dict(progress_dict)
            
            # Configure yt-dlp options with better network settings
            ydl_opts = {
                'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
                'outtmpl': os.path.join(save_path, f'{index:03d} - %(title)s.%(ext)s'),
                'ignoreerrors': False,
                'no_warnings': True,
                'quiet': True,
                'no_color': True,
                'retries': 10,
                'fragment_retries': 10,
                'skip_unavailable_fragments': True,
                'socket_timeout': 30,
                'http_chunk_size': 10485760,
                'concurrent_fragment_downloads': 1,
                'noprogress': False,
                'continuedl': True,
                'nooverwrites': False,
                'external_downloader_args': {'ffmpeg': ['-loglevel', 'quiet']},
            }
            
            def progress_hook(d):
                if d['status'] == 'downloading':
                    percent = d.get('_percent_str', '0%').strip()
                    speed = d.get('_speed_str', 'N/A').strip()
                    eta = d.get('_eta_str', 'N/A').strip()
                    progress_dict = dict(download_status['video_progress'])
                    progress_dict[video_title] = {
                        'status': 'downloading', 
                        'progress': percent,
                        'speed': speed,
                        'eta': eta,
                        'attempt': attempt + 1
                    }
                    download_status['video_progress'] = manager.dict(progress_dict)
                elif d['status'] == 'finished':
                    progress_dict = dict(download_status['video_progress'])
                    progress_dict[video_title] = {'status': 'finished', 'progress': '100%'}
                    download_status['video_progress'] = manager.dict(progress_dict)
            
            ydl_opts['progress_hooks'] = [progress_hook]
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Update completed count
            download_status['completed_videos'] = download_status['completed_videos'] + 1
            
            # Remove from current videos
            current_list = list(download_status['current_videos'])
            for item in current_list[:]:
                if video_title in item:
                    current_list.remove(item)
            download_status['current_videos'] = manager.list(current_list)
            
            return {'success': True, 'title': video_title}
            
        except Exception as e:
            error_str = str(e)
            print(f"Attempt {attempt + 1}/{max_retries} failed for '{video_title}': {error_str}")
            
            progress_dict = dict(download_status['video_progress'])
            progress_dict[video_title] = {
                'status': 'retrying', 
                'progress': f'Retry {attempt + 2}/{max_retries}',
                'error': error_str[:50]
            }
            download_status['video_progress'] = manager.dict(progress_dict)
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                error_msg = f"Failed to download '{video_title}' after {max_retries} attempts: {error_str[:100]}"
                errors_list = list(download_status['errors'])
                errors_list.append(error_msg)
                download_status['errors'] = manager.list(errors_list)
                download_status['completed_videos'] = download_status['completed_videos'] + 1
                
                current_list = list(download_status['current_videos'])
                for item in current_list[:]:
                    if video_title in item:
                        current_list.remove(item)
                download_status['current_videos'] = manager.list(current_list)
                
                return {'success': False, 'title': video_title, 'error': error_str}
    
    return {'success': False, 'title': video_title, 'error': 'Unknown error'}

def download_single_playlist(playlist_url, save_path, num_workers, playlist_index, total_playlists):
    """Download a single playlist"""
    global download_status
    
    try:
        # Reset playlist-specific status
        reset_playlist_status()
        download_status['current_playlist'] = playlist_index
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Playlist {playlist_index}/{total_playlists}: {playlist_url}")
        print(f"Save path: {save_path}")
        print(f"{'='*60}")
        
        # Get playlist info
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
        
        if not playlist_info or 'entries' not in playlist_info:
            error_msg = f"Playlist {playlist_index}: Could not extract playlist information"
            errors_list = list(download_status['errors'])
            errors_list.append(error_msg)
            download_status['errors'] = manager.list(errors_list)
            return {'success': False, 'playlist_index': playlist_index}
        
        playlist_name = playlist_info.get('title', f'Playlist {playlist_index}')
        download_status['playlist_name'] = playlist_name
        
        entries = [e for e in playlist_info['entries'] if e is not None]
        total_videos = len(entries)
        download_status['total_videos'] = total_videos
        
        print(f"Found {total_videos} videos in '{playlist_name}'. Starting download...")
        
        # Download videos in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {
                executor.submit(download_single_video, entry, save_path, idx + 1, total_videos): entry
                for idx, entry in enumerate(entries)
            }
            
            for future in as_completed(future_to_video):
                try:
                    result = future.result()
                    if result['success']:
                        print(f"✓ Completed: {result['title']}")
                    else:
                        print(f"✗ Failed: {result['title']} - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"✗ Task error: {str(e)}")
        
        # Mark playlist as completed
        completed_list = list(download_status['playlists_completed'])
        completed_list.append({
            'index': playlist_index,
            'name': playlist_name,
            'total': total_videos,
            'completed': download_status['completed_videos']
        })
        download_status['playlists_completed'] = manager.list(completed_list)
        
        print(f"✅ Playlist '{playlist_name}' complete! {download_status['completed_videos']}/{total_videos} videos downloaded.")
        return {'success': True, 'playlist_index': playlist_index, 'name': playlist_name}
        
    except Exception as e:
        error_msg = f"Playlist {playlist_index} error: {str(e)}"
        errors_list = list(download_status['errors'])
        errors_list.append(error_msg)
        download_status['errors'] = manager.list(errors_list)
        print(f"Error: {str(e)}")
        return {'success': False, 'playlist_index': playlist_index, 'error': str(e)}

def download_multiple_playlists(playlists, num_workers=MAX_WORKERS):
    """Download multiple playlists sequentially"""
    global download_status
    
    total_playlists = len(playlists)
    download_status['total_playlists'] = total_playlists
    
    print(f"\n{'#'*60}")
    print(f"Starting download of {total_playlists} playlists")
    print(f"{'#'*60}")
    
    for idx, playlist in enumerate(playlists, 1):
        playlist_url = playlist['url']
        save_path = playlist['path']
        
        if not playlist_url or not save_path:
            continue
        
        result = download_single_playlist(playlist_url, save_path, idx, total_playlists, num_workers)
        
        # Small delay between playlists
        if idx < total_playlists:
            print(f"\nWaiting 3 seconds before next playlist...")
            time.sleep(3)
    
    download_status['completed'] = True
    download_status['is_downloading'] = False
    
    print(f"\n{'#'*60}")
    print(f"All {total_playlists} playlists processed!")
    print(f"{'#'*60}")

def download_single_playlist(playlist_url, save_path, playlist_index, total_playlists, num_workers):
    """Download a single playlist - fixed version with num_workers parameter"""
    global download_status
    
    try:
        # Reset playlist-specific status
        reset_playlist_status()
        download_status['current_playlist'] = playlist_index
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Playlist {playlist_index}/{total_playlists}: {playlist_url}")
        print(f"Save path: {save_path}")
        print(f"{'='*60}")
        
        # Get playlist info
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
        
        if not playlist_info or 'entries' not in playlist_info:
            error_msg = f"Playlist {playlist_index}: Could not extract playlist information"
            errors_list = list(download_status['errors'])
            errors_list.append(error_msg)
            download_status['errors'] = manager.list(errors_list)
            return {'success': False, 'playlist_index': playlist_index}
        
        playlist_name = playlist_info.get('title', f'Playlist {playlist_index}')
        download_status['playlist_name'] = playlist_name
        
        entries = [e for e in playlist_info['entries'] if e is not None]
        total_videos = len(entries)
        download_status['total_videos'] = total_videos
        
        print(f"Found {total_videos} videos in '{playlist_name}'. Starting download...")
        
        # Download videos in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_video = {
                executor.submit(download_single_video, entry, save_path, idx + 1, total_videos): entry
                for idx, entry in enumerate(entries)
            }
            
            for future in as_completed(future_to_video):
                try:
                    result = future.result()
                    if result['success']:
                        print(f"✓ Completed: {result['title']}")
                    else:
                        print(f"✗ Failed: {result['title']} - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"✗ Task error: {str(e)}")
        
        # Mark playlist as completed
        completed_list = list(download_status['playlists_completed'])
        completed_list.append({
            'index': playlist_index,
            'name': playlist_name,
            'total': total_videos,
            'completed': download_status['completed_videos']
        })
        download_status['playlists_completed'] = manager.list(completed_list)
        
        print(f"✅ Playlist '{playlist_name}' complete! {download_status['completed_videos']}/{total_videos} videos downloaded.")
        return {'success': True, 'playlist_index': playlist_index, 'name': playlist_name}
        
    except Exception as e:
        error_msg = f"Playlist {playlist_index} error: {str(e)}"
        errors_list = list(download_status['errors'])
        errors_list.append(error_msg)
        download_status['errors'] = manager.list(errors_list)
        print(f"Error: {str(e)}")
        return {'success': False, 'playlist_index': playlist_index, 'error': str(e)}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    """Start downloading multiple playlists"""
    global download_status
    
    if download_status['is_downloading']:
        return jsonify({'error': 'A download is already in progress'}), 400
    
    data = request.get_json()
    playlists = data.get('playlists', [])
    num_workers = data.get('parallel_downloads', MAX_WORKERS)
    
    # Validate number of workers
    try:
        num_workers = int(num_workers)
        num_workers = max(1, min(num_workers, 8))
    except:
        num_workers = MAX_WORKERS
    
    # Filter valid playlists
    valid_playlists = []
    for p in playlists:
        url = p.get('url', '').strip()
        path = p.get('path', '').strip()
        if url and path and ('youtube.com' in url or 'youtu.be' in url):
            valid_playlists.append({'url': url, 'path': path})
    
    if not valid_playlists:
        return jsonify({'error': 'Please provide at least one valid playlist URL and save path'}), 400
    
    if len(valid_playlists) > 10:
        return jsonify({'error': 'Maximum 10 playlists allowed'}), 400
    
    # Reset and start download
    reset_status()
    download_status['is_downloading'] = True
    download_status['total_playlists'] = len(valid_playlists)
    
    # Start download in background thread
    thread = threading.Thread(target=download_multiple_playlists, args=(valid_playlists, num_workers))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'Started downloading {len(valid_playlists)} playlists with {num_workers} parallel downloads'})

@app.route('/status')
def status():
    """Get current download status"""
    return jsonify({
        'is_downloading': download_status['is_downloading'],
        'current_videos': list(download_status['current_videos']),
        'total_videos': download_status['total_videos'],
        'completed_videos': download_status['completed_videos'],
        'errors': list(download_status['errors']),
        'completed': download_status['completed'],
        'video_progress': dict(download_status['video_progress']),
        'current_playlist': download_status['current_playlist'],
        'total_playlists': download_status['total_playlists'],
        'playlist_name': download_status['playlist_name'],
        'playlists_completed': list(download_status['playlists_completed'])
    })

@app.route('/cancel', methods=['POST'])
def cancel():
    """Cancel the current download"""
    global download_status
    reset_status()
    return jsonify({'message': 'Download cancelled'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)