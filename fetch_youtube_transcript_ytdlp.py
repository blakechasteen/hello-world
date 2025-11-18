#!/usr/bin/env python3
"""
YouTube Transcript Fetcher using yt-dlp
========================================
More reliable than youtube-transcript-api because yt-dlp has better
bot detection evasion.

Usage:
    python fetch_youtube_transcript_ytdlp.py <URL>
"""

import sys
import json
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("❌ yt-dlp not installed!")
    print("Install with: pip install yt-dlp")
    sys.exit(1)


def fetch_transcript_ytdlp(url: str, output_file: str = "youtube_transcript.txt"):
    """Fetch transcript using yt-dlp"""

    print(f"🎥 Fetching transcript from: {url}")
    print("⏳ Please wait...")

    # yt-dlp options
    ydl_opts = {
        'skip_download': True,  # Don't download video
        'writesubtitles': True,  # Get subtitles
        'writeautomaticsub': True,  # Get auto-generated subtitles too
        'subtitleslangs': ['en'],  # English subtitles
        'quiet': True,  # Suppress output
        'no_warnings': True,
        'nocheckcertificate': True,  # Disable SSL verification (for environments with proxy)
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info
            info = ydl.extract_info(url, download=False)

            # Get video metadata
            video_id = info.get('id', 'unknown')
            title = info.get('title', 'Unknown Title')
            duration = info.get('duration', 0)

            print(f"✅ Found video: {title}")
            print(f"   Duration: {duration//60}:{duration%60:02d}")

            # Get subtitles
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})

            # Prefer manual subtitles, fall back to automatic
            transcript_data = None
            transcript_type = None

            if 'en' in subtitles:
                transcript_data = subtitles['en']
                transcript_type = 'manual'
            elif 'en' in automatic_captions:
                transcript_data = automatic_captions['en']
                transcript_type = 'automatic'
            else:
                print("❌ No English transcript found!")
                return

            print(f"   Transcript type: {transcript_type}")

            # Download the actual subtitle data
            # yt-dlp returns a list of formats, use the first one
            if isinstance(transcript_data, list) and len(transcript_data) > 0:
                subtitle_url = transcript_data[0].get('url')

                if subtitle_url:
                    # Download subtitle content
                    import urllib.request
                    with urllib.request.urlopen(subtitle_url) as response:
                        subtitle_content = response.read().decode('utf-8')

                    # Parse subtitle format (could be VTT, SRT, JSON, etc.)
                    # Most YouTube subtitles are in JSON3 format
                    transcript_text = parse_subtitle_content(subtitle_content)

                    # Save to file
                    output_path = Path(output_file)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"YouTube Transcript\n")
                        f.write(f"{'=' * 50}\n")
                        f.write(f"Title: {title}\n")
                        f.write(f"Video ID: {video_id}\n")
                        f.write(f"URL: {url}\n")
                        f.write(f"Transcript Type: {transcript_type}\n")
                        f.write(f"Duration: {duration//60}:{duration%60:02d}\n")
                        f.write(f"{'=' * 50}\n\n")
                        f.write(transcript_text)

                    print(f"💾 Saved to: {output_path}")

                    # Print to console
                    print(f"\n{'=' * 50}")
                    print("TRANSCRIPT:")
                    print('=' * 50)
                    print(transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text)
                else:
                    print("❌ No subtitle URL found!")
            else:
                print("❌ Unexpected subtitle format!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def parse_subtitle_content(content: str) -> str:
    """Parse subtitle content (VTT, SRT, or JSON3)"""
    # Try to detect format
    if content.startswith('WEBVTT'):
        # WebVTT format
        lines = content.split('\n')
        transcript = []
        for line in lines:
            line = line.strip()
            # Skip timing lines and empty lines
            if line and not line.startswith('WEBVTT') and '-->' not in line and not line.isdigit():
                transcript.append(line)
        return ' '.join(transcript)

    elif content.strip().startswith('{'):
        # JSON3 format
        try:
            data = json.loads(content)
            # Extract text from events
            if 'events' in data:
                transcript = []
                for event in data['events']:
                    if 'segs' in event:
                        for seg in event['segs']:
                            if 'utf8' in seg:
                                transcript.append(seg['utf8'])
                return ''.join(transcript)
        except:
            pass

    # Otherwise, just try to extract text (SRT format or unknown)
    lines = content.split('\n')
    transcript = []
    for line in lines:
        line = line.strip()
        # Skip timing lines, numbers, and empty lines
        if line and '-->' not in line and not line.isdigit():
            transcript.append(line)
    return ' '.join(transcript)


def main():
    # Get URL from command line or use default
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Use the original short URL from your request
        url = "https://youtube.com/shorts/Flqljv8clcY?si=dY_rwyIFjhlrXOxM"

    output_file = "youtube_transcript.txt"

    fetch_transcript_ytdlp(url, output_file)


if __name__ == "__main__":
    main()
