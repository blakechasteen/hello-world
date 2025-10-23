#!/usr/bin/env python3
"""
Standalone YouTube Video Transcriber
No dependencies except youtube-transcript-api

Usage: python youtube_transcript_tool.py [VIDEO_URL]
"""

import re
import sys
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    # If it's already just an ID (11 characters)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url

    # Parse URL
    parsed = urlparse(url)

    if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        elif parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
    elif parsed.hostname in ('youtu.be',):
        return parsed.path[1:]

    return None


def get_transcript(video_id: str, languages: List[str] = None) -> Dict[str, Any]:
    """Fetch transcript for a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("Error: youtube-transcript-api is required")
        print("Install with: pip install youtube-transcript-api")
        sys.exit(1)

    if languages is None:
        languages = ['en']

    try:
        api = YouTubeTranscriptApi()

        # List available transcripts
        transcript_list = api.list(video_id)

        # Find preferred language
        selected_transcript = None
        for transcript in transcript_list:
            if transcript.language_code in languages:
                selected_transcript = transcript
                break

        # Use first available if no match
        if selected_transcript is None and transcript_list:
            selected_transcript = transcript_list[0]

        if selected_transcript is None:
            raise Exception(f"No transcripts available for video: {video_id}")

        # Fetch segments
        segments = api.fetch(video_id, languages=[selected_transcript.language_code])

        # Combine text
        full_text = ' '.join([segment['text'] for segment in segments])

        # Calculate duration
        duration = 0
        if segments:
            last = segments[-1]
            duration = last['start'] + last.get('duration', 0)

        return {
            'text': full_text,
            'language': selected_transcript.language_code,
            'is_generated': selected_transcript.is_generated,
            'segments': segments,
            'duration': duration,
            'video_id': video_id
        }

    except Exception as e:
        raise Exception(f"Could not retrieve transcript: {e}")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print(f"No URL provided, using example: {video_url}")
        print("Usage: python youtube_transcript_tool.py YOUR_VIDEO_URL\n")

    print(f"Transcribing: {video_url}")
    print("-" * 80)

    # Extract video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        print(f"Error: Invalid YouTube URL: {video_url}")
        sys.exit(1)

    print(f"Video ID: {video_id}")

    try:
        # Get transcript
        result = get_transcript(video_id, languages=['en'])

        print(f"\n✓ Success!")
        print(f"  Language: {result['language']}")
        print(f"  Type: {'Auto-generated' if result['is_generated'] else 'Manual'}")
        print(f"  Duration: {result['duration']:.1f} seconds")
        print(f"  Segments: {len(result['segments'])}")

        print(f"\n{'='*80}")
        print("TRANSCRIPT")
        print(f"{'='*80}\n")
        print(result['text'])
        print(f"\n{'='*80}")
        print(f"Total: {len(result['text'])} characters")
        print(f"{'='*80}\n")

        # Save option
        save = input("Save to file? (y/n): ").lower().strip()
        if save == 'y':
            filename = f"transcript_{video_id}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"YouTube Transcript\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
                f.write(f"Language: {result['language']}\n")
                f.write(f"Duration: {result['duration']:.1f}s\n\n")
                f.write(f"{'='*80}\n\n")
                f.write(result['text'])
                f.write(f"\n\n{'='*80}\n")
                f.write(f"Segments: {len(result['segments'])}\n")

            print(f"✓ Saved to: {filename}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure video has captions enabled")
        print("  - Try a different video")
        print("  - Check internet connection")
        print("  - Some networks may block YouTube (403 errors)")
        sys.exit(1)


if __name__ == '__main__':
    main()
