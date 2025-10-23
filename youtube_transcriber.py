#!/usr/bin/env python3
"""
YouTube Caption Transcriber
A simple tool to extract captions/transcripts from YouTube videos.
"""

import re
import sys
from typing import Optional, List, Dict
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID

    Args:
        url: YouTube URL or video ID

    Returns:
        Video ID string or None if invalid
    """
    # If it's already just an ID (11 characters, alphanumeric)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url

    # Parse various YouTube URL formats
    parsed = urlparse(url)

    if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        elif parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
    elif parsed.hostname in ('youtu.be',):
        return parsed.path[1:]

    return None


def get_transcript(video_id: str, languages: List[str] = None) -> Dict:
    """
    Fetch transcript for a YouTube video.

    Args:
        video_id: YouTube video ID
        languages: List of language codes to try (e.g., ['en', 'es'])
                  If None, tries English first, then any available

    Returns:
        Dict with 'text', 'language', and 'segments' keys
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise ImportError(
            "youtube-transcript-api is required. Install with:\n"
            "  pip install youtube-transcript-api"
        )

    if languages is None:
        languages = ['en']

    try:
        # Create API instance and fetch transcript
        api = YouTubeTranscriptApi()

        # List available transcripts
        transcript_list = api.list(video_id)

        # Try to find transcript in preferred language
        selected_transcript = None
        for transcript in transcript_list:
            if transcript.language_code in languages:
                selected_transcript = transcript
                break

        # If no preferred language found, use first available
        if selected_transcript is None and transcript_list:
            selected_transcript = transcript_list[0]

        if selected_transcript is None:
            raise Exception(f"No transcripts available for video: {video_id}")

        # Fetch the transcript segments
        segments = api.fetch(video_id, languages=[selected_transcript.language_code])

        # Combine all text
        full_text = ' '.join([segment['text'] for segment in segments])

        return {
            'text': full_text,
            'language': selected_transcript.language_code,
            'is_generated': selected_transcript.is_generated,
            'segments': segments
        }

    except Exception as e:
        raise Exception(
            f"Could not retrieve transcript for video: {video_id}. "
            f"Error: {str(e)}\n"
            f"Note: This may fail in restricted environments (403 errors) "
            f"or if transcripts are disabled for the video."
        )


def transcribe_youtube(url: str, languages: List[str] = None) -> Dict:
    """
    Main function to transcribe a YouTube video.

    Args:
        url: YouTube URL or video ID
        languages: Preferred languages (e.g., ['en', 'es'])

    Returns:
        Dict with transcript data
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    return get_transcript(video_id, languages)


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python youtube_transcriber.py <youtube_url>")
        print("\nExample:")
        print("  python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  python youtube_transcriber.py dQw4w9WgXcQ")
        sys.exit(1)

    url = sys.argv[1]

    try:
        result = transcribe_youtube(url)

        print(f"Video ID: {extract_video_id(url)}")
        print(f"Language: {result['language']}")
        print(f"Type: {'Auto-generated' if result['is_generated'] else 'Manual'}")
        print(f"Segments: {len(result['segments'])}")
        print("\n" + "="*80)
        print("TRANSCRIPT:")
        print("="*80)
        print(result['text'])

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
