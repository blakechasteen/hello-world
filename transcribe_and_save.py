#!/usr/bin/env python3
"""
YouTube Transcriber with Auto-Save
Automatically saves transcripts to an organized folder structure

Usage:
    python transcribe_and_save.py VIDEO_URL [OPTIONS]

Options:
    --output-dir DIR    Save to specific directory (default: ./transcripts)
    --format FORMAT     Output format: txt, json, both (default: txt)
    --no-save          Don't save automatically, just display
    --chunk-size SECS  Split into chunks of N seconds (optional)
"""

import re
import sys
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs
from pathlib import Path


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url

    parsed = urlparse(url)

    if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        elif parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
        elif parsed.path.startswith('/shorts/'):
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
        transcript_list = api.list(video_id)

        selected_transcript = None
        for transcript in transcript_list:
            if transcript.language_code in languages:
                selected_transcript = transcript
                break

        if selected_transcript is None and transcript_list:
            selected_transcript = transcript_list[0]

        if selected_transcript is None:
            raise Exception(f"No transcripts available for video: {video_id}")

        segments = api.fetch(video_id, languages=[selected_transcript.language_code])

        # Convert segments to dicts if they're objects
        segment_dicts = []
        for seg in segments:
            if isinstance(seg, dict):
                segment_dicts.append(seg)
            else:
                segment_dicts.append({
                    'text': seg.text,
                    'start': seg.start,
                    'duration': seg.duration
                })

        full_text = ' '.join([seg['text'] for seg in segment_dicts])

        duration = 0
        if segment_dicts:
            last = segment_dicts[-1]
            duration = last['start'] + last.get('duration', 0)

        return {
            'text': full_text,
            'language': selected_transcript.language_code,
            'is_generated': selected_transcript.is_generated,
            'segments': segment_dicts,
            'duration': duration,
            'video_id': video_id
        }

    except Exception as e:
        raise Exception(f"Could not retrieve transcript: {e}")


def save_transcript_txt(data: Dict[str, Any], filepath: Path) -> None:
    """Save transcript as formatted text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YouTube Transcript\n")
        f.write("="*80 + "\n\n")

        f.write(f"Video ID: {data['video_id']}\n")
        f.write(f"URL: https://www.youtube.com/watch?v={data['video_id']}\n")
        f.write(f"Language: {data['language']}\n")
        f.write(f"Type: {'Auto-generated' if data['is_generated'] else 'Manual'}\n")
        f.write(f"Duration: {data['duration']:.1f} seconds\n")
        f.write(f"Segments: {len(data['segments'])}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("FULL TRANSCRIPT\n")
        f.write("="*80 + "\n\n")

        f.write(data['text'])

        f.write("\n\n" + "="*80 + "\n")
        f.write("TIMESTAMPED SEGMENTS\n")
        f.write("="*80 + "\n\n")

        for seg in data['segments']:
            timestamp = f"[{seg['start']:.1f}s]"
            f.write(f"{timestamp:<10} {seg['text']}\n")


def save_transcript_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save transcript as JSON file."""
    output = {
        'video_id': data['video_id'],
        'url': f"https://www.youtube.com/watch?v={data['video_id']}",
        'language': data['language'],
        'is_generated': data['is_generated'],
        'duration': data['duration'],
        'full_text': data['text'],
        'segments': data['segments'],
        'generated_at': datetime.now().isoformat()
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    """Main function."""
    # Parse arguments
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print(__doc__)
        sys.exit(0)

    video_url = args[0]
    output_dir = Path('./transcripts')
    output_format = 'txt'
    auto_save = True

    # Parse options
    i = 1
    while i < len(args):
        if args[i] == '--output-dir' and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
        elif args[i] == '--format' and i + 1 < len(args):
            output_format = args[i + 1]
            i += 2
        elif args[i] == '--no-save':
            auto_save = False
            i += 1
        else:
            i += 1

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
        print(f"  Characters: {len(result['text'])}")

        # Preview
        print(f"\n{'='*80}")
        print("TRANSCRIPT PREVIEW (first 500 characters)")
        print(f"{'='*80}\n")
        print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])

        # Save
        if auto_save:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"{video_id}_{timestamp}"

            saved_files = []

            if output_format in ('txt', 'both'):
                txt_path = output_dir / f"{base_filename}.txt"
                save_transcript_txt(result, txt_path)
                saved_files.append(str(txt_path))

            if output_format in ('json', 'both'):
                json_path = output_dir / f"{base_filename}.json"
                save_transcript_json(result, json_path)
                saved_files.append(str(json_path))

            print(f"\n{'='*80}")
            print("✓ SAVED TO:")
            for filepath in saved_files:
                print(f"  {filepath}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print("FULL TRANSCRIPT")
            print(f"{'='*80}\n")
            print(result['text'])
            print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure video has captions enabled")
        print("  - Try a different video")
        print("  - Check internet connection")
        sys.exit(1)


if __name__ == '__main__':
    main()
