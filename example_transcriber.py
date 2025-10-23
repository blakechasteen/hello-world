#!/usr/bin/env python3
"""
Example usage of the YouTube transcriber
"""

from youtube_transcriber import transcribe_youtube

# Example 1: Transcribe with video URL
def example_basic():
    """Basic transcription example"""
    print("Example 1: Basic transcription")
    print("-" * 80)

    # Replace with your video URL
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        result = transcribe_youtube(video_url)

        print(f"Language: {result['language']}")
        print(f"Is generated: {result['is_generated']}")
        print(f"Number of segments: {len(result['segments'])}")
        print(f"\nFirst 500 characters of transcript:")
        print(result['text'][:500] + "...")

    except Exception as e:
        print(f"Error: {e}")


# Example 2: Specify preferred language
def example_with_language():
    """Transcription with language preference"""
    print("\n\nExample 2: With language preference")
    print("-" * 80)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        # Try Spanish first, then English
        result = transcribe_youtube(video_url, languages=['es', 'en'])

        print(f"Language: {result['language']}")
        print(f"First 300 characters:")
        print(result['text'][:300] + "...")

    except Exception as e:
        print(f"Error: {e}")


# Example 3: Process transcript segments
def example_process_segments():
    """Process individual transcript segments"""
    print("\n\nExample 3: Processing segments")
    print("-" * 80)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        result = transcribe_youtube(video_url)

        print(f"Total segments: {len(result['segments'])}")
        print("\nFirst 5 segments with timestamps:")

        for i, segment in enumerate(result['segments'][:5]):
            start = segment['start']
            duration = segment['duration']
            text = segment['text']
            print(f"  [{start:.2f}s - {start+duration:.2f}s] {text}")

    except Exception as e:
        print(f"Error: {e}")


# Example 4: Save transcript to file
def example_save_to_file():
    """Save transcript to a text file"""
    print("\n\nExample 4: Save to file")
    print("-" * 80)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    output_file = "transcript.txt"

    try:
        result = transcribe_youtube(video_url)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Video: {video_url}\n")
            f.write(f"Language: {result['language']}\n")
            f.write(f"Generated: {result['is_generated']}\n")
            f.write("\n" + "="*80 + "\n\n")
            f.write(result['text'])

        print(f"Transcript saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    print("YouTube Transcriber Examples")
    print("="*80)
    print("\nNote: Replace video URLs with actual videos you want to transcribe")
    print("Examples may fail if transcripts are disabled or unavailable\n")

    example_basic()
    example_with_language()
    example_process_segments()
    example_save_to_file()
