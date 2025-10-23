# YouTube Caption Transcriber

A simple Python tool to extract captions/transcripts from YouTube videos.

## Installation

```bash
pip install -r requirements-transcriber.txt
```

Or install directly:
```bash
pip install youtube-transcript-api
```

## Usage

### Command Line

```bash
# Using full URL
python youtube_transcriber.py https://www.youtube.com/watch?v=VIDEO_ID

# Using video ID only
python youtube_transcriber.py VIDEO_ID

# Using short URL
python youtube_transcriber.py https://youtu.be/VIDEO_ID
```

### As a Python Module

```python
from youtube_transcriber import transcribe_youtube

# Get transcript
result = transcribe_youtube('https://www.youtube.com/watch?v=dQw4w9WgXcQ')

print(result['text'])           # Full transcript text
print(result['language'])       # Language code (e.g., 'en')
print(result['is_generated'])   # True if auto-generated
print(result['segments'])       # List of timed segments

# Specify preferred languages
result = transcribe_youtube('VIDEO_ID', languages=['es', 'en'])
```

## Features

- Extracts both manual and auto-generated captions
- Supports multiple YouTube URL formats
- Language preference support
- Returns both full text and timed segments
- Simple CLI and Python API

## Output Format

The transcriber returns a dictionary with:
- `text`: Complete transcript as a single string
- `language`: Language code of the transcript
- `is_generated`: Boolean indicating if auto-generated
- `segments`: List of dicts with 'text', 'start', 'duration' for each caption segment

## Example

```bash
$ python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

Video ID: dQw4w9WgXcQ
Language: en
Type: Auto-generated
Segments: 42

================================================================================
TRANSCRIPT:
================================================================================
[Full transcript text here...]
```

## Additional Examples

Check out `example_transcriber.py` for more usage examples including:
- Basic transcription
- Language preferences
- Processing individual segments
- Saving transcripts to files

Run examples:
```bash
python example_transcriber.py
```

## Troubleshooting

### 403 Forbidden Errors

If you encounter "403 Client Error: Forbidden" messages:
- This may occur in restricted server environments or when YouTube blocks certain IPs
- Try running from a different network or local machine
- Use a VPN if necessary
- Ensure you're not making too many requests in a short time

### No Transcripts Available

If you get "No transcripts available" errors:
- The video may have transcripts disabled by the creator
- The video may be age-restricted or private
- Try a different video to verify the tool is working

### Rate Limiting

If making many requests, add delays between calls:
```python
import time
from youtube_transcriber import transcribe_youtube

for video_id in video_ids:
    result = transcribe_youtube(video_id)
    # Process result...
    time.sleep(1)  # Wait 1 second between requests
```

## Notes

- Respects YouTube's transcript availability settings
- Only works for videos with captions/transcripts enabled
- Does not bypass any YouTube restrictions
- For educational and accessibility purposes
