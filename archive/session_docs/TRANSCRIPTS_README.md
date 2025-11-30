# YouTube Transcripts Saver

Easy-to-use script that automatically saves YouTube transcripts to organized files.

## Quick Start

```bash
# Automatically saves to ./transcripts/ folder
python transcribe_and_save.py "YOUR_VIDEO_URL"
```

## Features

✅ **Auto-saves** to `./transcripts/` folder (creates if needed)
✅ **Multiple formats**: TXT (formatted), JSON (structured data)
✅ **Timestamped filenames**: `VIDEO_ID_20251023_143022.txt`
✅ **Full transcript + segments**: Both continuous text and timestamped segments
✅ **Works with**: Regular videos, Shorts, all URL formats

## Usage Examples

### Basic - Auto-save as TXT

```bash
python transcribe_and_save.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Output: `./transcripts/VIDEO_ID_20251023_143022.txt`

### Save as JSON

```bash
python transcribe_and_save.py "VIDEO_URL" --format json
```

### Save Both Formats

```bash
python transcribe_and_save.py "VIDEO_URL" --format both
```

### Custom Output Directory

```bash
python transcribe_and_save.py "VIDEO_URL" --output-dir "C:\Users\blake\Documents\MyTranscripts"
```

### Just Display (No Save)

```bash
python transcribe_and_save.py "VIDEO_URL" --no-save
```

## Output Formats

### TXT Format (Human-Readable)

```
================================================================================
YouTube Transcript
================================================================================

Video ID: dQw4w9WgXcQ
URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
Language: en
Type: Auto-generated
Duration: 212.5 seconds
Segments: 42
Generated: 2025-10-23 14:30:22

================================================================================
FULL TRANSCRIPT
================================================================================

[Full continuous text here...]

================================================================================
TIMESTAMPED SEGMENTS
================================================================================

[0.0s]     First segment text
[3.5s]     Second segment text
...
```

### JSON Format (Machine-Readable)

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "language": "en",
  "is_generated": true,
  "duration": 212.5,
  "full_text": "Complete transcript...",
  "segments": [
    {"text": "First segment", "start": 0.0, "duration": 3.5},
    {"text": "Second segment", "start": 3.5, "duration": 4.2}
  ],
  "generated_at": "2025-10-23T14:30:22"
}
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir DIR` | Save location | `./transcripts` |
| `--format FORMAT` | Output format: `txt`, `json`, or `both` | `txt` |
| `--no-save` | Display only, don't save | Save enabled |
| `--help` | Show help | - |

## File Naming

Files are automatically named with:
- **Video ID**: Unique YouTube identifier
- **Timestamp**: When transcript was generated
- **Format**: `.txt` or `.json`

Example: `4g251atrdX8_20251023_143022.txt`

## Tips

### Batch Processing Multiple Videos

Create a list of URLs and process them:

```bash
# videos.txt contains one URL per line
for url in $(cat videos.txt); do
    python transcribe_and_save.py "$url"
    sleep 2  # Be nice to YouTube
done
```

### Organize by Topic

```bash
python transcribe_and_save.py "VIDEO_URL" --output-dir "./transcripts/python_tutorials"
python transcribe_and_save.py "VIDEO_URL" --output-dir "./transcripts/music_theory"
```

### Create a Shortcut (Windows)

Create a batch file `transcribe.bat`:

```batch
@echo off
python "C:\Users\blake\Documents\mythRL\hello-world\transcribe_and_save.py" %*
pause
```

Then drag-and-drop a YouTube URL onto it!

### PowerShell Function

Add to your PowerShell profile:

```powershell
function ytran {
    param([string]$url)
    python "C:\path\to\transcribe_and_save.py" $url
}
```

Usage: `ytran "VIDEO_URL"`

## Troubleshooting

### "No transcripts available"
- Video doesn't have captions enabled
- Try a different video
- Some Shorts don't have captions

### "Permission denied" when saving
- Check if you have write access to the output directory
- Try a different `--output-dir` location

### "Module not found: youtube_transcript_api"
```bash
pip install youtube-transcript-api
```

## Other Scripts

- **`youtube_transcript_tool.py`**: Simple script with interactive save prompt
- **`transcribe_and_save.py`**: Auto-saves with organized output (THIS ONE)
- **HoloLoom integration**: For advanced use with the HoloLoom system

## Default Output Structure

```
hello-world/
├── transcribe_and_save.py
└── transcripts/               ← Created automatically
    ├── VIDEO_ID1_20251023_143022.txt
    ├── VIDEO_ID1_20251023_143022.json
    ├── VIDEO_ID2_20251023_144530.txt
    └── ...
```

Happy transcribing! 🎥→📝
