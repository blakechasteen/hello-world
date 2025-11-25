# Skill: FFmpeg

## Metadata

- **Name**: `ffmpeg`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `creative`
- **Tags**: `video, audio, processing, encoding, multimedia`

## Description

**Short Description**:
Video and audio processing with FFmpeg for transcoding, editing, and format conversion.

**Detailed Description**:
The FFmpeg skill provides comprehensive multimedia processing capabilities using the industry-standard FFmpeg library. Transcode videos between formats, extract audio streams, trim/concat clips, apply filters (scale, crop, rotate, watermark), generate thumbnails, and optimize for web streaming. Supports all major codecs (H.264, H.265, VP9, AV1) and containers (MP4, WebM, MKV, MOV). Ideal for video pipelines, content processing, streaming preparation, and automated video production.

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read)
- [x] File system access (write)
- [x] Code execution (bash)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `ffmpeg` binary (core multimedia processor)
- `ffprobe` (media file analyzer)
- Optional: Hardware acceleration (NVIDIA NVENC, Intel QSV, AMD VCE)

**HoloLoom Integration**: Integrates with content processing pipelines, video streaming preparation, thumbnail generation, and multimedia workflows.

## Input Schema

```json
{
  "operation": "string - transcode|extract_audio|trim|concat|thumbnail|apply_filter",
  "parameters": {
    "input": "string (required) - Input file path",
    "output": "string (required) - Output file path",
    "codec": "string (optional for transcode) - Video codec: h264|h265|vp9|av1",
    "audio_codec": "string (optional) - Audio codec: aac|mp3|opus|vorbis",
    "format": "string (optional) - Container format: mp4|webm|mkv|mov",
    "crf": "number (optional) - Constant Rate Factor (0-51, lower=better quality)",
    "preset": "string (optional) - Encoding preset: ultrafast|fast|medium|slow|veryslow",
    "start_time": "string (required for trim) - Start timestamp (HH:MM:SS)",
    "duration": "string (required for trim) - Duration (HH:MM:SS)",
    "inputs": "array (required for concat) - List of input files to concatenate",
    "timestamp": "string (optional for thumbnail) - Thumbnail timestamp (default: 00:00:05)",
    "filter": "string (required for apply_filter) - FFmpeg filter: scale|crop|rotate|watermark",
    "filter_params": "object (required for apply_filter) - Filter-specific parameters",
    "hw_accel": "string (optional) - Hardware acceleration: nvenc|qsv|vaapi"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Processing details",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "input": "string - Input file path",
    "output": "string - Output file path",
    "input_duration_seconds": "number - Input duration",
    "output_duration_seconds": "number - Output duration",
    "input_size_mb": "number - Input file size",
    "output_size_mb": "number - Output file size",
    "compression_ratio": "number - Size reduction ratio",
    "codec": "string - Video codec used",
    "audio_codec": "string - Audio codec used",
    "resolution": "string - Video resolution (1920x1080)",
    "fps": "number - Frames per second",
    "bitrate_kbps": "number - Output bitrate"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Transcode to Web-Optimized MP4

**Input**:
```json
{
  "operation": "transcode",
  "parameters": {
    "input": "raw_footage/interview.mov",
    "output": "web/interview_optimized.mp4",
    "codec": "h264",
    "audio_codec": "aac",
    "format": "mp4",
    "crf": 23,
    "preset": "medium",
    "resolution": "1920x1080"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "transcode",
    "input": "raw_footage/interview.mov",
    "output": "web/interview_optimized.mp4",
    "input_duration_seconds": 180.5,
    "output_duration_seconds": 180.5,
    "input_size_mb": 850,
    "output_size_mb": 45,
    "compression_ratio": 18.9,
    "codec": "h264",
    "audio_codec": "aac",
    "resolution": "1920x1080",
    "fps": 30,
    "bitrate_kbps": 2000
  },
  "message": "Video transcoded: 850MB -> 45MB (18.9x compression)",
  "execution_time_ms": 45000
}
```

**Explanation**: Converts high-quality MOV to web-optimized MP4 with H.264 encoding. CRF=23 balances quality and file size.

### Example 2: Extract Audio Track

**Input**:
```json
{
  "operation": "extract_audio",
  "parameters": {
    "input": "videos/podcast_episode_05.mp4",
    "output": "audio/podcast_episode_05.mp3",
    "audio_codec": "mp3",
    "bitrate": "192k"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "extract_audio",
    "input": "videos/podcast_episode_05.mp4",
    "output": "audio/podcast_episode_05.mp3",
    "duration_seconds": 3600,
    "audio_codec": "mp3",
    "bitrate_kbps": 192,
    "sample_rate_hz": 44100,
    "channels": 2,
    "output_size_mb": 82
  },
  "message": "Audio extracted: 82MB MP3 (192kbps)",
  "execution_time_ms": 12000
}
```

**Explanation**: Extracts audio stream from video file for podcast distribution. 192kbps MP3 provides good quality at reasonable file size.

### Example 3: Trim Video Clip

**Input**:
```json
{
  "operation": "trim",
  "parameters": {
    "input": "recordings/webinar_full.mp4",
    "output": "clips/highlight_reel.mp4",
    "start_time": "00:15:30",
    "duration": "00:02:45"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "trim",
    "input": "recordings/webinar_full.mp4",
    "output": "clips/highlight_reel.mp4",
    "start_time": "00:15:30",
    "duration_seconds": 165,
    "input_duration_seconds": 3600,
    "output_size_mb": 25,
    "success": true
  },
  "message": "Video trimmed: 2m 45s clip extracted",
  "execution_time_ms": 3500
}
```

**Explanation**: Extracts 2-minute 45-second highlight clip from hour-long webinar. Fast operation with stream copy (no re-encoding).

### Example 4: Concatenate Multiple Clips

**Input**:
```json
{
  "operation": "concat",
  "parameters": {
    "inputs": [
      "clips/intro.mp4",
      "clips/main_content.mp4",
      "clips/outro.mp4"
    ],
    "output": "final/complete_video.mp4"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "concat",
    "inputs": ["clips/intro.mp4", "clips/main_content.mp4", "clips/outro.mp4"],
    "output": "final/complete_video.mp4",
    "input_count": 3,
    "total_duration_seconds": 420,
    "output_size_mb": 68,
    "success": true
  },
  "message": "3 clips concatenated: 7m 0s total duration",
  "execution_time_ms": 8500
}
```

**Explanation**: Combines intro, main content, and outro into single video. Seamless concatenation without re-encoding (if formats match).

### Example 5: Generate Thumbnail

**Input**:
```json
{
  "operation": "thumbnail",
  "parameters": {
    "input": "videos/tutorial_pt1.mp4",
    "output": "thumbnails/tutorial_pt1_thumb.jpg",
    "timestamp": "00:00:10",
    "width": 1280,
    "height": 720
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "thumbnail",
    "input": "videos/tutorial_pt1.mp4",
    "output": "thumbnails/tutorial_pt1_thumb.jpg",
    "timestamp": "00:00:10",
    "width": 1280,
    "height": 720,
    "format": "jpeg",
    "size_kb": 185
  },
  "message": "Thumbnail generated at 00:00:10",
  "execution_time_ms": 800
}
```

**Explanation**: Extracts single frame at 10-second mark as thumbnail. Useful for video previews and content management.

## Testing Checklist

- [x] **Functionality**: All 6 operations execute correctly
- [x] **Error Handling**: Graceful handling of corrupt videos, unsupported codecs
- [x] **Security**: No command injection, safe file path handling
- [x] **Performance**: Operations complete within expected time
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: FFmpeg binary documented
- [x] **Edge Cases**: Handles corrupt files, unusual resolutions, long videos
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom video processing pipelines

## Security Considerations

**Potential Risks**:
- **Command Injection**: File paths could contain shell commands -> Validate and sanitize all inputs
- **Resource Exhaustion**: Large video processing -> Implement timeouts and memory limits
- **Path Traversal**: Output paths could escape workspace -> Validate output directories

**Data Privacy**:
- [x] Does not upload videos to external servers
- [x] Does not log video content or metadata
- [x] Does not access files outside designated directories

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] File operations restricted to input/output directories
- [x] Timeouts prevent indefinite processing

## Performance Characteristics

- **Expected Latency**: 1000-300000ms (1s-5min depending on file size and operation)
- **Token Usage**: 100-1000 tokens per execution
- **Resource Requirements**: CPU/GPU, sufficient disk space for output
- **Scalability**: Limited by hardware and concurrent operations

**Operation-Specific Latencies**:
- `transcode`: 10000-300000ms (depends on file size, codec, resolution)
- `extract_audio`: 2000-30000ms (faster than transcode)
- `trim`: 1000-10000ms (fast with stream copy)
- `concat`: 2000-20000ms (depends on number and size of inputs)
- `thumbnail`: 500-2000ms (very fast, single frame)
- `apply_filter`: 5000-60000ms (depends on filter complexity)

## License

MIT License

## Related Documentation

- **FFmpeg Docs**: [ffmpeg.org/documentation.html](https://ffmpeg.org/documentation.html)
- **FFmpeg Filters**: [ffmpeg.org/ffmpeg-filters.html](https://ffmpeg.org/ffmpeg-filters.html)
- **Codec Guide**: [trac.ffmpeg.org/wiki/Encode](https://trac.ffmpeg.org/wiki/Encode)
- **HoloLoom Creative Skills**: [../README.md](../README.md)
