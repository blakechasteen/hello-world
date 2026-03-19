"""FFmpeg Skill - Video/audio processing"""

import asyncio
import os
import time
from typing import Any

from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
)


class FFmpegSkill(BaseSkill):
    """Video/audio processing with FFmpeg."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ffmpeg",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.CREATIVE,
            tags=["video", "audio", "processing", "conversion", "media"],
            description_short="Video/audio processing with FFmpeg",
            description_long="Convert formats, extract audio, resize video, create thumbnails, trim clips, add watermarks, and process media with FFmpeg.",
            requires_code_execution=True,
            requires_filesystem_read=True,
            requires_filesystem_write=True,
            external_dependencies=["ffmpeg"],
            expected_latency_ms="1000-120000ms",
            token_usage="100-1000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["convert", "extract_audio", "thumbnail", "trim", "resize", "watermark", "concat"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()
        try:
            # Dispatch to operation handler
            if skill_input.operation == "convert":
                result = await self._convert(skill_input.parameters)
            elif skill_input.operation == "extract_audio":
                result = await self._extract_audio(skill_input.parameters)
            elif skill_input.operation == "thumbnail":
                result = await self._thumbnail(skill_input.parameters)
            elif skill_input.operation == "trim":
                result = await self._trim(skill_input.parameters)
            elif skill_input.operation == "resize":
                result = await self._resize(skill_input.parameters)
            elif skill_input.operation == "watermark":
                result = await self._watermark(skill_input.parameters)
            elif skill_input.operation == "concat":
                result = await self._concat(skill_input.parameters)
            else:
                raise ValueError(f"Unknown operation: {skill_input.operation}")

            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"FFmpeg '{skill_input.operation}' completed",
                execution_time_ms=(time.time() - start_time) * 1000,
                details=result
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"FFmpeg operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _run_ffmpeg(self, args: list[str]) -> str:
        """Run ffmpeg command."""
        cmd = ["ffmpeg", "-y"] + args  # -y to overwrite output files
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"FFmpeg command failed: {error_msg}")

        return stdout.decode('utf-8', errors='ignore')

    async def _get_file_info(self, file_path: str) -> dict[str, Any]:
        """Get video/audio file information using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()

        if proc.returncode == 0:
            import json
            return json.loads(stdout.decode('utf-8'))
        return {}

    async def _convert(self, params: dict[str, Any]):
        """Convert video/audio format."""
        input_file = params["input"]
        output_file = params["output"]
        codec = params.get("codec", "copy")
        audio_codec = params.get("audio_codec", "copy")
        crf = params.get("crf", 23)  # Quality (lower = better, 18-28 reasonable)
        preset = params.get("preset", "medium")  # Speed vs compression

        args = [
            "-i", input_file,
            "-c:v", codec,
            "-c:a", audio_codec,
            "-crf", str(crf),
            "-preset", preset,
            output_file
        ]

        await self._run_ffmpeg(args)

        # Get file sizes
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB

        return {
            "operation": "convert",
            "input": input_file,
            "output": output_file,
            "input_size_mb": round(input_size, 2),
            "output_size_mb": round(output_size, 2),
            "compression_ratio": round(input_size / output_size, 2) if output_size > 0 else 0,
            "codec": codec,
            "preset": preset
        }

    async def _extract_audio(self, params: dict[str, Any]):
        """Extract audio track from video."""
        input_file = params["input"]
        output_file = params["output"]
        audio_codec = params.get("codec", "mp3")
        bitrate = params.get("bitrate", "192k")

        args = [
            "-i", input_file,
            "-vn",  # No video
            "-c:a", audio_codec,
            "-b:a", bitrate,
            output_file
        ]

        await self._run_ffmpeg(args)

        output_size = os.path.getsize(output_file) / (1024 * 1024)

        return {
            "operation": "extract_audio",
            "input": input_file,
            "output": output_file,
            "audio_codec": audio_codec,
            "bitrate": bitrate,
            "size_mb": round(output_size, 2)
        }

    async def _thumbnail(self, params: dict[str, Any]):
        """Extract thumbnail/frame from video."""
        input_file = params["input"]
        output_file = params["output"]
        timestamp = params.get("timestamp", "00:00:01")  # HH:MM:SS
        width = params.get("width", -1)
        height = params.get("height", -1)

        args = [
            "-i", input_file,
            "-ss", timestamp,
            "-vframes", "1",
            "-vf", f"scale={width}:{height}",
            output_file
        ]

        await self._run_ffmpeg(args)

        return {
            "operation": "thumbnail",
            "input": input_file,
            "output": output_file,
            "timestamp": timestamp
        }

    async def _trim(self, params: dict[str, Any]):
        """Trim video to specific duration."""
        input_file = params["input"]
        output_file = params["output"]
        start_time = params.get("start", "00:00:00")
        duration = params.get("duration")
        end_time = params.get("end")

        args = ["-i", input_file, "-ss", start_time]

        if duration:
            args.extend(["-t", str(duration)])
        elif end_time:
            args.extend(["-to", end_time])

        args.extend(["-c", "copy", output_file])

        await self._run_ffmpeg(args)

        return {
            "operation": "trim",
            "input": input_file,
            "output": output_file,
            "start": start_time,
            "duration": duration or "to end",
            "end": end_time
        }

    async def _resize(self, params: dict[str, Any]):
        """Resize video resolution."""
        input_file = params["input"]
        output_file = params["output"]
        width = params.get("width", 1920)
        height = params.get("height", 1080)

        args = [
            "-i", input_file,
            "-vf", f"scale={width}:{height}",
            output_file
        ]

        await self._run_ffmpeg(args)

        return {
            "operation": "resize",
            "input": input_file,
            "output": output_file,
            "resolution": f"{width}x{height}"
        }

    async def _watermark(self, params: dict[str, Any]):
        """Add watermark to video."""
        input_file = params["input"]
        watermark_file = params["watermark"]
        output_file = params["output"]
        position = params.get("position", "bottom-right")

        # Map position to FFmpeg overlay coordinates
        position_map = {
            "top-left": "10:10",
            "top-right": "W-w-10:10",
            "bottom-left": "10:H-h-10",
            "bottom-right": "W-w-10:H-h-10",
            "center": "(W-w)/2:(H-h)/2"
        }

        overlay_position = position_map.get(position, "W-w-10:H-h-10")

        args = [
            "-i", input_file,
            "-i", watermark_file,
            "-filter_complex", f"overlay={overlay_position}",
            output_file
        ]

        await self._run_ffmpeg(args)

        return {
            "operation": "watermark",
            "input": input_file,
            "watermark": watermark_file,
            "output": output_file,
            "position": position
        }

    async def _concat(self, params: dict[str, Any]):
        """Concatenate multiple video files."""
        inputs = params["inputs"]
        output_file = params["output"]

        # Create temporary concat file list
        concat_file = "concat_list.txt"
        with open(concat_file, 'w') as f:
            for input_file in inputs:
                f.write(f"file '{input_file}'\n")

        args = [
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_file
        ]

        await self._run_ffmpeg(args)

        # Clean up temp file
        os.remove(concat_file)

        return {
            "operation": "concat",
            "inputs": inputs,
            "output": output_file,
            "input_count": len(inputs)
        }


from hololoom.skills.base import register_skill

register_skill(FFmpegSkill())
