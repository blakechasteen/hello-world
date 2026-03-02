"""
Docker Skill - Container orchestration and management
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from hololoom.skills.base import (BaseSkill, SkillInput, SkillOutput, SkillMetadata,
                                   SkillCategory, SkillStatus)


class DockerSkill(BaseSkill):
    """Docker container orchestration skill."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="docker",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.INFRASTRUCTURE,
            tags=["docker", "containers", "devops", "orchestration"],
            description_short="Container orchestration and management via Docker",
            description_long="Wrap Docker CLI for container management, image building, compose orchestration, and health monitoring.",
            requires_code_execution=True,
            requires_filesystem_read=True,
            external_dependencies=["docker", "docker-compose"],
            expected_latency_ms="500-10000ms",
            token_usage="200-800 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["ps", "build", "run", "stop", "logs", "compose_up", "compose_down", "inspect", "prune"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()
        try:
            result = await self._dispatch(skill_input.operation, skill_input.parameters)
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Docker operation '{skill_input.operation}' completed",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"output": result}
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Docker operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _dispatch(self, operation: str, parameters: Dict[str, Any]):
        if operation == "ps":
            return await self._ps(parameters)
        elif operation == "build":
            return await self._build(parameters)
        elif operation == "run":
            return await self._run(parameters)
        elif operation == "stop":
            return await self._stop(parameters)
        elif operation == "logs":
            return await self._logs(parameters)
        elif operation == "compose_up":
            return await self._compose_up(parameters)
        elif operation == "compose_down":
            return await self._compose_down(parameters)
        elif operation == "inspect":
            return await self._inspect(parameters)
        elif operation == "prune":
            return await self._prune(parameters)

    async def _run_docker(self, args: List[str]) -> str:
        cmd = ["docker"] + args
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Docker command failed: {stderr.decode()}")
        return stdout.decode()

    async def _ps(self, params):
        all_containers = params.get("all", False)
        args = ["ps", "--format", "json"]
        if all_containers:
            args.append("-a")
        output = await self._run_docker(args)
        return [json.loads(line) for line in output.strip().split('\n') if line]

    async def _build(self, params):
        dockerfile = params.get("dockerfile", "Dockerfile")
        tag = params["tag"]
        context = params.get("context", ".")
        await self._run_docker(["build", "-f", dockerfile, "-t", tag, context])
        return {"tag": tag, "message": f"Image '{tag}' built successfully"}

    async def _run(self, params):
        image = params["image"]
        name = params.get("name")
        detach = params.get("detach", True)
        ports = params.get("ports", [])
        env = params.get("env", {})
        args = ["run"]
        if detach:
            args.append("-d")
        if name:
            args.extend(["--name", name])
        for port_mapping in ports:
            args.extend(["-p", port_mapping])
        for key, val in env.items():
            args.extend(["-e", f"{key}={val}"])
        args.append(image)
        container_id = (await self._run_docker(args)).strip()
        return {"container_id": container_id, "image": image, "name": name}

    async def _stop(self, params):
        container = params["container"]
        await self._run_docker(["stop", container])
        return {"container": container, "message": f"Container '{container}' stopped"}

    async def _logs(self, params):
        container = params["container"]
        tail = params.get("tail", 100)
        logs = await self._run_docker(["logs", "--tail", str(tail), container])
        return {"container": container, "logs": logs}

    async def _compose_up(self, params):
        file = params.get("file", "docker-compose.yml")
        detach = params.get("detach", True)
        args = ["docker-compose", "-f", file, "up"]
        if detach:
            args.append("-d")
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        return {"file": file, "message": "Services started", "output": stdout.decode()}

    async def _compose_down(self, params):
        file = params.get("file", "docker-compose.yml")
        proc = await asyncio.create_subprocess_exec("docker-compose", "-f", file, "down", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        return {"file": file, "message": "Services stopped"}

    async def _inspect(self, params):
        container = params["container"]
        output = await self._run_docker(["inspect", container])
        return json.loads(output)[0]

    async def _prune(self, params):
        prune_type = params.get("type", "system")  # system, images, containers, volumes
        if prune_type == "system":
            await self._run_docker(["system", "prune", "-f"])
        elif prune_type == "images":
            await self._run_docker(["image", "prune", "-f"])
        elif prune_type == "containers":
            await self._run_docker(["container", "prune", "-f"])
        elif prune_type == "volumes":
            await self._run_docker(["volume", "prune", "-f"])
        return {"type": prune_type, "message": f"{prune_type} pruned"}


from hololoom.skills.base import register_skill
register_skill(DockerSkill())
