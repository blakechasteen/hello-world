"""
EdWIN Multimodal Extension

Extends EdWIN tutoring with visual learning support:
- Diagrams and illustrations
- Video lessons (YouTube integration)
- Interactive visualizations
- Photo-based learning

Leverages HoloLoom's MultimodalRAG infrastructure.

**Implementation Date**: November 15, 2025
"""

from typing import Optional, List, Union
from pathlib import Path
from dataclasses import dataclass

try:
    from HoloLoom.rag.multimodal_rag import MultimodalRAG, MultimodalRAGResult
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False
    print("⚠️  Multimodal RAG not available - visual features disabled")

try:
    from HoloLoom.spinningWheel.youtube import transcribe_youtube
    HAS_YOUTUBE = True
except ImportError:
    HAS_YOUTUBE = False
    print("⚠️  YouTube support not available")

from .tutoring_engine import EdWINTutor, TutoringResponse
from .curriculum_graph import EdWINKnowledgeGraph
from .student_model import EdWINStudentModel


@dataclass
class MultimodalLearningResource:
    """
    Multimodal learning resource (image, video, interactive demo)
    """
    id: str
    type: str  # "image", "video", "interactive"
    title: str
    description: str
    url: str
    objective_ids: List[str]  # Related curriculum objectives
    grade_level: int
    tags: List[str]

    # Image-specific
    image_path: Optional[str] = None
    caption: Optional[str] = None

    # Video-specific
    video_id: Optional[str] = None  # YouTube video ID
    duration_minutes: Optional[int] = None
    transcript: Optional[str] = None

    # Interactive-specific
    demo_url: Optional[str] = None  # e.g., Desmos, PhET simulation


class EdWINMultimodalTutor(EdWINTutor):
    """
    Enhanced EdWIN tutor with multimodal support

    Features:
    - Visual Q&A: Answer questions about diagrams/images
    - Video lessons: YouTube video integration
    - Interactive demos: Links to Desmos, PhET simulations
    - Photo retrieval: Find relevant diagrams for concepts

    Example:
        >>> tutor = EdWINMultimodalTutor(curriculum_kg)
        >>> await tutor.initialize()
        >>>
        >>> # Visual Q&A
        >>> result = await tutor.answer_with_image(
        ...     question="Explain this diagram",
        ...     image_path="pythagorean_theorem.png",
        ...     student_model=student
        ... )
        >>>
        >>> # Get video lessons
        >>> videos = await tutor.get_video_lessons(
        ...     objective_id="math.geometry.8.pythagorean"
        ... )
    """

    def __init__(
        self,
        curriculum_kg: EdWINKnowledgeGraph,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-3-5-sonnet-20241022",
        enable_caching: bool = True,
        enable_visual_compression: bool = True
    ):
        """
        Initialize multimodal tutor

        Args:
            curriculum_kg: Curriculum knowledge graph
            llm_provider: LLM provider
            llm_model: Model name
            enable_caching: Enable query caching
            enable_visual_compression: Enable visual compression (5-20x token savings)
        """
        super().__init__(
            curriculum_kg=curriculum_kg,
            llm_provider=llm_provider,
            llm_model=llm_model,
            enable_caching=enable_caching
        )

        self.enable_visual_compression = enable_visual_compression

        # Replace SimpleRAG with MultimodalRAG
        if HAS_MULTIMODAL:
            from HoloLoom.config import Config
            config = Config.fused()
            self.multimodal_rag = MultimodalRAG(
                config=config,
                llm_provider=llm_provider,
                llm_model=llm_model,
                enable_caching=enable_caching,
                enable_visual_compression=enable_visual_compression
            )
        else:
            self.multimodal_rag = None

        # Learning resources database (in-memory for now)
        self.resources: List[MultimodalLearningResource] = []
        self._initialize_default_resources()

    async def initialize(self, ingest_curriculum: bool = True):
        """Initialize tutor with multimodal capabilities"""
        await super().initialize(ingest_curriculum=ingest_curriculum)

        if HAS_MULTIMODAL and self.multimodal_rag:
            print("🎨 Multimodal features enabled (images, videos, OCR)")
        else:
            print("⚠️  Multimodal features disabled (install dependencies)")

    async def answer_with_image(
        self,
        question: str,
        image_path: str,
        student_model: EdWINStudentModel,
        mode: str = "verify"
    ) -> TutoringResponse:
        """
        Answer question about an image

        Uses OCR + CLIP to understand image content, then generates explanation.

        Args:
            question: Student's question
            image_path: Path to image file
            student_model: Student profile
            mode: Reasoning mode

        Returns:
            TutoringResponse with visual context
        """
        if not HAS_MULTIMODAL or not self.multimodal_rag:
            return await self.answer_question(question, student_model, mode)

        # Build context
        context = self._build_context(question, student_model)

        # Query with image
        result = await self.multimodal_rag.query_with_image(
            question=f"{context}\n\nStudent Question: {question}",
            image=image_path
        )

        # Convert to TutoringResponse
        response = TutoringResponse(
            question=question,
            answer=result.response,
            confidence=result.confidence,
            sources=result.sources[:5],
            reasoning_mode=mode,
            images=[image_path],
            related_objectives=self._get_related_objectives(question, student_model),
            next_steps=self._get_next_steps(student_model)
        )

        return response

    async def get_visual_resources(
        self,
        objective_id: str,
        resource_type: Optional[str] = None
    ) -> List[MultimodalLearningResource]:
        """
        Get visual learning resources for an objective

        Args:
            objective_id: Learning objective ID
            resource_type: Filter by type ("image", "video", "interactive")

        Returns:
            List of relevant resources
        """
        resources = [
            r for r in self.resources
            if objective_id in r.objective_ids
        ]

        if resource_type:
            resources = [r for r in resources if r.type == resource_type]

        return resources

    async def get_video_lessons(
        self,
        objective_id: str
    ) -> List[MultimodalLearningResource]:
        """Get video lessons for an objective"""
        return await self.get_visual_resources(objective_id, resource_type="video")

    async def get_diagrams(
        self,
        objective_id: str
    ) -> List[MultimodalLearningResource]:
        """Get diagrams/images for an objective"""
        return await self.get_visual_resources(objective_id, resource_type="image")

    async def get_interactive_demos(
        self,
        objective_id: str
    ) -> List[MultimodalLearningResource]:
        """Get interactive demonstrations for an objective"""
        return await self.get_visual_resources(objective_id, resource_type="interactive")

    async def ingest_video_lesson(
        self,
        video_url: str,
        objective_ids: List[str],
        title: str,
        grade_level: int
    ) -> MultimodalLearningResource:
        """
        Ingest YouTube video lesson

        Transcribes video and adds to knowledge base.

        Args:
            video_url: YouTube URL
            objective_ids: Related curriculum objectives
            title: Lesson title
            grade_level: Target grade

        Returns:
            Created learning resource
        """
        if not HAS_YOUTUBE:
            raise ValueError("YouTube support not available - install dependencies")

        # Extract video ID from URL
        video_id = self._extract_youtube_id(video_url)

        # Transcribe video
        print(f"📹 Transcribing video: {title}...")
        shards = await transcribe_youtube(video_id)

        # Ingest transcript into RAG
        if shards:
            transcript = "\n".join(shard.content for shard in shards)
            await self.rag.ingest(f"Video Lesson: {title}\n\n{transcript}")

            # Create resource
            resource = MultimodalLearningResource(
                id=f"video_{video_id}",
                type="video",
                title=title,
                description=f"Video lesson on {title}",
                url=video_url,
                objective_ids=objective_ids,
                grade_level=grade_level,
                tags=["video", "lesson"],
                video_id=video_id,
                duration_minutes=len(shards),  # Approximate
                transcript=transcript
            )

            self.resources.append(resource)
            print(f"✅ Video lesson added: {title}")

            return resource
        else:
            raise ValueError(f"Failed to transcribe video: {video_url}")

    async def ingest_diagram(
        self,
        image_path: str,
        objective_ids: List[str],
        title: str,
        caption: str,
        grade_level: int
    ) -> MultimodalLearningResource:
        """
        Ingest diagram/illustration

        Stores image with CLIP embeddings for retrieval.

        Args:
            image_path: Path to image
            objective_ids: Related objectives
            title: Diagram title
            caption: Description
            grade_level: Target grade

        Returns:
            Created learning resource
        """
        if not HAS_MULTIMODAL or not self.multimodal_rag:
            raise ValueError("Multimodal support not available")

        # Ingest photo into multimodal RAG
        await self.multimodal_rag.ingest_photo(
            image=image_path,
            tags=[f"objective:{obj_id}" for obj_id in objective_ids],
            description=caption
        )

        # Create resource
        resource = MultimodalLearningResource(
            id=f"image_{Path(image_path).stem}",
            type="image",
            title=title,
            description=caption,
            url=image_path,
            objective_ids=objective_ids,
            grade_level=grade_level,
            tags=["diagram", "image"],
            image_path=image_path,
            caption=caption
        )

        self.resources.append(resource)
        print(f"✅ Diagram added: {title}")

        return resource

    def _extract_youtube_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        import re

        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract video ID from: {url}")

    def _initialize_default_resources(self):
        """Initialize with some default educational resources"""

        # Default interactive demos
        default_demos = [
            MultimodalLearningResource(
                id="desmos_graphing",
                type="interactive",
                title="Desmos Graphing Calculator",
                description="Interactive graphing calculator for algebra and calculus",
                url="https://www.desmos.com/calculator",
                objective_ids=["math.algebra.8.linear_equations", "math.algebra.9.quadratic"],
                grade_level=8,
                tags=["math", "algebra", "graphing"],
                demo_url="https://www.desmos.com/calculator"
            ),
            MultimodalLearningResource(
                id="phet_forces",
                type="interactive",
                title="PhET Forces and Motion",
                description="Interactive simulation of forces and motion",
                url="https://phet.colorado.edu/en/simulations/forces-and-motion-basics",
                objective_ids=["science.physical.6.forces"],
                grade_level=6,
                tags=["science", "physics", "forces"],
                demo_url="https://phet.colorado.edu/en/simulations/forces-and-motion-basics"
            ),
        ]

        self.resources.extend(default_demos)


# Helper functions

async def create_multimodal_tutor(
    curriculum_kg: Optional[EdWINKnowledgeGraph] = None,
    llm_provider: str = "anthropic"
) -> EdWINMultimodalTutor:
    """
    Create and initialize multimodal tutoring engine

    Args:
        curriculum_kg: Optional curriculum graph
        llm_provider: LLM provider

    Returns:
        Initialized EdWINMultimodalTutor
    """
    if curriculum_kg is None:
        from .curriculum_graph import create_curriculum_graph
        curriculum_kg = await create_curriculum_graph()

    tutor = EdWINMultimodalTutor(
        curriculum_kg=curriculum_kg,
        llm_provider=llm_provider
    )

    await tutor.initialize()

    return tutor
