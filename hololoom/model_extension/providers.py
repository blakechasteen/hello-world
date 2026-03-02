"""
LLM Provider Adapters for Model Extension SDK

Unified interface for multiple LLM providers:
- Anthropic (Claude)
- OpenAI (GPT-4, GPT-3.5)
- Ollama (local models)
- Google (Gemini)
- Cohere (Command)

Each provider implements the LLMProvider protocol, enabling
seamless switching between providers without code changes.

Example:
    provider = create_provider("anthropic", api_key="...")
    response = await provider.generate("What is Thompson Sampling?")

    # Or with streaming
    async for chunk in provider.stream("Explain recursion"):
        print(chunk, end="")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    AsyncIterator,
    Dict,
    Any,
    Optional,
    List,
    Protocol,
    runtime_checkable,
)
from enum import Enum
import asyncio


class ProviderType(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GOOGLE = "google"
    COHERE = "cohere"


class MessageRole(Enum):
    """Role for chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    system_prompt: Optional[str] = None

    def to_anthropic(self) -> Dict[str, Any]:
        """Convert to Anthropic API format."""
        config = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            config["top_k"] = self.top_k
        if self.stop_sequences:
            config["stop_sequences"] = self.stop_sequences
        return config

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        config = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.stop_sequences:
            config["stop"] = self.stop_sequences
        return config

    def to_ollama(self) -> Dict[str, Any]:
        """Convert to Ollama API format."""
        config = {
            "num_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            config["top_k"] = self.top_k
        if self.stop_sequences:
            config["stop"] = self.stop_sequences
        return config

    def to_google(self) -> Dict[str, Any]:
        """Convert to Google Gemini API format."""
        config = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            config["top_k"] = self.top_k
        if self.stop_sequences:
            config["stop_sequences"] = self.stop_sequences
        return config

    def to_cohere(self) -> Dict[str, Any]:
        """Convert to Cohere API format."""
        config = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "p": self.top_p,
        }
        if self.top_k is not None:
            config["k"] = self.top_k
        if self.stop_sequences:
            config["stop_sequences"] = self.stop_sequences
        if self.presence_penalty != 0.0:
            config["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            config["frequency_penalty"] = self.frequency_penalty
        return config


@dataclass
class Message:
    """Chat message structure."""
    role: str  # "user", "assistant", "system"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", self.usage.get("prompt_tokens", 0))

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", self.usage.get("completion_tokens", 0))

    @property
    def total_tokens(self) -> int:
        # Check for direct total_tokens key first
        if "total_tokens" in self.usage:
            return self.usage["total_tokens"]
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        ...

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response from chat messages."""
        ...

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        ...

    async def stream_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream chat generation."""
        ...

    def is_available(self) -> bool:
        """Check if provider is available."""
        ...


class BaseLLMProvider(ABC):
    """Base class for LLM providers with common functionality."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def provider_name(self) -> str:
        """Return the provider name as string."""
        return self.provider_type.value

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response from chat messages."""
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream chat generation."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        ...

    def _ensure_config(self, config: Optional[GenerationConfig]) -> GenerationConfig:
        """Ensure config exists."""
        return config or GenerationConfig()


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        super().__init__(model, api_key, base_url, timeout)
        self._anthropic = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._anthropic is None:
            try:
                import anthropic
                self._anthropic = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._anthropic

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        messages = [Message(role="user", content=prompt)]
        return await self._generate_impl(messages, config, system_prompt)

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response from chat messages."""
        # Extract system prompt if first message is system
        system_prompt = None
        chat_messages = messages

        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            chat_messages = messages[1:]

        return await self._generate_impl(chat_messages, config, system_prompt)

    async def _generate_impl(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig],
        system_prompt: Optional[str],
    ) -> GenerationResult:
        """Implementation of generation."""
        config = self._ensure_config(config)
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            **config.to_anthropic(),
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = await client.messages.create(**kwargs)

        return GenerationResult(
            text=response.content[0].text,
            model=response.model,
            provider="anthropic",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            metadata={"id": response.id},
        )

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        messages = [Message(role="user", content=prompt)]
        async for chunk in self._stream_impl(messages, config, system_prompt):
            yield chunk

    async def stream_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream chat generation."""
        system_prompt = None
        chat_messages = messages

        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            chat_messages = messages[1:]

        async for chunk in self._stream_impl(chat_messages, config, system_prompt):
            yield chunk

    async def _stream_impl(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig],
        system_prompt: Optional[str],
    ) -> AsyncIterator[str]:
        """Implementation of streaming."""
        config = self._ensure_config(config)
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            **config.to_anthropic(),
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        try:
            import anthropic
            return self.api_key is not None or anthropic.api_key is not None
        except ImportError:
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    DEFAULT_MODEL = "gpt-4-turbo-preview"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        super().__init__(model, api_key, base_url, timeout)
        self._openai = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._openai is None:
            try:
                from openai import AsyncOpenAI
                self._openai = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
        return self._openai

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return await self.chat(messages, config)

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response from chat messages."""
        config = self._ensure_config(config)
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            **config.to_openai(),
        )

        choice = response.choices[0]

        return GenerationResult(
            text=choice.message.content or "",
            model=response.model,
            provider="openai",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason,
            metadata={"id": response.id},
        )

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        async for chunk in self.stream_chat(messages, config):
            yield chunk

    async def stream_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream chat generation."""
        config = self._ensure_config(config)
        client = self._get_client()

        stream = await client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            stream=True,
            **config.to_openai(),
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            import openai
            return self.api_key is not None
        except ImportError:
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider."""

    DEFAULT_MODEL = "llama3.2:3b"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,  # Not used, for interface consistency
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(
            model,
            api_key,
            base_url or self.DEFAULT_BASE_URL,
            timeout,
        )
        self._ollama = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    def _get_client(self):
        """Get or create Ollama client."""
        if self._ollama is None:
            try:
                import ollama
                self._ollama = ollama.AsyncClient(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "ollama package not installed. "
                    "Install with: pip install ollama"
                )
        return self._ollama

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        config = self._ensure_config(config)
        client = self._get_client()

        options = config.to_ollama()

        response = await client.generate(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
            options=options,
        )

        return GenerationResult(
            text=response.get("response", ""),
            model=self.model,
            provider="ollama",
            usage={
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            },
            finish_reason="stop" if response.get("done") else None,
            metadata={
                "total_duration": response.get("total_duration"),
                "load_duration": response.get("load_duration"),
            },
        )

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response from chat messages."""
        config = self._ensure_config(config)
        client = self._get_client()

        options = config.to_ollama()

        response = await client.chat(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            options=options,
        )

        return GenerationResult(
            text=response.get("message", {}).get("content", ""),
            model=self.model,
            provider="ollama",
            usage={
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            },
            finish_reason="stop" if response.get("done") else None,
            metadata={
                "total_duration": response.get("total_duration"),
            },
        )

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        config = self._ensure_config(config)
        client = self._get_client()

        options = config.to_ollama()

        async for chunk in await client.generate(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
            options=options,
            stream=True,
        ):
            if chunk.get("response"):
                yield chunk["response"]

    async def stream_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream chat generation."""
        config = self._ensure_config(config)
        client = self._get_client()

        options = config.to_ollama()

        async for chunk in await client.chat(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            options=options,
            stream=True,
        ):
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import ollama
            import httpx
            # Try to connect to Ollama
            try:
                response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
                return response.status_code == 200
            except Exception:
                return False
        except ImportError:
            return False


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider."""

    DEFAULT_MODEL = "gemini-1.5-pro"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        super().__init__(model, api_key, base_url, timeout)
        self._genai = None
        self._model_instance = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    def _get_model(self):
        """Get or create Gemini model."""
        if self._model_instance is None:
            try:
                import google.generativeai as genai

                if self.api_key:
                    genai.configure(api_key=self.api_key)

                self._genai = genai
                self._model_instance = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._model_instance

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        config = self._ensure_config(config)
        model = self._get_model()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Run in executor since google-generativeai is sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                full_prompt,
                generation_config=config.to_google(),
            ),
        )

        return GenerationResult(
            text=response.text,
            model=self.model,
            provider="google",
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            },
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
        )

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response from chat messages."""
        config = self._ensure_config(config)
        model = self._get_model()

        # Convert messages to Gemini format
        history = []
        for msg in messages[:-1]:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})

        loop = asyncio.get_event_loop()

        # Start chat with history
        chat = model.start_chat(history=history)

        # Send last message
        last_message = messages[-1].content if messages else ""
        response = await loop.run_in_executor(
            None,
            lambda: chat.send_message(
                last_message,
                generation_config=config.to_google(),
            ),
        )

        return GenerationResult(
            text=response.text,
            model=self.model,
            provider="google",
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            },
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
        )

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        config = self._ensure_config(config)
        model = self._get_model()

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        loop = asyncio.get_event_loop()

        # Get streaming response
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                full_prompt,
                generation_config=config.to_google(),
                stream=True,
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def stream_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream chat generation."""
        # For simplicity, use non-streaming chat and yield result
        result = await self.chat(messages, config)
        yield result.text

    def is_available(self) -> bool:
        """Check if Google Gemini is available."""
        try:
            import google.generativeai
            return self.api_key is not None
        except ImportError:
            return False


class CohereProvider(BaseLLMProvider):
    """Cohere Command provider."""

    DEFAULT_MODEL = "command-r-plus"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        super().__init__(model, api_key, base_url, timeout)
        self._cohere = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.COHERE

    def _get_client(self):
        """Get or create Cohere client."""
        if self._cohere is None:
            try:
                import cohere
                self._cohere = cohere.AsyncClientV2(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "cohere package not installed. "
                    "Install with: pip install cohere"
                )
        return self._cohere

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return await self.chat(messages, config)

    async def chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate response from chat messages."""
        config = self._ensure_config(config)
        client = self._get_client()

        # Convert messages to Cohere format
        cohere_messages = []
        for msg in messages:
            role = msg.role
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "assistant"
            else:
                role = "user"
            cohere_messages.append({"role": role, "content": msg.content})

        response = await client.chat(
            model=self.model,
            messages=cohere_messages,
            **config.to_cohere(),
        )

        return GenerationResult(
            text=response.message.content[0].text if response.message.content else "",
            model=self.model,
            provider="cohere",
            usage={
                "input_tokens": response.usage.tokens.input_tokens if hasattr(response, 'usage') else 0,
                "output_tokens": response.usage.tokens.output_tokens if hasattr(response, 'usage') else 0,
            },
            finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else None,
            metadata={"id": response.id if hasattr(response, 'id') else None},
        )

    async def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        async for chunk in self.stream_chat(messages, config):
            yield chunk

    async def stream_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream chat generation."""
        config = self._ensure_config(config)
        client = self._get_client()

        cohere_messages = []
        for msg in messages:
            role = msg.role
            if role == "system":
                role = "system"
            elif role == "assistant":
                role = "assistant"
            else:
                role = "user"
            cohere_messages.append({"role": role, "content": msg.content})

        async for event in client.chat_stream(
            model=self.model,
            messages=cohere_messages,
            **config.to_cohere(),
        ):
            if hasattr(event, 'text') and event.text:
                yield event.text

    def is_available(self) -> bool:
        """Check if Cohere is available."""
        try:
            import cohere
            return self.api_key is not None
        except ImportError:
            return False


def create_provider(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> BaseLLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider: Provider name ("anthropic", "openai", "ollama", "google", "cohere")
        model: Model name (uses provider default if not specified)
        api_key: API key for the provider
        base_url: Base URL for API (optional)
        timeout: Request timeout in seconds

    Returns:
        LLM provider instance

    Example:
        >>> provider = create_provider("anthropic", api_key="sk-...")
        >>> result = await provider.generate("What is AI?")
    """
    provider_lower = provider.lower()

    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,  # Alias
        "cohere": CohereProvider,
    }

    if provider_lower not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {list(providers.keys())}"
        )

    provider_class = providers[provider_lower]

    kwargs = {"api_key": api_key, "base_url": base_url, "timeout": timeout}
    if model:
        kwargs["model"] = model

    return provider_class(**kwargs)


def get_best_available_provider(
    api_keys: Optional[Dict[str, str]] = None,
) -> Optional[BaseLLMProvider]:
    """
    Get the best available provider based on what's installed and configured.

    Priority: Anthropic > OpenAI > Google > Cohere > Ollama (local)

    Args:
        api_keys: Dict of provider -> api_key mappings

    Returns:
        Best available provider or None if none available
    """
    api_keys = api_keys or {}

    # Try providers in priority order
    priority = [
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("google", GoogleProvider),
        ("cohere", CohereProvider),
        ("ollama", OllamaProvider),
    ]

    for name, provider_class in priority:
        try:
            provider = provider_class(api_key=api_keys.get(name))
            if provider.is_available():
                return provider
        except Exception:
            continue

    return None
