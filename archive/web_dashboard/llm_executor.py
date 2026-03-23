"""
LLM Agent Executor for HoloLoom Workflow Builder

Provides execution backends for all LLM agent types:
- Simple LLM Prompt
- Structured Output LLM
- Prompt Chain
- Few-Shot Learner
- Multi-Model Consensus
- RAG Prompt

Supports multiple providers: OpenAI, Anthropic, Ollama, local models
"""

import json
import os
from dataclasses import dataclass
from typing import Any

# ============================================================================
# LLM Client Abstraction
# ============================================================================

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    usage: dict[str, int]
    model: str
    provider: str
    finish_reason: str = "stop"
    raw_response: dict | None = None


class LLMClient:
    """Abstract base for LLM providers"""

    def __init__(self, provider: str, api_key: str | None = None):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()

    def _get_api_key(self) -> str | None:
        """Get API key from environment"""
        key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'ollama': None  # Local, no API key needed
        }
        env_var = key_map.get(self.provider)
        return os.getenv(env_var) if env_var else None

    async def complete(self,
                      prompt: str,
                      system: str = "",
                      model: str = "gpt-4",
                      temperature: float = 0.7,
                      max_tokens: int = 1000) -> LLMResponse:
        """Complete a prompt - must be implemented by subclasses"""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def __init__(self, api_key: str | None = None):
        super().__init__('openai', api_key)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: openai package not installed. Install with: pip install openai")

    async def complete(self,
                      prompt: str,
                      system: str = "You are a helpful assistant.",
                      model: str = "gpt-4",
                      temperature: float = 0.7,
                      max_tokens: int = 1000) -> LLMResponse:
        if not self.available:
            raise RuntimeError("OpenAI client not available. Install openai package.")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            model=model,
            provider='openai',
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump()
        )


class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""

    def __init__(self, api_key: str | None = None):
        super().__init__('anthropic', api_key)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: anthropic package not installed. Install with: pip install anthropic")

    async def complete(self,
                      prompt: str,
                      system: str = "You are a helpful assistant.",
                      model: str = "claude-3-opus-20240229",
                      temperature: float = 0.7,
                      max_tokens: int = 1000) -> LLMResponse:
        if not self.available:
            raise RuntimeError("Anthropic client not available. Install anthropic package.")

        # Map common model names to full Anthropic model IDs
        model_map = {
            'claude-3-opus': 'claude-3-opus-20240229',
            'claude-3-sonnet': 'claude-3-sonnet-20240229',
            'claude-3-haiku': 'claude-3-haiku-20240307'
        }
        full_model = model_map.get(model, model)

        response = await self.client.messages.create(
            model=full_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )

        return LLMResponse(
            content=response.content[0].text,
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            },
            model=full_model,
            provider='anthropic',
            finish_reason=response.stop_reason,
            raw_response={'id': response.id, 'model': response.model}
        )


class OllamaClient(LLMClient):
    """Ollama local model client"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__('ollama', api_key=None)
        self.base_url = base_url
        try:
            import httpx
            self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: httpx package not installed. Install with: pip install httpx")

    async def complete(self,
                      prompt: str,
                      system: str = "You are a helpful assistant.",
                      model: str = "llama3",
                      temperature: float = 0.7,
                      max_tokens: int = 1000) -> LLMResponse:
        if not self.available:
            raise RuntimeError("Ollama client not available. Install httpx package.")

        payload = {
            "model": model,
            "prompt": f"{system}\n\nUser: {prompt}\n\nAssistant:",
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }

        response = await self.client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data.get('response', ''),
            usage={
                'prompt_tokens': data.get('prompt_eval_count', 0),
                'completion_tokens': data.get('eval_count', 0),
                'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
            },
            model=model,
            provider='ollama',
            finish_reason='stop',
            raw_response=data
        )


# ============================================================================
# LLM Executor Functions
# ============================================================================

def create_client(provider: str) -> LLMClient:
    """Factory function to create appropriate LLM client"""
    clients = {
        'openai': OpenAIClient,
        'anthropic': AnthropicClient,
        'ollama': OllamaClient
    }

    client_class = clients.get(provider)
    if not client_class:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(clients.keys())}")

    return client_class()


async def execute_llm_prompt(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Execute simple LLM prompt

    Config:
        provider: 'openai', 'anthropic', 'ollama'
        model: model name
        temperature: 0.0-2.0
        max_tokens: int
        system_prompt: system instructions
        user_prompt_template: template with ${variable} substitutions

    Inputs:
        prompt: user prompt (or variables for template)
        context: optional additional context

    Returns:
        response: LLM response text
        usage: token usage stats
    """
    provider = config.get('provider', 'openai')
    model = config.get('model', 'gpt-4')
    temperature = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens', 1000)
    system_prompt = config.get('system_prompt', 'You are a helpful assistant.')
    user_prompt_template = config.get('user_prompt_template', '${input.text}')

    # Substitute variables in template
    user_prompt = substitute_template(user_prompt_template, inputs)

    # Add context if provided
    if 'context' in inputs and inputs['context']:
        user_prompt = f"Context: {inputs['context']}\n\n{user_prompt}"

    # Execute
    client = create_client(provider)
    response = await client.complete(
        prompt=user_prompt,
        system=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return {
        'response': response.content,
        'usage': response.usage,
        'model': response.model,
        'provider': response.provider
    }


async def execute_structured_llm(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Execute LLM with structured JSON output

    Config:
        provider: 'openai', 'anthropic'
        model: model name
        temperature: 0.0-1.0 (lower for structured output)
        output_schema: JSON Schema defining expected output
        enforce_schema: bool
        retry_on_invalid: bool
        max_retries: int

    Returns:
        structured_data: parsed JSON matching schema
        raw_response: original LLM response
        valid: bool (whether output matches schema)
    """
    provider = config.get('provider', 'openai')
    model = config.get('model', 'gpt-4')
    temperature = config.get('temperature', 0.3)
    output_schema = json.loads(config.get('output_schema', '{}'))
    enforce_schema = config.get('enforce_schema', True)
    retry_on_invalid = config.get('retry_on_invalid', True)
    max_retries = config.get('max_retries', 3)

    # Build system prompt with schema instructions
    schema_str = json.dumps(output_schema, indent=2)
    system_prompt = f"""You must respond with valid JSON matching this schema:

{schema_str}

Respond ONLY with the JSON object, no additional text or explanation."""

    user_prompt = inputs.get('prompt', '')

    client = create_client(provider)

    for attempt in range(max_retries):
        response = await client.complete(
            prompt=user_prompt,
            system=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=2000
        )

        # Try to parse JSON
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response.content.strip()
            if content.startswith('```'):
                # Extract from code block
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content

            structured_data = json.loads(content)

            # Validate against schema if enforced
            if enforce_schema:
                is_valid = validate_schema(structured_data, output_schema)
                if not is_valid and retry_on_invalid and attempt < max_retries - 1:
                    # Add feedback to prompt and retry
                    user_prompt += "\n\nPrevious attempt was invalid. Ensure the JSON matches the schema exactly."
                    continue
            else:
                is_valid = True

            return {
                'structured_data': structured_data,
                'raw_response': response.content,
                'valid': is_valid,
                'attempts': attempt + 1
            }

        except json.JSONDecodeError as e:
            if retry_on_invalid and attempt < max_retries - 1:
                user_prompt += f"\n\nPrevious response was not valid JSON. Error: {str(e)}. Please respond with valid JSON only."
                continue
            else:
                return {
                    'structured_data': None,
                    'raw_response': response.content,
                    'valid': False,
                    'error': str(e),
                    'attempts': attempt + 1
                }

    return {
        'structured_data': None,
        'raw_response': response.content,
        'valid': False,
        'error': 'Max retries exceeded',
        'attempts': max_retries
    }


async def execute_prompt_chain(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Execute chain of prompts sequentially

    Config:
        provider: 'openai', 'anthropic', 'ollama'
        model: model name
        temperature: default temperature
        chain_steps: [
            {name: 'step1', prompt: 'template', temperature: 0.3},
            {name: 'step2', prompt: 'template using ${step1.output}'},
            ...
        ]
        preserve_all_steps: bool

    Returns:
        final_response: output of last step
        intermediate_steps: [{name, input, output}, ...]
    """
    provider = config.get('provider', 'openai')
    model = config.get('model', 'gpt-4')
    default_temp = config.get('temperature', 0.7)
    chain_steps = json.loads(config.get('chain_steps', '[]'))
    preserve_all_steps = config.get('preserve_all_steps', True)

    client = create_client(provider)

    # Track results from each step
    step_results = {}
    intermediate_steps = []

    for step in chain_steps:
        step_name = step['name']
        prompt_template = step['prompt']
        temperature = step.get('temperature', default_temp)

        # Substitute variables from previous steps and inputs
        context = {**inputs, **step_results}
        prompt = substitute_template(prompt_template, context)

        # Execute this step
        response = await client.complete(
            prompt=prompt,
            system="You are a helpful assistant.",
            model=model,
            temperature=temperature,
            max_tokens=2000
        )

        # Store result
        step_results[step_name] = {
            'output': response.content,
            'usage': response.usage
        }

        if preserve_all_steps:
            intermediate_steps.append({
                'name': step_name,
                'input': prompt,
                'output': response.content,
                'usage': response.usage
            })

    # Last step is the final output
    last_step = chain_steps[-1]['name'] if chain_steps else None
    final_response = step_results.get(last_step, {}).get('output', '') if last_step else ''

    return {
        'final_response': final_response,
        'intermediate_steps': intermediate_steps if preserve_all_steps else [],
        'all_results': step_results
    }


async def execute_few_shot(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Execute few-shot learning prompt

    Config:
        provider, model, temperature
        task_description: description of task
        examples: [{input, output}, ...]
        num_examples: how many to use
        auto_select_examples: bool (NYI - would use semantic search)

    Returns:
        response: LLM output
        confidence: estimated confidence
    """
    provider = config.get('provider', 'openai')
    model = config.get('model', 'gpt-4')
    temperature = config.get('temperature', 0.5)
    task_description = config.get('task_description', '')
    examples = json.loads(config.get('examples', '[]'))
    num_examples = config.get('num_examples', 3)

    # Build few-shot prompt
    few_shot_prompt = f"{task_description}\n\nHere are some examples:\n\n"

    for i, example in enumerate(examples[:num_examples]):
        few_shot_prompt += f"Example {i+1}:\n"
        few_shot_prompt += f"Input: {example['input']}\n"
        few_shot_prompt += f"Output: {example['output']}\n\n"

    few_shot_prompt += "Now perform the same task on this new input:\n"
    few_shot_prompt += f"Input: {inputs.get('query', '')}\n"
    few_shot_prompt += "Output:"

    client = create_client(provider)
    response = await client.complete(
        prompt=few_shot_prompt,
        system="You are a helpful assistant that learns from examples.",
        model=model,
        temperature=temperature,
        max_tokens=1000
    )

    return {
        'response': response.content,
        'confidence': 0.8,  # Simple heuristic, could be improved
        'num_examples_used': min(num_examples, len(examples))
    }


async def execute_llm_consensus(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Execute multi-model consensus

    Config:
        models: [{provider, model, weight}, ...]
        consensus_strategy: 'majority_vote', 'weighted_average', 'all_agree', 'best_of_n'
        temperature: float
        require_unanimous: bool
        min_agreement_threshold: float

    Returns:
        consensus_response: final consensus output
        agreement_score: 0.0-1.0
        individual_responses: [{model, response}, ...]
    """
    models_config = json.loads(config.get('models', '[]'))
    consensus_strategy = config.get('consensus_strategy', 'majority_vote')
    temperature = config.get('temperature', 0.7)
    config.get('require_unanimous', False)
    config.get('min_agreement_threshold', 0.6)

    prompt = inputs.get('prompt', '')

    # Query all models in parallel
    tasks = []
    for model_cfg in models_config:
        provider = model_cfg['provider']
        model = model_cfg['model']
        weight = model_cfg.get('weight', 1.0)

        client = create_client(provider)
        task = client.complete(
            prompt=prompt,
            system="You are a helpful assistant.",
            model=model,
            temperature=temperature,
            max_tokens=1000
        )
        tasks.append((task, weight, f"{provider}/{model}"))

    # Gather all responses
    responses = []
    for (task, weight, model_name) in tasks:
        try:
            response = await task
            responses.append({
                'model': model_name,
                'response': response.content,
                'weight': weight
            })
        except Exception as e:
            print(f"Error from {model_name}: {e}")

    # Apply consensus strategy
    if consensus_strategy == 'majority_vote':
        # Simple: pick most common response (considering weights)
        # For simplicity, just return weighted average of first/last
        consensus = responses[0]['response'] if responses else ""
        agreement_score = 0.7  # Placeholder

    elif consensus_strategy == 'weighted_average':
        # For text, just concatenate with weights noted
        consensus = "\n\n".join([f"[{r['model']}]: {r['response']}" for r in responses])
        agreement_score = 0.8

    elif consensus_strategy == 'all_agree':
        # Check if all responses similar
        if len(set(r['response'] for r in responses)) == 1:
            consensus = responses[0]['response']
            agreement_score = 1.0
        else:
            consensus = "Models disagree. Review individual responses."
            agreement_score = 0.0

    else:  # best_of_n
        # Return highest weighted response
        best = max(responses, key=lambda r: r['weight'])
        consensus = best['response']
        agreement_score = best['weight'] / sum(r['weight'] for r in responses)

    return {
        'consensus_response': consensus,
        'agreement_score': agreement_score,
        'individual_responses': responses
    }


async def execute_rag_prompt(config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Execute RAG (Retrieval-Augmented Generation) prompt

    Config:
        provider, model, temperature
        retrieval_k: number of documents to retrieve
        cite_sources: bool
        source_format: 'inline', 'footnotes', 'appendix'
        system_prompt: instructions
        require_citations: bool
        hallucination_check: bool

    Inputs:
        query: user question
        context: retrieved documents (or auto-retrieve if not provided)

    Returns:
        response: answer with citations
        sources: list of source documents used
        confidence: estimated confidence
    """
    provider = config.get('provider', 'openai')
    model = config.get('model', 'gpt-4')
    temperature = config.get('temperature', 0.3)
    config.get('retrieval_k', 5)
    cite_sources = config.get('cite_sources', True)
    source_format = config.get('source_format', 'inline')
    system_prompt = config.get('system_prompt', 'Answer based on provided context. Cite sources.')

    query = inputs.get('query', '')
    context = inputs.get('context', '')

    # If no context provided, retrieve from memory (integrate with HoloLoom memory)
    if not context:
        # Placeholder: In real implementation, would call Memory Search
        context = "No context provided. Using general knowledge."

    # Build RAG prompt
    if cite_sources:
        if source_format == 'inline':
            rag_prompt = f"""Context documents:
{context}

Question: {query}

Answer the question based on the context above. Cite sources inline using [Source X] notation."""
        elif source_format == 'footnotes':
            rag_prompt = f"""Context documents:
{context}

Question: {query}

Answer with footnote citations [1], [2], etc."""
        else:  # appendix
            rag_prompt = f"""Context documents:
{context}

Question: {query}

Answer first, then list sources in appendix."""
    else:
        rag_prompt = f"""Context: {context}

Question: {query}"""

    client = create_client(provider)
    response = await client.complete(
        prompt=rag_prompt,
        system=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=2000
    )

    return {
        'response': response.content,
        'sources': [{'text': context}],  # Simplified - would parse actual sources
        'confidence': 0.85
    }


# ============================================================================
# Helper Functions
# ============================================================================

def substitute_template(template: str, context: dict[str, Any]) -> str:
    """
    Substitute ${variable.path} placeholders in template

    Supports:
    - ${input.text}
    - ${step1.output}
    - ${previous_step.result.field}
    """
    import re

    def replace_var(match):
        var_path = match.group(1)
        parts = var_path.split('.')

        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, f"${{{var_path}}}")  # Keep placeholder if not found
            else:
                return f"${{{var_path}}}"

        return str(value)

    return re.sub(r'\$\{([^}]+)\}', replace_var, template)


def validate_schema(data: dict, schema: dict) -> bool:
    """
    Simple JSON Schema validation

    In production, use jsonschema library:
    from jsonschema import validate
    validate(instance=data, schema=schema)
    """
    try:
        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                return False

        # Check types
        properties = schema.get('properties', {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get('type')
                if expected_type == 'string' and not isinstance(value, str):
                    return False
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    return False
                elif expected_type == 'object' and not isinstance(value, dict):
                    return False
                elif expected_type == 'array' and not isinstance(value, list):
                    return False

        return True
    except Exception:
        return False


# ============================================================================
# Executor Dispatch
# ============================================================================

async def execute_llm_agent(agent_type: str, config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Main dispatcher for LLM agent execution

    Args:
        agent_type: 'llm_prompt', 'structured_llm', 'prompt_chain', etc.
        config: agent configuration
        inputs: input data

    Returns:
        Dict with agent-specific outputs
    """
    executors = {
        'llm_prompt': execute_llm_prompt,
        'structured_llm': execute_structured_llm,
        'prompt_chain': execute_prompt_chain,
        'few_shot': execute_few_shot,
        'llm_consensus': execute_llm_consensus,
        'rag_prompt': execute_rag_prompt
    }

    executor = executors.get(agent_type)
    if not executor:
        raise ValueError(f"Unknown LLM agent type: {agent_type}")

    return await executor(config, inputs)
