#!/usr/bin/env python3
"""
Template Fixer Tests
====================

Tests Charlotte's template-based fixing system.

"Templeton says: Test templates, not cheese!"
"""

import asyncio
import pytest
from dataclasses import dataclass

from .template_fixer import TemplateFixer, fix_with_template
from .templates import get_template, get_template_catalog, list_templates
from .xterminator_types import (
    FixProposal,
    FixStrategy,
    RiskLevel,
    CodeContext,
    ContextType
)


# Test helper
def create_proposal(
    original_code: str,
    category: str,
    file_path: str = "test.py",
    line_number: int = 10,
    context_code: str = None,
    lines_before: list = None,
    lines_after: list = None
) -> FixProposal:
    """Create test fix proposal"""
    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path=file_path,
        line_number=line_number,
        code_snippet=context_code or original_code,
        lines_before=lines_before or [],
        lines_after=lines_after or [],
        in_try_block=False,
        has_tests=True
    )

    return FixProposal(
        fix_id=f"test_{category}_{line_number}",
        issue_category=category,
        issue_severity="medium",
        risk_level=RiskLevel.MEDIUM,
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.85,
        original_code=original_code,
        context=context,
        explanation=f"Test fix for {category}"
    )


class TestTemplateLoader:
    """Test template loading and catalog"""

    def test_template_count(self):
        """Verify we have expected number of templates"""
        templates = list_templates()
        assert len(templates) >= 10, f"Expected ≥10 templates, got {len(templates)}"

    def test_get_template(self):
        """Test getting template by name"""
        template = get_template('add_try_except_file_io')
        assert template is not None
        assert template.name == 'add_try_except_file_io'
        assert template.category == 'error_handling'

    def test_template_catalog(self):
        """Test catalog generation"""
        catalog = get_template_catalog()
        assert "CHARLOTTE'S TEMPLATE CATALOG" in catalog
        assert "Some Spider spun templates!" in catalog
        assert len(catalog) > 100


class TestErrorHandlingTemplates:
    """Test error handling templates"""

    @pytest.mark.asyncio
    async def test_add_try_except_file_io(self):
        """Test wrapping file I/O in try/except"""
        original = 'f = open("config.txt")\ndata = f.read()'
        full_code = f'''import os

def load_config():
    {original}
    return data
'''

        proposal = create_proposal(
            original_code=original,
            category='error_handling',
            line_number=4
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify try/except added
        assert 'try:' in fixed_code
        assert 'except (IOError, OSError)' in fixed_code
        assert 'logger.error' in fixed_code

        # Verify imports added
        assert 'import logging' in fixed_code

    @pytest.mark.asyncio
    async def test_add_try_except_json(self):
        """Test wrapping JSON parsing in try/except"""
        original = 'data = json.load(f)'
        full_code = f'''import json

def load_data(f):
    {original}
    return data
'''

        proposal = create_proposal(
            original_code=original,
            category='error_handling',
            line_number=4
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify JSON error handling
        assert 'try:' in fixed_code
        assert 'except json.JSONDecodeError' in fixed_code
        assert 'logger.error' in fixed_code

    @pytest.mark.asyncio
    async def test_add_try_except_requests(self):
        """Test wrapping HTTP requests in try/except"""
        original = 'response = requests.get(url)'
        full_code = f'''import requests

def fetch_data(url):
    {original}
    return response.json()
'''

        proposal = create_proposal(
            original_code=original,
            category='error_handling',
            line_number=4
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify request error handling
        assert 'try:' in fixed_code
        assert 'timeout=30' in fixed_code
        assert 'raise_for_status()' in fixed_code
        assert 'except requests.RequestException' in fixed_code


class TestResourceManagement:
    """Test resource management templates"""

    @pytest.mark.asyncio
    async def test_add_context_manager(self):
        """Test converting open() to context manager"""
        original = 'f = open("data.txt")\ncontent = f.read()\nf.close()'
        full_code = f'''def read_file():
    {original}
    return content
'''

        proposal = create_proposal(
            original_code=original,
            category='resource_management',
            line_number=2
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify context manager
        assert 'with open(' in fixed_code
        assert ' as f:' in fixed_code


class TestHardcodedValues:
    """Test hardcoded value templates"""

    @pytest.mark.asyncio
    async def test_move_to_env_var(self):
        """Test moving secrets to environment variables"""
        original = 'API_KEY = "sk-abc123..."'
        full_code = f'''import requests

{original}

def call_api():
    headers = {{"Authorization": f"Bearer {{API_KEY}}"}}
    return requests.get(API_ENDPOINT, headers=headers)
'''

        proposal = create_proposal(
            original_code=original,
            category='hardcoded_values',
            line_number=3
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify env var
        assert 'os.getenv("API_KEY"' in fixed_code
        assert 'raise ValueError("API_KEY' in fixed_code

        # Verify import added
        assert 'import os' in fixed_code


class TestCodeQuality:
    """Test code quality templates"""

    @pytest.mark.asyncio
    async def test_fix_timezone_naive(self):
        """Test fixing naive datetime"""
        original = 'now = datetime.now()'
        full_code = f'''from datetime import datetime

def get_timestamp():
    {original}
    return now
'''

        proposal = create_proposal(
            original_code=original,
            category='datetime',
            line_number=4
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify timezone-aware
        assert 'datetime.now(timezone.utc)' in fixed_code
        assert 'from datetime import datetime, timezone' in fixed_code


class TestIndentationPreservation:
    """Test that indentation is preserved correctly"""

    @pytest.mark.asyncio
    async def test_preserves_indentation(self):
        """Test indentation preservation"""
        original = '        data = json.load(f)'
        full_code = f'''import json

class DataLoader:
    def load(self, f):
{original}
        return data
'''

        proposal = create_proposal(
            original_code=original,
            category='error_handling',
            line_number=5
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify indentation preserved
        lines = fixed_code.split('\n')
        for line in lines:
            if 'try:' in line or 'except' in line:
                # Should be indented with 8 spaces (class method level)
                assert line.startswith('        ')


class TestImportManagement:
    """Test smart import management"""

    @pytest.mark.asyncio
    async def test_adds_missing_imports(self):
        """Test adding missing imports"""
        original = 'response = requests.get(url)'
        full_code = f'''def fetch_data(url):
    {original}
    return response.json()
'''

        proposal = create_proposal(
            original_code=original,
            category='error_handling',
            line_number=2
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify imports added
        assert 'import requests' in fixed_code
        assert 'import logging' in fixed_code

    @pytest.mark.asyncio
    async def test_skips_existing_imports(self):
        """Test not duplicating existing imports"""
        original = 'response = requests.get(url)'
        full_code = f'''import requests
import logging

logger = logging.getLogger(__name__)

def fetch_data(url):
    {original}
    return response.json()
'''

        proposal = create_proposal(
            original_code=original,
            category='error_handling',
            line_number=7
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Count occurrences of imports - should only appear once
        assert fixed_code.count('import requests') == 1
        assert fixed_code.count('import logging') == 1


class TestDiffGeneration:
    """Test diff generation"""

    @pytest.mark.asyncio
    async def test_generates_valid_diff(self):
        """Test diff generation"""
        original = 'API_KEY = "sk-abc123"'
        full_code = f'''{original}

def call_api():
    pass
'''

        proposal = create_proposal(
            original_code=original,
            category='hardcoded_values',
            line_number=1,
            file_path='api_client.py'
        )

        fixer = TemplateFixer()
        result = await fixer.fix_issue(proposal, full_code)

        assert result is not None
        fixed_code, diff = result

        # Verify diff format
        assert '---' in diff
        assert '+++' in diff
        assert 'api_client.py' in diff


class TestCanFix:
    """Test can_fix method"""

    def test_can_fix_template_strategy(self):
        """Test can_fix returns True for TEMPLATE strategy"""
        proposal = create_proposal(
            original_code='data = json.load(f)',
            category='error_handling'
        )
        proposal.fix_strategy = FixStrategy.TEMPLATE

        fixer = TemplateFixer()
        assert fixer.can_fix(proposal) is True

    def test_cannot_fix_ast_strategy(self):
        """Test can_fix returns False for AST strategy"""
        proposal = create_proposal(
            original_code='x = 1',
            category='copy_paste'
        )
        proposal.fix_strategy = FixStrategy.AST

        fixer = TemplateFixer()
        assert fixer.can_fix(proposal) is False

    def test_cannot_fix_unknown_category(self):
        """Test can_fix returns False for unknown category"""
        proposal = create_proposal(
            original_code='x = 1',
            category='unknown_category_xyz'
        )
        proposal.fix_strategy = FixStrategy.TEMPLATE

        fixer = TemplateFixer()
        assert fixer.can_fix(proposal) is False


# Run tests
if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])
