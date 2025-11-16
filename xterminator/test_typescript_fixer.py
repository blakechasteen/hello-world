"""
Test Suite for TypeScript Fixer
================================

Tests all TypeScript autofix capabilities.

Author: mythRL Team
Date: November 16, 2025
"""

import pytest
import asyncio
from typescript_fixer import TypeScriptFixer


class TestTypeScriptFixer:
    """Test suite for TypeScriptFixer"""

    @pytest.fixture
    def fixer(self):
        """Create fixer instance for testing"""
        return TypeScriptFixer(use_llm=False)

    @pytest.mark.asyncio
    async def test_unused_import_single(self, fixer):
        """Test removing single unused import"""
        code = """import { Unused } from './module';

export class Example {
    test() {
        return 42;
    }
}
"""
        issue = {
            'category': 'dead_code',
            'message': "Unused import 'Unused'",
            'line_number': 1,
            'code_snippet': "import { Unused } from './module';"
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert 'Unused' not in fixed_code
        assert 'export class Example' in fixed_code

    @pytest.mark.asyncio
    async def test_unused_import_multiple(self, fixer):
        """Test removing one import from multiple"""
        code = """import { Used, Unused, AlsoUsed } from './module';

export class Example {
    test() {
        const x = new Used();
        const y = new AlsoUsed();
        return x;
    }
}
"""
        issue = {
            'category': 'dead_code',
            'message': "Unused import 'Unused'",
            'line_number': 1,
            'code_snippet': "import { Used, Unused, AlsoUsed } from './module';"
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert 'Unused' not in fixed_code
        assert 'Used' in fixed_code
        assert 'AlsoUsed' in fixed_code

    @pytest.mark.asyncio
    async def test_unused_namespace_import(self, fixer):
        """Test removing namespace import"""
        code = """import * as vscode from 'vscode';

export class Example {
    test() {
        return 42;
    }
}
"""
        issue = {
            'category': 'dead_code',
            'message': "Unused import 'vscode'",
            'line_number': 1,
            'code_snippet': "import * as vscode from 'vscode';"
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert 'import * as vscode' not in fixed_code

    @pytest.mark.asyncio
    async def test_fix_any_type(self, fixer):
        """Test replacing 'any' with 'unknown'"""
        code = """export function process(data: any) {
    return data.value;
}
"""
        issue = {
            'category': 'type_safety',
            'message': "Parameter 'data' implicitly has an 'any' type",
            'line_number': 1,
            'code_snippet': 'export function process(data: any) {'
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert ': unknown' in fixed_code
        assert ': any' not in fixed_code

    @pytest.mark.asyncio
    async def test_fix_any_array(self, fixer):
        """Test replacing 'any[]' with 'unknown[]'"""
        code = """const items: any[] = [];
"""
        issue = {
            'category': 'type_safety',
            'message': "Variable 'items' has type 'any[]'",
            'line_number': 1,
            'code_snippet': 'const items: any[] = [];'
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert 'unknown[]' in fixed_code
        assert 'any[]' not in fixed_code

    @pytest.mark.asyncio
    async def test_fix_non_null_assertion_property(self, fixer):
        """Test replacing non-null assertion with optional chaining"""
        code = """const value = obj!.property;
"""
        issue = {
            'category': 'type_safety',
            'message': "Non-null assertion is risky",
            'line_number': 1,
            'code_snippet': "const value = obj!.property;"
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert 'obj?.property' in fixed_code
        assert '!.' not in fixed_code

    @pytest.mark.asyncio
    async def test_fix_non_null_assertion_array(self, fixer):
        """Test replacing non-null assertion for array access"""
        code = """const item = arr![0];
"""
        issue = {
            'category': 'type_safety',
            'message': "Non-null assertion is risky",
            'line_number': 1,
            'code_snippet': "const item = arr![0];"
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert 'arr?.[0]' in fixed_code
        assert '![' not in fixed_code

    @pytest.mark.asyncio
    async def test_fix_console_log(self, fixer):
        """Test commenting out console.log"""
        code = """export class Example {
    test() {
        console.log('Debug message');
        return 42;
    }
}
"""
        issue = {
            'category': 'code_quality',
            'message': "Unexpected console.log",
            'line_number': 3,
            'code_snippet': "console.log('Debug message');"
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is not None

        fixed_code, diff = result
        assert '// console.log' in fixed_code

    @pytest.mark.asyncio
    async def test_detect_unused_imports(self, fixer):
        """Test detecting all unused imports"""
        code = """import { Used, Unused1, Unused2 } from './module';
import * as vscode from 'vscode';
import React from 'react';

export class Example {
    test() {
        return new Used();
    }
}
"""
        issues = fixer.detect_unused_imports(code)

        # Should detect: Unused1, Unused2, vscode, React
        assert len(issues) >= 2
        assert any('Unused1' in issue['message'] for issue in issues)
        assert any('Unused2' in issue['message'] for issue in issues)

    @pytest.mark.asyncio
    async def test_no_fix_for_empty_issue(self, fixer):
        """Test that invalid issues return None"""
        code = """export class Example {}"""
        issue = {
            'category': 'unknown_category',
            'message': "Unknown issue",
            'line_number': 1,
            'code_snippet': ""
        }

        result = await fixer.fix_issue(issue, code, 'test.ts')
        assert result is None

    def test_infer_type_from_usage(self, fixer):
        """Test type inference from variable usage"""
        code = """const age = 42;
const name = 'Alice';
const isValid = true;
"""
        # Number
        assert fixer._infer_type_from_usage('age', code) == 'number'

        # String
        assert fixer._infer_type_from_usage('name', code) == 'string'

        # Boolean
        assert fixer._infer_type_from_usage('isValid', code) == 'boolean'


# Integration tests with real Trough files (optional)
class TestTroughIntegration:
    """Integration tests with actual Trough TypeScript files"""

    @pytest.fixture
    def fixer(self):
        return TypeScriptFixer(use_llm=False)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scan_trough_files(self, fixer):
        """Test scanning actual Trough TypeScript files"""
        import os
        from pathlib import Path

        trough_src = Path(__file__).parent.parent / 'trough' / 'src'
        if not trough_src.exists():
            pytest.skip("Trough directory not found")

        # Scan for unused imports in all TS files
        total_issues = 0
        for ts_file in trough_src.glob('*.ts'):
            code = ts_file.read_text()
            issues = fixer.detect_unused_imports(code)
            if issues:
                print(f"\n{ts_file.name}: {len(issues)} unused imports")
                for issue in issues:
                    print(f"  Line {issue['line_number']}: {issue['message']}")
                total_issues += len(issues)

        print(f"\nTotal unused imports found: {total_issues}")


# Performance tests
class TestPerformance:
    """Performance benchmarks for TypeScript fixer"""

    @pytest.fixture
    def fixer(self):
        return TypeScriptFixer(use_llm=False)

    def test_detect_performance(self, fixer, benchmark):
        """Benchmark unused import detection"""
        # Generate large TypeScript file
        code = "import { Used } from './module';\n"
        code += "\n".join([f"import {{ Unused{i} }} from './module{i}';" for i in range(100)])
        code += "\nexport class Example { test() { return new Used(); } }\n"

        def detect():
            return fixer.detect_unused_imports(code)

        result = benchmark(detect)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fix_performance(self, fixer, benchmark):
        """Benchmark fix application"""
        code = """import { Unused } from './module';
export class Example {}
"""
        issue = {
            'category': 'dead_code',
            'message': "Unused import 'Unused'",
            'line_number': 1,
            'code_snippet': "import { Unused } from './module';"
        }

        async def fix():
            return await fixer.fix_issue(issue, code, 'test.ts')

        # Run benchmark
        import time
        start = time.time()
        for _ in range(100):
            await fix()
        duration = time.time() - start

        print(f"\n100 fixes in {duration:.3f}s ({duration*10:.1f}ms per fix)")
        assert duration < 1.0  # Should be fast


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
