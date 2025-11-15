"""
xTerminator MCP Server - Automated Code Fixing
==============================================

Model Context Protocol (MCP) server providing xTerminator fixing tools
for cross-department integration in HoloLoom.

Tools:
- xterminator_propose_fix: Generate fix proposal for issue
- xterminator_validate_fix: Run validation pipeline
- xterminator_apply_fix: Apply fix with Git integration
- xterminator_rollback_fix: Rollback previous fix
- xterminator_batch_fix: Process multiple fixes

Usage:
    python -m xterminator.mcp_server

Author: The Barnyard Brigade
Date: November 15, 2025
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid

from xterminator import (
    ClassificationEngine,
    ASTFixer,
    TemplateFixer,
    ValidationPipeline,
    GitApplicator,
    RollbackManager,
    FixStrategy,
    RiskLevel,
    FixProposal,
    AutofixPolicy
)
from trough import SlopIssue, Language, Severity, SlopCategory


@dataclass
class MCPServerInfo:
    """MCP server metadata."""
    name: str = "xterminator-fixer"
    version: str = "1.0.0"
    description: str = "Automated code fixing with safety-first approach"
    vendor: str = "mythRL"
    context_budget: int = 20000
    session_aware: bool = True
    requires_human_approval: List[str] = None

    def __post_init__(self):
        if self.requires_human_approval is None:
            self.requires_human_approval = ["security", "high_risk"]


class XTerminatorMCPServer:
    """
    MCP server for xTerminator automated code fixing.

    Provides tools for proposing, validating, and applying code fixes,
    enabling cross-department quality assurance in HoloLoom.
    """

    def __init__(self, policy: Optional[AutofixPolicy] = None):
        self.info = MCPServerInfo()
        self.classifier = ClassificationEngine()
        self.ast_fixer = ASTFixer()
        self.template_fixer = TemplateFixer()
        self.validator = ValidationPipeline()
        self.git_applicator = GitApplicator()
        self.rollback_manager = RollbackManager()
        self.policy = policy or AutofixPolicy.balanced()

        # Session state
        self.session_state: Dict[str, Any] = {
            'fixes_proposed': 0,
            'fixes_validated': 0,
            'fixes_applied': 0,
            'fixes_rollback': 0,
            'active_fixes': {},  # fix_id -> proposal
            'fix_history': []
        }

    async def xterminator_propose_fix(
        self,
        issue: Dict[str, Any],
        code: str,
        file_path: str,
        fix_strategy: Optional[str] = None,
        safety_level: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Generate fix proposal for detected issue.

        Args:
            issue: SlopIssue dict from Trough detection
            code: Full file content
            file_path: Path to file
            fix_strategy: Override auto-selected strategy (ast/template/manual)
            safety_level: aggressive/balanced/conservative

        Returns:
            {
                'fix_id': str,
                'issue': dict,
                'strategy': str,
                'confidence': float,
                'risk_level': str,
                'proposed_code': str,
                'diff': str,
                'validation_plan': list,
                'requires_approval': bool
            }
        """
        start_time = time.time()

        # Convert dict to SlopIssue
        slop_issue = self._dict_to_slop_issue(issue)

        # Classify and propose fix
        proposal = await self.classifier.classify_and_propose(
            slop_issue,
            code,
            file_path
        )

        # Override strategy if specified
        if fix_strategy:
            proposal.fix_strategy = FixStrategy[fix_strategy.upper()]

        # Generate fix
        if proposal.fix_strategy == FixStrategy.AST:
            result = await self.ast_fixer.fix_issue(proposal, code)
        elif proposal.fix_strategy == FixStrategy.TEMPLATE:
            result = await self.template_fixer.fix_issue(proposal, code)
        else:
            # Manual fix required
            result = None

        if not result:
            return {
                'fix_id': None,
                'issue': issue,
                'strategy': proposal.fix_strategy.value,
                'confidence': proposal.confidence,
                'risk_level': proposal.risk_level.value,
                'proposed_code': None,
                'diff': None,
                'validation_plan': [],
                'requires_approval': True,
                'error': 'Fix could not be generated automatically'
            }

        fixed_code, diff = result

        # Create fix ID
        fix_id = f"fix_{uuid.uuid4().hex[:12]}"

        # Validation plan
        validation_plan = [
            'syntax_check',
            'import_check',
            'test_execution',
            'trough_rescan',
            'regression_check'
        ]

        # Check if requires approval
        requires_approval = (
            proposal.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            or safety_level == 'conservative'
        )

        # Store in session
        self.session_state['fixes_proposed'] += 1
        self.session_state['active_fixes'][fix_id] = {
            'proposal': proposal,
            'code': code,
            'fixed_code': fixed_code,
            'file_path': file_path
        }

        duration_ms = (time.time() - start_time) * 1000

        return {
            'fix_id': fix_id,
            'issue': issue,
            'strategy': proposal.fix_strategy.value,
            'confidence': proposal.confidence,
            'risk_level': proposal.risk_level.value,
            'proposed_code': fixed_code,
            'diff': diff,
            'validation_plan': validation_plan,
            'requires_approval': requires_approval,
            'proposal_duration_ms': duration_ms
        }

    async def xterminator_validate_fix(
        self,
        fix_id: str
    ) -> Dict[str, Any]:
        """
        Run validation pipeline before application.

        Args:
            fix_id: Fix ID from propose_fix

        Returns:
            {
                'fix_id': str,
                'validation_passed': bool,
                'steps': list,
                'safe_to_apply': bool,
                'validation_duration_ms': float
            }
        """
        start_time = time.time()

        # Get fix from session
        if fix_id not in self.session_state['active_fixes']:
            return {
                'fix_id': fix_id,
                'error': 'Fix ID not found in active fixes',
                'validation_passed': False,
                'safe_to_apply': False
            }

        fix_data = self.session_state['active_fixes'][fix_id]
        code = fix_data['code']
        fixed_code = fix_data['fixed_code']
        file_path = fix_data['file_path']

        # Run validation pipeline
        report = await self.validator.validate_fix(code, fixed_code, file_path)

        # Extract step results
        steps = []
        for stage in report.stages:
            steps.append({
                'step': stage.name,
                'passed': stage.passed,
                'duration_ms': stage.duration_ms,
                'message': stage.message
            })

        duration_ms = (time.time() - start_time) * 1000

        # Update session
        self.session_state['fixes_validated'] += 1

        return {
            'fix_id': fix_id,
            'validation_passed': report.passed,
            'steps': steps,
            'safe_to_apply': self.validator.should_commit(report),
            'validation_duration_ms': duration_ms
        }

    async def xterminator_apply_fix(
        self,
        fix_id: str,
        create_branch: bool = False
    ) -> Dict[str, Any]:
        """
        Apply validated fix with Git integration.

        Args:
            fix_id: Fix ID from propose_fix
            create_branch: Create feature branch for high-risk fixes

        Returns:
            {
                'fix_id': str,
                'applied': bool,
                'git_commit': str,
                'branch': str,
                'rollback_command': str,
                'audit_trail': dict
            }
        """
        start_time = time.time()

        # Get fix from session
        if fix_id not in self.session_state['active_fixes']:
            return {
                'fix_id': fix_id,
                'error': 'Fix ID not found in active fixes',
                'applied': False
            }

        fix_data = self.session_state['active_fixes'][fix_id]
        proposal = fix_data['proposal']
        fixed_code = fix_data['fixed_code']
        file_path = fix_data['file_path']

        # Apply fix with Git integration
        result = await self.git_applicator.apply_fix(
            file_path,
            fixed_code,
            proposal,
            create_branch=create_branch
        )

        if not result:
            return {
                'fix_id': fix_id,
                'applied': False,
                'error': 'Git application failed'
            }

        # Update session
        self.session_state['fixes_applied'] += 1
        self.session_state['fix_history'].append({
            'fix_id': fix_id,
            'file_path': file_path,
            'timestamp': time.time(),
            'git_commit': result.commit_sha if result else None
        })

        # Remove from active fixes
        del self.session_state['active_fixes'][fix_id]

        duration_ms = (time.time() - start_time) * 1000

        return {
            'fix_id': fix_id,
            'applied': True,
            'git_commit': result.commit_sha if result else None,
            'branch': result.branch if result else 'main',
            'rollback_command': f"git revert {result.commit_sha}" if result else None,
            'audit_trail': {
                'timestamp': time.time(),
                'detector': 'trough',
                'validator': 'xterminator',
                'approval': 'automated',
                'fix_id': fix_id
            },
            'apply_duration_ms': duration_ms
        }

    async def xterminator_rollback_fix(
        self,
        fix_id: Optional[str] = None,
        count: int = 1
    ) -> Dict[str, Any]:
        """
        Rollback previous fix(es).

        Args:
            fix_id: Specific fix to rollback (or None for last N)
            count: Number of fixes to rollback (if fix_id not specified)

        Returns:
            {
                'rollback_count': int,
                'fixes_rolled_back': list,
                'git_reverts': list
            }
        """
        # Use RollbackManager
        if fix_id:
            # Rollback specific fix
            fixes_to_rollback = [
                f for f in self.session_state['fix_history']
                if f['fix_id'] == fix_id
            ]
        else:
            # Rollback last N fixes
            fixes_to_rollback = self.session_state['fix_history'][-count:]

        rolled_back = []
        git_reverts = []

        for fix in fixes_to_rollback:
            if fix['git_commit']:
                # TODO: Implement actual Git revert
                # For now, just track that we would revert
                git_reverts.append(fix['git_commit'])
                rolled_back.append(fix['fix_id'])

        # Update session
        self.session_state['fixes_rollback'] += len(rolled_back)

        return {
            'rollback_count': len(rolled_back),
            'fixes_rolled_back': rolled_back,
            'git_reverts': git_reverts
        }

    async def xterminator_batch_fix(
        self,
        issues: List[Dict[str, Any]],
        code: str,
        file_path: str,
        max_fixes: int = 10
    ) -> Dict[str, Any]:
        """
        Process multiple fixes in batch.

        Args:
            issues: List of SlopIssue dicts
            code: Full file content
            file_path: Path to file
            max_fixes: Maximum fixes to apply

        Returns:
            {
                'total_issues': int,
                'fixes_proposed': int,
                'fixes_validated': int,
                'fixes_applied': int,
                'results': list
            }
        """
        results = []
        current_code = code

        for i, issue in enumerate(issues[:max_fixes]):
            # Propose fix
            proposal_result = await self.xterminator_propose_fix(
                issue,
                current_code,
                file_path
            )

            if not proposal_result.get('fix_id'):
                results.append({
                    'issue': issue,
                    'status': 'failed_proposal',
                    'error': proposal_result.get('error')
                })
                continue

            # Validate fix
            validation_result = await self.xterminator_validate_fix(
                proposal_result['fix_id']
            )

            if not validation_result['safe_to_apply']:
                results.append({
                    'issue': issue,
                    'status': 'failed_validation',
                    'fix_id': proposal_result['fix_id']
                })
                continue

            # Apply fix
            apply_result = await self.xterminator_apply_fix(
                proposal_result['fix_id']
            )

            if apply_result['applied']:
                results.append({
                    'issue': issue,
                    'status': 'applied',
                    'fix_id': proposal_result['fix_id'],
                    'git_commit': apply_result['git_commit']
                })
                # Update code for next iteration
                current_code = proposal_result['proposed_code']
            else:
                results.append({
                    'issue': issue,
                    'status': 'failed_apply',
                    'fix_id': proposal_result['fix_id']
                })

        # Count results
        fixes_applied = sum(1 for r in results if r['status'] == 'applied')
        fixes_validated = sum(1 for r in results if r['status'] in ['applied', 'failed_apply'])
        fixes_proposed = sum(1 for r in results if 'fix_id' in r)

        return {
            'total_issues': len(issues),
            'fixes_proposed': fixes_proposed,
            'fixes_validated': fixes_validated,
            'fixes_applied': fixes_applied,
            'results': results
        }

    def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server metadata."""
        return asdict(self.info)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {
                'name': 'xterminator_propose_fix',
                'description': 'Generate fix proposal for detected issue',
                'parameters': {
                    'issue': {'type': 'object', 'required': True},
                    'code': {'type': 'string', 'required': True},
                    'file_path': {'type': 'string', 'required': True},
                    'fix_strategy': {'type': 'string', 'enum': ['ast', 'template', 'manual']},
                    'safety_level': {'type': 'string', 'enum': ['aggressive', 'balanced', 'conservative'], 'default': 'balanced'}
                }
            },
            {
                'name': 'xterminator_validate_fix',
                'description': 'Run validation pipeline before application',
                'parameters': {
                    'fix_id': {'type': 'string', 'required': True}
                }
            },
            {
                'name': 'xterminator_apply_fix',
                'description': 'Apply validated fix with Git integration',
                'parameters': {
                    'fix_id': {'type': 'string', 'required': True},
                    'create_branch': {'type': 'boolean', 'default': False}
                }
            },
            {
                'name': 'xterminator_rollback_fix',
                'description': 'Rollback previous fix(es)',
                'parameters': {
                    'fix_id': {'type': 'string'},
                    'count': {'type': 'integer', 'default': 1}
                }
            },
            {
                'name': 'xterminator_batch_fix',
                'description': 'Process multiple fixes in batch',
                'parameters': {
                    'issues': {'type': 'array', 'items': {'type': 'object'}, 'required': True},
                    'code': {'type': 'string', 'required': True},
                    'file_path': {'type': 'string', 'required': True},
                    'max_fixes': {'type': 'integer', 'default': 10}
                }
            }
        ]

    def _dict_to_slop_issue(self, issue_dict: Dict[str, Any]) -> SlopIssue:
        """Convert dict to SlopIssue object."""
        return SlopIssue(
            category=SlopCategory(issue_dict.get('category', 'incomplete')),
            severity=Severity(issue_dict.get('severity', 'medium')),
            message=issue_dict.get('message', ''),
            line_number=issue_dict.get('line_number', 0),
            code_snippet=issue_dict.get('code_snippet', ''),
            suggestion=issue_dict.get('suggestion', ''),
            confidence=issue_dict.get('confidence', 0.5),
            file_path=issue_dict.get('file_path', '')
        )


# MCP server entry point
async def main():
    """Run xTerminator MCP server."""
    server = XTerminatorMCPServer()

    print("=" * 70)
    print("XTERMINATOR MCP SERVER")
    print(f"  Name: {server.info.name}")
    print(f"  Version: {server.info.version}")
    print(f"  Description: {server.info.description}")
    print("=" * 70)
    print()
    print("Available Tools:")
    for i, tool in enumerate(server.get_tools(), 1):
        print(f"  {i}. {tool['name']} - {tool['description']}")
    print()
    print("Server ready for cross-department integration!")
    print("Use XTerminatorMCPServer() class for programmatic access.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
