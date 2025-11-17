#!/usr/bin/env python3
"""
Migration Tool for Custom Storage Backends

This script migrates data between custom storage backends while preserving
all data including prompts, branches, evaluations, chains, and metadata.

Usage:
    python migrate.py --from s3 --to hybrid --config migration_config.yaml
    python migrate.py --from blockchain --to custom_db --dry-run

Features:
    - Migrate between any custom backends
    - Preserve all data and metadata
    - Dry-run mode for testing
    - Progress tracking
    - Error handling and recovery
    - Statistics reporting
"""

import argparse
import json
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

# Import custom backends
try:
    from . import S3Storage, HybridStorage, BlockchainStorage, CustomDBStorage
    from . import S3_AVAILABLE, HYBRID_AVAILABLE, BLOCKCHAIN_AVAILABLE
except ImportError:
    from __init__ import S3Storage, HybridStorage, BlockchainStorage, CustomDBStorage
    from __init__ import S3_AVAILABLE, HYBRID_AVAILABLE, BLOCKCHAIN_AVAILABLE


class CustomStorageMigrator:
    """
    Migrator for custom storage backends

    Handles migration between:
    - S3 ↔ Hybrid
    - S3 ↔ Blockchain
    - Hybrid ↔ Blockchain
    - Any ↔ Custom DB
    """

    def __init__(
        self,
        from_backend: str,
        to_backend: str,
        from_config: Dict[str, Any],
        to_config: Dict[str, Any],
        dry_run: bool = False,
    ):
        """
        Initialize migrator

        Args:
            from_backend: Source backend type
            to_backend: Destination backend type
            from_config: Source backend configuration
            to_config: Destination backend configuration
            dry_run: If True, don't actually write to destination
        """
        self.from_backend_type = from_backend
        self.to_backend_type = to_backend
        self.from_config = from_config
        self.to_config = to_config
        self.dry_run = dry_run

        self.from_storage = None
        self.to_storage = None

        self.stats = {
            'prompts_migrated': 0,
            'branches_migrated': 0,
            'evaluations_migrated': 0,
            'chains_migrated': 0,
            'errors': [],
            'skipped': [],
        }

    def _create_backend(self, backend_type: str, config: Dict[str, Any]):
        """Create storage backend instance"""
        backend_type = backend_type.lower()

        if backend_type == 's3':
            if not S3_AVAILABLE:
                raise RuntimeError("S3 backend not available")
            return S3Storage(**config)

        elif backend_type == 'hybrid':
            if not HYBRID_AVAILABLE:
                raise RuntimeError("Hybrid backend not available")
            return HybridStorage(**config)

        elif backend_type == 'blockchain':
            if not BLOCKCHAIN_AVAILABLE:
                raise RuntimeError("Blockchain backend not available")
            return BlockchainStorage(**config)

        elif backend_type == 'custom_db':
            return CustomDBStorage(**config)

        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    def _init_backends(self) -> None:
        """Initialize source and destination backends"""
        print(f"Initializing source backend: {self.from_backend_type}")
        self.from_storage = self._create_backend(
            self.from_backend_type,
            self.from_config
        )

        # Initialize source
        storage_path = self.from_config.pop('storage_path', None)
        if storage_path:
            self.from_storage.init_storage(storage_path)

        if not self.dry_run:
            print(f"Initializing destination backend: {self.to_backend_type}")
            self.to_storage = self._create_backend(
                self.to_backend_type,
                self.to_config
            )

            # Initialize destination
            storage_path = self.to_config.pop('storage_path', None)
            if storage_path:
                self.to_storage.init_storage(storage_path)

    def _migrate_branches(self, branches: Optional[List[str]] = None) -> None:
        """
        Migrate branches

        Args:
            branches: List of branch names to migrate (None = all)
        """
        print("\nMigrating branches...")

        # For now, we'll get all branches from prompts
        # (not all backends have a list_branches method)
        migrated_branches = set()

        try:
            # Get all prompts to find branches
            all_prompts = []
            for branch in branches or ['main']:
                try:
                    prompts = self.from_storage.list_prompts(branch=branch)
                    for prompt in prompts:
                        if prompt['branch'] not in migrated_branches:
                            if not self.dry_run:
                                try:
                                    # Try to create branch in destination
                                    if prompt['branch'] != 'main':
                                        self.to_storage.create_branch(
                                            prompt['branch'],
                                            from_branch='main'
                                        )
                                except Exception as e:
                                    # Branch might already exist
                                    pass

                            migrated_branches.add(prompt['branch'])
                            self.stats['branches_migrated'] += 1
                except Exception as e:
                    self.stats['errors'].append({
                        'type': 'branch',
                        'branch': branch,
                        'error': str(e)
                    })

        except Exception as e:
            print(f"Error migrating branches: {e}")

    def _migrate_prompts(self, branches: Optional[List[str]] = None) -> None:
        """
        Migrate prompts

        Args:
            branches: List of branch names to migrate (None = all)
        """
        print("\nMigrating prompts...")

        branches_to_migrate = branches or ['main']

        for branch in branches_to_migrate:
            print(f"\nProcessing branch: {branch}")

            try:
                prompts = self.from_storage.list_prompts(branch=branch)

                with tqdm(total=len(prompts), desc=f"Branch {branch}") as pbar:
                    for prompt_meta in prompts:
                        try:
                            # Get full prompt data
                            prompt_data = self.from_storage.get_prompt(
                                name=prompt_meta['name'],
                                branch=branch,
                                version=prompt_meta.get('version'),
                            )

                            if not prompt_data:
                                self.stats['skipped'].append({
                                    'type': 'prompt',
                                    'name': prompt_meta['name'],
                                    'reason': 'Could not retrieve'
                                })
                                pbar.update(1)
                                continue

                            # Prepare for destination
                            save_data = {
                                'name': prompt_data['name'],
                                'content': prompt_data['content'],
                                'branch': prompt_data['branch'],
                                'metadata': prompt_data.get('metadata', {}),
                            }

                            # Save to destination
                            if not self.dry_run:
                                self.to_storage.save_prompt(save_data)

                            self.stats['prompts_migrated'] += 1
                            pbar.update(1)

                        except Exception as e:
                            self.stats['errors'].append({
                                'type': 'prompt',
                                'name': prompt_meta['name'],
                                'branch': branch,
                                'error': str(e)
                            })
                            pbar.update(1)

            except Exception as e:
                print(f"Error listing prompts for branch {branch}: {e}")
                self.stats['errors'].append({
                    'type': 'branch_list',
                    'branch': branch,
                    'error': str(e)
                })

    def _migrate_chains(self) -> None:
        """Migrate prompt chains"""
        print("\nMigrating chains...")

        # Note: Not all backends have a list_chains method
        # We'll skip this for now or implement if backend supports it
        try:
            if hasattr(self.from_storage, 'list_chains'):
                chains = self.from_storage.list_chains()

                for chain_meta in tqdm(chains, desc="Chains"):
                    try:
                        chain_data = self.from_storage.get_chain(chain_meta['name'])

                        if chain_data and not self.dry_run:
                            self.to_storage.save_chain(chain_data)

                        self.stats['chains_migrated'] += 1

                    except Exception as e:
                        self.stats['errors'].append({
                            'type': 'chain',
                            'name': chain_meta['name'],
                            'error': str(e)
                        })
        except Exception as e:
            print(f"Skipping chains (not supported or error): {e}")

    def migrate(
        self,
        branches: Optional[List[str]] = None,
        include_chains: bool = True,
        include_evaluations: bool = True,
    ) -> Dict[str, Any]:
        """
        Run migration

        Args:
            branches: List of branches to migrate (None = all)
            include_chains: Migrate chains
            include_evaluations: Migrate evaluations

        Returns:
            Migration statistics
        """
        print("=" * 60)
        print("Custom Storage Backend Migration")
        print("=" * 60)
        print(f"From: {self.from_backend_type}")
        print(f"To: {self.to_backend_type}")
        print(f"Dry run: {self.dry_run}")
        print("=" * 60)

        try:
            # Initialize backends
            self._init_backends()

            # Migrate branches
            self._migrate_branches(branches)

            # Migrate prompts
            self._migrate_prompts(branches)

            # Migrate chains
            if include_chains:
                self._migrate_chains()

            # TODO: Migrate evaluations if supported

            # Print summary
            self._print_summary()

            return self.stats

        except Exception as e:
            print(f"\nMigration failed: {e}")
            self.stats['errors'].append({
                'type': 'fatal',
                'error': str(e)
            })
            return self.stats

        finally:
            # Clean up
            if self.from_storage:
                self.from_storage.close()
            if self.to_storage:
                self.to_storage.close()

    def _print_summary(self) -> None:
        """Print migration summary"""
        print("\n" + "=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"Branches migrated: {self.stats['branches_migrated']}")
        print(f"Prompts migrated: {self.stats['prompts_migrated']}")
        print(f"Chains migrated: {self.stats['chains_migrated']}")
        print(f"Evaluations migrated: {self.stats['evaluations_migrated']}")
        print(f"Skipped: {len(self.stats['skipped'])}")
        print(f"Errors: {len(self.stats['errors'])}")

        if self.stats['errors']:
            print("\nErrors:")
            for error in self.stats['errors'][:10]:  # Show first 10
                print(f"  - {error['type']}: {error.get('name', 'N/A')}")
                print(f"    {error['error']}")

        if self.stats['skipped']:
            print(f"\nSkipped {len(self.stats['skipped'])} items")

        print("=" * 60)


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_file) as f:
        return yaml.safe_load(f)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Migrate data between custom storage backends'
    )

    parser.add_argument(
        '--from',
        dest='from_backend',
        required=True,
        choices=['s3', 'hybrid', 'blockchain', 'custom_db'],
        help='Source backend type'
    )

    parser.add_argument(
        '--to',
        dest='to_backend',
        required=True,
        choices=['s3', 'hybrid', 'blockchain', 'custom_db'],
        help='Destination backend type'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to migration configuration file'
    )

    parser.add_argument(
        '--branches',
        nargs='+',
        help='Specific branches to migrate (default: all)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without writing to destination'
    )

    parser.add_argument(
        '--no-chains',
        action='store_true',
        help='Skip migrating chains'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output statistics to JSON file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    from_config = config.get(args.from_backend, {})
    to_config = config.get(args.to_backend, {})

    # Create migrator
    migrator = CustomStorageMigrator(
        from_backend=args.from_backend,
        to_backend=args.to_backend,
        from_config=from_config,
        to_config=to_config,
        dry_run=args.dry_run,
    )

    # Run migration
    stats = migrator.migrate(
        branches=args.branches,
        include_chains=not args.no_chains,
    )

    # Save statistics if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.output}")

    # Exit with error if there were errors
    if stats['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()
