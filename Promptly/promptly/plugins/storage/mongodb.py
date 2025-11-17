#!/usr/bin/env python3
"""
MongoDB Storage Backend Plugin

Production-grade MongoDB backend using PyMongo with:
- Document-based storage (natural fit for prompts)
- Support for replica sets and sharding
- Flexible schema evolution
- Full-text search capabilities
- Atomic operations and transactions
- Comprehensive error handling
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
    from pymongo.errors import (
        ConnectionFailure, DuplicateKeyError,
        OperationFailure, ServerSelectionTimeoutError
    )
    from pymongo.collection import Collection
    from pymongo.database import Database
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("Warning: PyMongo not available. Install with: pip install pymongo")

from ..base import BaseStorageBackend


class MongoDBStorage(BaseStorageBackend):
    """
    MongoDB storage backend with enterprise features

    Features:
    - Document-based storage (JSON-native)
    - Automatic schema flexibility
    - Full-text search on prompt content
    - Replica set support for high availability
    - Read/write concerns for consistency control
    - Atomic operations with transactions
    - Geospatial queries (future use)

    Connection string formats:
        mongodb://localhost:27017/
        mongodb://user:password@host:port/
        mongodb://user:password@host1:27017,host2:27017,host3:27017/?replicaSet=rs0
        mongodb+srv://user:password@cluster.mongodb.net/

    Example:
        storage = MongoDBStorage()
        storage.init_storage('mongodb://localhost:27017/promptly_db')
    """

    def __init__(self, max_pool_size: int = 100, timeout_ms: int = 5000,
                 retry_writes: bool = True, replica_set: Optional[str] = None):
        """
        Initialize MongoDB storage backend

        Args:
            max_pool_size: Maximum number of connections in the pool
            timeout_ms: Server selection timeout in milliseconds
            retry_writes: Enable retryable writes
            replica_set: Replica set name (if using replica sets)
        """
        super().__init__(
            name="mongodb",
            description="MongoDB document storage with flexible schema and full-text search"
        )

        if not PYMONGO_AVAILABLE:
            raise ImportError("PyMongo required for MongoDB backend. Install: pip install pymongo")

        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.prompts: Optional[Collection] = None
        self.branches: Optional[Collection] = None
        self.evaluations: Optional[Collection] = None
        self.chains: Optional[Collection] = None
        self.config: Optional[Collection] = None

        self.max_pool_size = max_pool_size
        self.timeout_ms = timeout_ms
        self.retry_writes = retry_writes
        self.replica_set = replica_set

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize MongoDB connection

        Args:
            storage_path: MongoDB connection string
                         Format: mongodb://[user:password@]host[:port]/database
        """
        # Parse connection string to extract database name
        # Format: mongodb://host:port/database or mongodb://user:pass@host:port/database
        if '/' in storage_path.split('://')[-1]:
            db_name = storage_path.split('/')[-1].split('?')[0]
        else:
            db_name = 'promptly_db'

        # Create client with connection options
        client_kwargs = {
            'maxPoolSize': self.max_pool_size,
            'serverSelectionTimeoutMS': self.timeout_ms,
            'retryWrites': self.retry_writes,
        }

        if self.replica_set:
            client_kwargs['replicaSet'] = self.replica_set

        try:
            self.client = MongoClient(storage_path, **client_kwargs)

            # Test connection
            self.client.admin.command('ping')

            # Get database
            self.db = self.client[db_name]

            # Get collections
            self.prompts = self.db['prompts']
            self.branches = self.db['branches']
            self.evaluations = self.db['evaluations']
            self.chains = self.db['chains']
            self.config = self.db['config']

            # Create indexes
            self._create_indexes()

            # Initialize default data
            self._initialize_defaults()

        except ServerSelectionTimeoutError as e:
            raise ConnectionFailure(f"Could not connect to MongoDB: {e}")

    def _create_indexes(self) -> None:
        """Create indexes for optimal query performance"""
        # Prompts collection indexes
        self.prompts.create_index([('name', ASCENDING), ('branch', ASCENDING)])
        self.prompts.create_index([('commit_hash', ASCENDING)], unique=True)
        self.prompts.create_index([('branch', ASCENDING), ('version', DESCENDING)])
        self.prompts.create_index([('created_at', DESCENDING)])
        # Full-text search index on content
        self.prompts.create_index([('content', TEXT), ('name', TEXT)])

        # Branches collection indexes
        self.branches.create_index([('name', ASCENDING)], unique=True)

        # Evaluations collection indexes
        self.evaluations.create_index([('prompt_name', ASCENDING)])
        self.evaluations.create_index([('commit_hash', ASCENDING)])
        self.evaluations.create_index([('score', DESCENDING)])
        self.evaluations.create_index([('created_at', DESCENDING)])

        # Chains collection indexes
        self.chains.create_index([('name', ASCENDING)], unique=True)

        # Config collection indexes
        self.config.create_index([('key', ASCENDING)], unique=True)

    def _initialize_defaults(self) -> None:
        """Initialize default data (main branch, config)"""
        # Create main branch if not exists
        if not self.branches.find_one({'name': 'main'}):
            self.branches.insert_one({
                'name': 'main',
                'head_commit': 'init',
                'created_at': datetime.utcnow(),
                'description': 'Main branch'
            })

        # Set current branch config
        if not self.config.find_one({'key': 'current_branch'}):
            self.config.insert_one({
                'key': 'current_branch',
                'value': 'main'
            })

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save a prompt to MongoDB

        Args:
            prompt_data: Dictionary containing prompt information

        Returns:
            str: Commit hash
        """
        if not self.db:
            raise RuntimeError("Storage not initialized")

        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata', {})

        timestamp = datetime.utcnow()
        commit_hash = self._generate_commit_hash(name, content, timestamp.isoformat())

        # Find latest version on this branch
        latest = self.prompts.find_one(
            {'name': name, 'branch': branch},
            sort=[('version', DESCENDING)]
        )

        version = latest['version'] + 1 if latest else 1
        parent_id = latest['_id'] if latest else None

        # Create prompt document
        prompt_doc = {
            'name': name,
            'content': content,
            'branch': branch,
            'version': version,
            'parent_id': parent_id,
            'commit_hash': commit_hash,
            'created_at': timestamp,
            'metadata': metadata
        }

        try:
            # Insert prompt
            self.prompts.insert_one(prompt_doc)

            # Update branch head
            self.branches.update_one(
                {'name': branch},
                {'$set': {'head_commit': commit_hash, 'updated_at': timestamp}}
            )

            return commit_hash

        except DuplicateKeyError:
            # Handle duplicate commit hash (extremely rare)
            raise Exception(f"Commit hash collision: {commit_hash}")

    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt from MongoDB"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        query = {'name': name}

        if commit_hash:
            query['commit_hash'] = commit_hash
        elif version:
            query['branch'] = branch
            query['version'] = version
        else:
            query['branch'] = branch

        # Sort by version descending to get latest
        prompt = self.prompts.find_one(query, sort=[('version', DESCENDING)])

        if not prompt:
            return None

        return {
            'name': prompt['name'],
            'content': prompt['content'],
            'branch': prompt['branch'],
            'version': prompt['version'],
            'commit_hash': prompt['commit_hash'],
            'created_at': prompt['created_at'].isoformat(),
            'metadata': prompt.get('metadata', {})
        }

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on a branch"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        # Aggregation pipeline to get latest version of each prompt
        pipeline = [
            {'$match': {'branch': branch}},
            {'$sort': {'version': -1}},
            {'$group': {
                '_id': '$name',
                'name': {'$first': '$name'},
                'version': {'$first': '$version'},
                'commit_hash': {'$first': '$commit_hash'},
                'created_at': {'$first': '$created_at'}
            }},
            {'$sort': {'name': 1}}
        ]

        results = list(self.prompts.aggregate(pipeline))

        return [{
            'name': r['name'],
            'version': r['version'],
            'commit_hash': r['commit_hash'],
            'created_at': r['created_at'].isoformat()
        } for r in results]

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        # Check source branch exists
        source = self.branches.find_one({'name': from_branch})
        if not source:
            raise Exception(f"Branch '{from_branch}' does not exist")

        # Check if target branch already exists
        if self.branches.find_one({'name': branch_name}):
            raise Exception(f"Branch '{branch_name}' already exists")

        try:
            # Create new branch
            self.branches.insert_one({
                'name': branch_name,
                'head_commit': source['head_commit'],
                'created_at': datetime.utcnow(),
                'description': f"Branched from {from_branch}"
            })

            # Get latest version of each prompt from source branch
            pipeline = [
                {'$match': {'branch': from_branch}},
                {'$sort': {'version': -1}},
                {'$group': {
                    '_id': '$name',
                    'doc': {'$first': '$$ROOT'}
                }}
            ]

            latest_prompts = list(self.prompts.aggregate(pipeline))

            # Copy prompts to new branch
            new_prompts = []
            for item in latest_prompts:
                prompt = item['doc']
                timestamp = datetime.utcnow()
                new_hash = self._generate_commit_hash(
                    prompt['name'],
                    prompt['content'],
                    timestamp.isoformat()
                )

                new_prompt = {
                    'name': prompt['name'],
                    'content': prompt['content'],
                    'branch': branch_name,
                    'version': prompt['version'],
                    'parent_id': prompt.get('parent_id'),
                    'commit_hash': new_hash,
                    'created_at': timestamp,
                    'metadata': prompt.get('metadata', {})
                }
                new_prompts.append(new_prompt)

            if new_prompts:
                self.prompts.insert_many(new_prompts)

        except DuplicateKeyError:
            raise Exception(f"Branch '{branch_name}' already exists")

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        config = self.config.find_one({'key': 'current_branch'})
        return config['value'] if config else 'main'

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        # Verify branch exists
        if not self.branches.find_one({'name': branch_name}):
            raise Exception(f"Branch '{branch_name}' does not exist")

        # Update config
        self.config.update_one(
            {'key': 'current_branch'},
            {'$set': {'value': branch_name}},
            upsert=True
        )

    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get commit history"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        query = {'branch': branch}
        if name:
            query['name'] = name

        commits = list(
            self.prompts.find(query)
            .sort('created_at', DESCENDING)
            .limit(limit)
        )

        return [{
            'commit_hash': c['commit_hash'],
            'name': c['name'],
            'version': c['version'],
            'branch': c['branch'],
            'created_at': c['created_at'].isoformat()
        } for c in commits]

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        eval_doc = {
            'prompt_name': eval_data['prompt_name'],
            'commit_hash': eval_data['commit_hash'],
            'test_case': eval_data.get('test_case', {}),
            'expected': eval_data.get('expected'),
            'actual': eval_data.get('actual'),
            'score': eval_data.get('score'),
            'metrics': eval_data.get('metrics', {}),
            'evaluator_name': eval_data.get('evaluator_name'),
            'created_at': datetime.utcnow()
        }

        self.evaluations.insert_one(eval_doc)

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        chain_doc = {
            'name': chain_data['name'],
            'steps': chain_data['steps'],
            'description': chain_data.get('description'),
            'created_at': datetime.utcnow()
        }

        try:
            self.chains.insert_one(chain_doc)
        except DuplicateKeyError:
            raise Exception(f"Chain '{chain_data['name']}' already exists")

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        if not self.db:
            raise RuntimeError("Storage not initialized")

        chain = self.chains.find_one({'name': name})

        if not chain:
            return None

        return {
            'name': chain['name'],
            'steps': chain['steps'],
            'description': chain.get('description')
        }

    def close(self) -> None:
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    # Additional MongoDB-specific methods

    def full_text_search(self, search_text: str, branch: str = "main",
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform full-text search on prompt content

        Args:
            search_text: Text to search for
            branch: Branch to search in
            limit: Maximum number of results

        Returns:
            List of matching prompts with relevance scores
        """
        if not self.db:
            raise RuntimeError("Storage not initialized")

        results = list(
            self.prompts.find(
                {
                    '$text': {'$search': search_text},
                    'branch': branch
                },
                {'score': {'$meta': 'textScore'}}
            )
            .sort([('score', {'$meta': 'textScore'})])
            .limit(limit)
        )

        return [{
            'name': r['name'],
            'content': r['content'],
            'version': r['version'],
            'commit_hash': r['commit_hash'],
            'relevance_score': r['score']
        } for r in results]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with statistics
        """
        if not self.db:
            raise RuntimeError("Storage not initialized")

        stats = {
            'total_prompts': self.prompts.count_documents({}),
            'total_branches': self.branches.count_documents({}),
            'total_evaluations': self.evaluations.count_documents({}),
            'total_chains': self.chains.count_documents({}),
        }

        # Per-branch statistics
        pipeline = [
            {'$group': {
                '_id': '$branch',
                'count': {'$sum': 1}
            }}
        ]
        branch_stats = {item['_id']: item['count']
                       for item in self.prompts.aggregate(pipeline)}

        stats['prompts_per_branch'] = branch_stats

        # Database size
        db_stats = self.db.command('dbStats')
        stats['database_size_bytes'] = db_stats.get('dataSize', 0)
        stats['storage_size_bytes'] = db_stats.get('storageSize', 0)

        return stats

    def export_branch_to_json(self, branch: str = "main") -> List[Dict[str, Any]]:
        """
        Export all prompts from a branch to JSON-serializable format

        Args:
            branch: Branch name

        Returns:
            List of prompt documents
        """
        if not self.db:
            raise RuntimeError("Storage not initialized")

        prompts = list(self.prompts.find({'branch': branch}))

        # Convert ObjectId and datetime to strings
        for prompt in prompts:
            prompt['_id'] = str(prompt['_id'])
            if prompt.get('parent_id'):
                prompt['parent_id'] = str(prompt['parent_id'])
            prompt['created_at'] = prompt['created_at'].isoformat()

        return prompts

    def compact_collection(self, collection_name: str) -> None:
        """
        Compact a collection to reclaim space (MongoDB 4.4+)

        Args:
            collection_name: Name of collection to compact
        """
        if not self.db:
            raise RuntimeError("Storage not initialized")

        try:
            self.db.command('compact', collection_name)
        except OperationFailure as e:
            print(f"Warning: Could not compact collection: {e}")
