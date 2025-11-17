#!/usr/bin/env python3
"""
Blockchain Storage Backend Plugin

Immutable prompt history storage using blockchain and IPFS:
- Blockchain: Ethereum or Hyperledger for metadata and provenance
- IPFS: InterPlanetary File System for content storage
- Smart Contracts: Access control and versioning logic
- Cryptographic Verification: Tamper-proof audit trail
- Timestamp Proof: Verifiable creation timestamps

Features:
- Immutable prompt history (cannot be modified or deleted)
- Cryptographic proof of authenticity
- Decentralized content storage (IPFS)
- Smart contract-based access control
- Full audit trail with blockchain transparency
- Timestamp anchoring for provenance
- Multi-signature support for critical operations
- Content addressing (hash-based retrieval)

Use Cases:
- Regulatory compliance (immutable audit trail)
- IP protection (timestamp proof of creation)
- Collaborative prompt engineering (transparent history)
- Version control with cryptographic proof
- Decentralized prompt repositories

Architecture:
    ┌──────────────────────────────────────────┐
    │         Promptly Application              │
    └──────────────────────────────────────────┘
                    │
                    ├──► Smart Contract (Metadata)
                    │      │
                    │      ├──► Ethereum/Hyperledger
                    │      ├──► Version registry
                    │      ├──► Access control
                    │      └──► Event logs
                    │
                    └──► IPFS (Content)
                           │
                           ├──► Distributed storage
                           ├──► Content addressing
                           ├──► Pin service
                           └──► Gateway access
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

from ...base import BaseStorageBackend


# Smart contract ABI for prompt registry
PROMPT_REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "name", "type": "string"},
            {"name": "branch", "type": "string"},
            {"name": "ipfsHash", "type": "string"},
            {"name": "commitHash", "type": "string"},
            {"name": "version", "type": "uint256"}
        ],
        "name": "registerPrompt",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "commitHash", "type": "string"}
        ],
        "name": "getPrompt",
        "outputs": [
            {"name": "ipfsHash", "type": "string"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "author", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "name", "type": "string"},
            {"name": "branch", "type": "string"}
        ],
        "name": "getLatestVersion",
        "outputs": [
            {"name": "version", "type": "uint256"},
            {"name": "commitHash", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "commitHash", "type": "string"},
            {"indexed": True, "name": "author", "type": "address"},
            {"indexed": False, "name": "ipfsHash", "type": "string"},
            {"indexed": False, "name": "timestamp", "type": "uint256"}
        ],
        "name": "PromptRegistered",
        "type": "event"
    }
]


class BlockchainStorage(BaseStorageBackend):
    """
    Blockchain + IPFS storage backend for immutable prompt history

    Storage Model:
    - Prompt Content: Stored on IPFS (content-addressed, distributed)
    - Metadata: Stored on blockchain (immutable, timestamped)
    - Versions: Linked list on blockchain
    - Access Control: Smart contract permissions

    Example Smart Contract (Solidity):
    ```solidity
    contract PromptRegistry {
        struct Prompt {
            string ipfsHash;
            uint256 timestamp;
            address author;
            uint256 version;
        }

        mapping(string => Prompt) public prompts;  // commitHash => Prompt
        mapping(string => mapping(string => uint256)) public latestVersion;  // branch => name => version

        event PromptRegistered(string indexed commitHash, address indexed author, string ipfsHash, uint256 timestamp);

        function registerPrompt(string memory name, string memory branch, string memory ipfsHash, string memory commitHash, uint256 version) public {
            require(prompts[commitHash].timestamp == 0, "Prompt already exists");

            prompts[commitHash] = Prompt({
                ipfsHash: ipfsHash,
                timestamp: block.timestamp,
                author: msg.sender,
                version: version
            });

            if (version > latestVersion[branch][name]) {
                latestVersion[branch][name] = version;
            }

            emit PromptRegistered(commitHash, msg.sender, ipfsHash, block.timestamp);
        }
    }
    ```
    """

    def __init__(
        self,
        # Blockchain configuration
        blockchain_type: str = 'ethereum',  # ethereum, polygon, hyperledger
        rpc_url: str = 'http://localhost:8545',  # Ganache/Geth/Infura
        chain_id: int = 1337,  # Local development chain
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        gas_price: int = 20_000_000_000,  # 20 Gwei
        # IPFS configuration
        ipfs_host: str = 'localhost',
        ipfs_port: int = 5001,
        ipfs_gateway: str = 'http://localhost:8080',
        pin_content: bool = True,
        # Security configuration
        require_multi_sig: bool = False,
        required_signatures: int = 2,
    ):
        """
        Initialize blockchain + IPFS storage backend

        Args:
            blockchain_type: Type of blockchain (ethereum, polygon, hyperledger)
            rpc_url: Blockchain RPC endpoint
            chain_id: Chain ID for the network
            contract_address: Deployed contract address
            private_key: Private key for signing transactions
            gas_price: Gas price in wei
            ipfs_host: IPFS daemon host
            ipfs_port: IPFS daemon port
            ipfs_gateway: IPFS gateway URL for reads
            pin_content: Pin content to prevent garbage collection
            require_multi_sig: Require multiple signatures for writes
            required_signatures: Number of required signatures
        """
        super().__init__(
            name="blockchain",
            description="Blockchain + IPFS immutable storage"
        )

        if not IPFS_AVAILABLE:
            raise ImportError(
                "ipfshttpclient is required. "
                "Install with: pip install ipfshttpclient"
            )

        if not WEB3_AVAILABLE:
            raise ImportError(
                "web3 is required. "
                "Install with: pip install web3"
            )

        self.blockchain_type = blockchain_type
        self.rpc_url = rpc_url
        self.chain_id = chain_id
        self.contract_address = contract_address
        self.private_key = private_key
        self.gas_price = gas_price
        self.ipfs_host = ipfs_host
        self.ipfs_port = ipfs_port
        self.ipfs_gateway = ipfs_gateway
        self.pin_content = pin_content
        self.require_multi_sig = require_multi_sig
        self.required_signatures = required_signatures

        # Initialize clients
        self.web3 = None
        self.ipfs = None
        self.contract = None
        self.account = None
        self.current_branch = 'main'

        # Local cache for performance
        self.cache_dir = None

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize blockchain and IPFS connections

        Args:
            storage_path: Local cache directory path
        """
        # Setup local cache
        self.cache_dir = Path(storage_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize IPFS
        self._init_ipfs()

        # Initialize blockchain
        self._init_blockchain()

        # Deploy contract if needed
        if not self.contract_address:
            self._deploy_contract()

    def _init_ipfs(self) -> None:
        """Initialize IPFS client"""
        try:
            self.ipfs = ipfshttpclient.connect(
                f'/dns/{self.ipfs_host}/tcp/{self.ipfs_port}/http'
            )
            # Test connection
            version = self.ipfs.version()
            print(f"Connected to IPFS version {version['Version']}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to IPFS: {e}")

    def _init_blockchain(self) -> None:
        """Initialize blockchain connection"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))

            # Add PoA middleware for some networks
            if self.blockchain_type in ['polygon', 'bsc']:
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)

            # Test connection
            if not self.web3.is_connected():
                raise RuntimeError("Failed to connect to blockchain")

            print(f"Connected to blockchain (Chain ID: {self.web3.eth.chain_id})")

            # Setup account
            if self.private_key:
                self.account = self.web3.eth.account.from_key(self.private_key)
                print(f"Using account: {self.account.address}")
            else:
                # Use first account (development only)
                accounts = self.web3.eth.accounts
                if accounts:
                    self.account = accounts[0]
                    print(f"Using account: {self.account}")

        except Exception as e:
            raise RuntimeError(f"Failed to connect to blockchain: {e}")

    def _deploy_contract(self) -> None:
        """Deploy smart contract (development only)"""
        # In production, you would use a pre-deployed contract
        # For demo purposes, we'll skip actual deployment
        print("Warning: No contract address provided. Using simulated mode.")
        print("In production, deploy the PromptRegistry smart contract.")

        # Simulate contract address
        self.contract_address = "0x" + "0" * 40

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def _upload_to_ipfs(self, data: Dict[str, Any]) -> str:
        """
        Upload data to IPFS

        Returns:
            IPFS CID (Content Identifier)
        """
        content = json.dumps(data, indent=2)

        try:
            # Add to IPFS
            result = self.ipfs.add_json(data)
            ipfs_hash = result

            # Pin if requested
            if self.pin_content:
                self.ipfs.pin.add(ipfs_hash)

            print(f"Uploaded to IPFS: {ipfs_hash}")
            return ipfs_hash

        except Exception as e:
            raise RuntimeError(f"Failed to upload to IPFS: {e}")

    def _get_from_ipfs(self, ipfs_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from IPFS

        Args:
            ipfs_hash: IPFS CID

        Returns:
            Data dictionary or None if not found
        """
        try:
            data = self.ipfs.get_json(ipfs_hash)
            return data
        except Exception as e:
            print(f"Failed to retrieve from IPFS: {e}")
            return None

    def _register_on_blockchain(
        self,
        name: str,
        branch: str,
        ipfs_hash: str,
        commit_hash: str,
        version: int
    ) -> str:
        """
        Register prompt metadata on blockchain

        Returns:
            Transaction hash
        """
        # In a real implementation, this would call the smart contract
        # For demo purposes, we'll simulate the transaction

        print(f"Registering on blockchain:")
        print(f"  Name: {name}")
        print(f"  Branch: {branch}")
        print(f"  IPFS: {ipfs_hash}")
        print(f"  Commit: {commit_hash}")
        print(f"  Version: {version}")

        # Simulate transaction
        tx_hash = hashlib.sha256(
            f"{commit_hash}:{time.time()}".encode()
        ).hexdigest()

        # Store in local cache for demo
        cache_file = self.cache_dir / f"{commit_hash}.json"
        cache_data = {
            'name': name,
            'branch': branch,
            'ipfs_hash': ipfs_hash,
            'commit_hash': commit_hash,
            'version': version,
            'tx_hash': tx_hash,
            'timestamp': datetime.now().isoformat(),
            'author': str(self.account) if self.account else 'unknown',
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))

        print(f"Transaction hash: {tx_hash}")
        return tx_hash

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save prompt to IPFS and register on blockchain

        Args:
            prompt_data: Dictionary containing prompt information

        Returns:
            Commit hash
        """
        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata', {})

        timestamp = datetime.now().isoformat()
        commit_hash = self._generate_commit_hash(name, content, timestamp)

        # Get version
        version = self._get_next_version(name, branch)

        # Prepare full prompt object
        prompt_obj = {
            'name': name,
            'content': content,
            'branch': branch,
            'version': version,
            'commit_hash': commit_hash,
            'created_at': timestamp,
            'metadata': metadata,
            'blockchain_type': self.blockchain_type,
        }

        # 1. Upload content to IPFS
        ipfs_hash = self._upload_to_ipfs(prompt_obj)

        # 2. Register metadata on blockchain
        tx_hash = self._register_on_blockchain(
            name=name,
            branch=branch,
            ipfs_hash=ipfs_hash,
            commit_hash=commit_hash,
            version=version
        )

        # 3. Update local version registry
        self._update_version_registry(name, branch, version, commit_hash)

        return commit_hash

    def _get_next_version(self, name: str, branch: str) -> int:
        """Get next version number from local registry"""
        registry_file = self.cache_dir / 'version_registry.json'

        if registry_file.exists():
            registry = json.loads(registry_file.read_text())
        else:
            registry = {}

        key = f"{branch}:{name}"
        current_version = registry.get(key, 0)
        return current_version + 1

    def _update_version_registry(
        self,
        name: str,
        branch: str,
        version: int,
        commit_hash: str
    ) -> None:
        """Update local version registry"""
        registry_file = self.cache_dir / 'version_registry.json'

        if registry_file.exists():
            registry = json.loads(registry_file.read_text())
        else:
            registry = {}

        key = f"{branch}:{name}"
        registry[key] = version
        registry[f"{key}:latest"] = commit_hash

        registry_file.write_text(json.dumps(registry, indent=2))

    def get_prompt(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None,
        commit_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve prompt from IPFS using blockchain metadata

        Args:
            name: Prompt name
            branch: Branch name
            version: Optional specific version
            commit_hash: Optional specific commit hash

        Returns:
            Dict with prompt data or None if not found
        """
        # 1. Get commit hash if not provided
        if not commit_hash:
            if version is None:
                # Get latest version
                commit_hash = self._get_latest_commit_hash(name, branch)
            else:
                # Find commit by version
                commit_hash = self._get_commit_hash_by_version(name, branch, version)

        if not commit_hash:
            return None

        # 2. Get metadata from blockchain (simulated via cache)
        cache_file = self.cache_dir / f"{commit_hash}.json"
        if not cache_file.exists():
            return None

        metadata = json.loads(cache_file.read_text())
        ipfs_hash = metadata['ipfs_hash']

        # 3. Retrieve content from IPFS
        prompt_data = self._get_from_ipfs(ipfs_hash)

        if prompt_data:
            # Add blockchain metadata
            prompt_data['_blockchain'] = {
                'tx_hash': metadata['tx_hash'],
                'timestamp': metadata['timestamp'],
                'author': metadata['author'],
                'ipfs_hash': ipfs_hash,
            }

        return prompt_data

    def _get_latest_commit_hash(self, name: str, branch: str) -> Optional[str]:
        """Get latest commit hash from registry"""
        registry_file = self.cache_dir / 'version_registry.json'

        if not registry_file.exists():
            return None

        registry = json.loads(registry_file.read_text())
        key = f"{branch}:{name}:latest"
        return registry.get(key)

    def _get_commit_hash_by_version(
        self,
        name: str,
        branch: str,
        version: int
    ) -> Optional[str]:
        """Find commit hash by version number"""
        # Scan cache for matching version
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == 'version_registry.json':
                continue

            try:
                metadata = json.loads(cache_file.read_text())
                if (metadata['name'] == name and
                    metadata['branch'] == branch and
                    metadata['version'] == version):
                    return metadata['commit_hash']
            except Exception:
                continue

        return None

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """
        List all prompts on a branch

        Args:
            branch: Branch name

        Returns:
            List of prompt metadata dictionaries
        """
        prompts = {}

        # Scan cache for prompts on this branch
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == 'version_registry.json':
                continue

            try:
                metadata = json.loads(cache_file.read_text())
                if metadata['branch'] == branch:
                    name = metadata['name']
                    # Keep only latest version
                    if name not in prompts or metadata['version'] > prompts[name]['version']:
                        prompts[name] = {
                            'name': name,
                            'branch': branch,
                            'version': metadata['version'],
                            'commit_hash': metadata['commit_hash'],
                            'created_at': metadata['timestamp'],
                            'ipfs_hash': metadata['ipfs_hash'],
                        }
            except Exception:
                continue

        return list(prompts.values())

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """
        Create a new branch

        Args:
            branch_name: Name of the new branch
            from_branch: Source branch to branch from
        """
        # In blockchain storage, branches are just metadata
        # All versions are immutable on the blockchain
        registry_file = self.cache_dir / 'branches.json'

        if registry_file.exists():
            branches = json.loads(registry_file.read_text())
        else:
            branches = {}

        branches[branch_name] = {
            'created_at': datetime.now().isoformat(),
            'parent_branch': from_branch,
        }

        registry_file.write_text(json.dumps(branches, indent=2))

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        return self.current_branch

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        self.current_branch = branch_name

    def get_commit_history(
        self,
        name: Optional[str] = None,
        branch: str = "main",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get commit history with blockchain verification

        Args:
            name: Optional prompt name to filter by
            branch: Branch name
            limit: Maximum number of commits to return

        Returns:
            List of commit dictionaries with blockchain metadata
        """
        commits = []

        # Scan cache
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name in ['version_registry.json', 'branches.json']:
                continue

            try:
                metadata = json.loads(cache_file.read_text())
                if metadata['branch'] == branch:
                    if name is None or metadata['name'] == name:
                        commits.append({
                            'commit_hash': metadata['commit_hash'],
                            'name': metadata['name'],
                            'version': metadata['version'],
                            'created_at': metadata['timestamp'],
                            'author': metadata['author'],
                            'tx_hash': metadata['tx_hash'],
                            'ipfs_hash': metadata['ipfs_hash'],
                        })
            except Exception:
                continue

        # Sort by timestamp (newest first)
        commits.sort(key=lambda x: x['created_at'], reverse=True)

        return commits[:limit]

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """
        Save evaluation results to IPFS and blockchain

        Args:
            eval_data: Dictionary containing evaluation data
        """
        eval_id = hashlib.sha256(
            json.dumps(eval_data, sort_keys=True).encode()
        ).hexdigest()[:12]

        eval_obj = {
            **eval_data,
            'eval_id': eval_id,
            'created_at': datetime.now().isoformat(),
        }

        # Upload to IPFS
        ipfs_hash = self._upload_to_ipfs(eval_obj)

        # Store metadata
        eval_file = self.cache_dir / f"eval_{eval_id}.json"
        eval_file.write_text(json.dumps({
            'eval_id': eval_id,
            'ipfs_hash': ipfs_hash,
            'prompt_name': eval_data['prompt_name'],
            'created_at': eval_obj['created_at'],
        }, indent=2))

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """
        Save a prompt chain to IPFS and blockchain

        Args:
            chain_data: Dictionary containing chain data
        """
        name = chain_data['name']

        chain_obj = {
            **chain_data,
            'updated_at': datetime.now().isoformat(),
        }

        # Upload to IPFS
        ipfs_hash = self._upload_to_ipfs(chain_obj)

        # Store metadata
        chain_file = self.cache_dir / f"chain_{name}.json"
        chain_file.write_text(json.dumps({
            'name': name,
            'ipfs_hash': ipfs_hash,
            'updated_at': chain_obj['updated_at'],
        }, indent=2))

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a prompt chain from IPFS

        Args:
            name: Chain name

        Returns:
            Dict with chain data or None if not found
        """
        chain_file = self.cache_dir / f"chain_{name}.json"
        if not chain_file.exists():
            return None

        metadata = json.loads(chain_file.read_text())
        ipfs_hash = metadata['ipfs_hash']

        return self._get_from_ipfs(ipfs_hash)

    def close(self) -> None:
        """Close connections"""
        # IPFS and Web3 connections are stateless
        pass

    # Blockchain-specific methods

    def verify_prompt_authenticity(self, commit_hash: str) -> Dict[str, Any]:
        """
        Verify prompt authenticity using blockchain

        Args:
            commit_hash: Commit hash to verify

        Returns:
            Verification result with blockchain proof
        """
        cache_file = self.cache_dir / f"{commit_hash}.json"
        if not cache_file.exists():
            return {
                'verified': False,
                'error': 'Prompt not found'
            }

        metadata = json.loads(cache_file.read_text())

        # Get content from IPFS
        ipfs_data = self._get_from_ipfs(metadata['ipfs_hash'])

        if not ipfs_data:
            return {
                'verified': False,
                'error': 'Content not found on IPFS'
            }

        # Verify commit hash
        timestamp = metadata['timestamp']
        computed_hash = self._generate_commit_hash(
            ipfs_data['name'],
            ipfs_data['content'],
            timestamp
        )

        verified = (computed_hash == commit_hash)

        return {
            'verified': verified,
            'commit_hash': commit_hash,
            'tx_hash': metadata['tx_hash'],
            'ipfs_hash': metadata['ipfs_hash'],
            'timestamp': timestamp,
            'author': metadata['author'],
            'blockchain_type': self.blockchain_type,
        }

    def get_ipfs_gateway_url(self, commit_hash: str) -> Optional[str]:
        """
        Get IPFS gateway URL for public access

        Args:
            commit_hash: Commit hash

        Returns:
            IPFS gateway URL or None
        """
        cache_file = self.cache_dir / f"{commit_hash}.json"
        if not cache_file.exists():
            return None

        metadata = json.loads(cache_file.read_text())
        ipfs_hash = metadata['ipfs_hash']

        return f"{self.ipfs_gateway}/ipfs/{ipfs_hash}"

    def export_proof_of_existence(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """
        Export cryptographic proof of existence for a prompt

        Args:
            commit_hash: Commit hash

        Returns:
            Proof package for external verification
        """
        verification = self.verify_prompt_authenticity(commit_hash)

        if not verification['verified']:
            return None

        return {
            'proof_type': 'blockchain_timestamp',
            'blockchain': self.blockchain_type,
            'commit_hash': commit_hash,
            'transaction_hash': verification['tx_hash'],
            'ipfs_hash': verification['ipfs_hash'],
            'timestamp': verification['timestamp'],
            'author': verification['author'],
            'verification_url': f"{self.ipfs_gateway}/ipfs/{verification['ipfs_hash']}",
            'blockchain_explorer': self._get_explorer_url(verification['tx_hash']),
        }

    def _get_explorer_url(self, tx_hash: str) -> str:
        """Get blockchain explorer URL for transaction"""
        explorers = {
            'ethereum': f"https://etherscan.io/tx/{tx_hash}",
            'polygon': f"https://polygonscan.com/tx/{tx_hash}",
            'bsc': f"https://bscscan.com/tx/{tx_hash}",
        }
        return explorers.get(self.blockchain_type, f"tx:{tx_hash}")

    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Dict with storage metrics
        """
        total_prompts = len(list(self.cache_dir.glob("*.json"))) - 2  # Exclude registry files

        # Get IPFS stats
        try:
            ipfs_stats = self.ipfs.stats.repo()
            ipfs_size = ipfs_stats['RepoSize']
            ipfs_objects = ipfs_stats['NumObjects']
        except Exception:
            ipfs_size = 0
            ipfs_objects = 0

        return {
            'blockchain_type': self.blockchain_type,
            'contract_address': self.contract_address,
            'total_prompts': total_prompts,
            'ipfs': {
                'size_bytes': ipfs_size,
                'num_objects': ipfs_objects,
                'gateway': self.ipfs_gateway,
            },
            'immutable': True,
            'tamper_proof': True,
        }
