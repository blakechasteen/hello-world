#!/usr/bin/env python3
"""
S3/MinIO Storage Backend Plugin

Production-grade S3-compatible object storage backend with:
- Object versioning for prompt history
- Metadata as S3 tags and object metadata
- Lifecycle policies for automatic archiving
- CloudFront CDN integration for fast reads
- Presigned URLs for secure access
- Cross-region replication support
- Connection pooling and retries
- Comprehensive error handling

Supports both AWS S3 and MinIO (self-hosted S3-compatible storage).
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from contextlib import contextmanager

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    from botocore.config import Config as BotoConfig
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("Warning: boto3 not available. Install with: pip install boto3")

from ...base import BaseStorageBackend


class S3Storage(BaseStorageBackend):
    """
    S3/MinIO-based storage backend for Promptly

    Features:
    - Automatic versioning with S3 object versions
    - Metadata storage using S3 tags and object metadata
    - Lifecycle policies for old version archiving
    - Presigned URLs for temporary access
    - CloudFront CDN integration
    - Cross-region replication
    - Connection pooling and automatic retries

    Storage Structure:
        s3://bucket/
        ├── prompts/{branch}/{name}/
        │   ├── v{N}.json           # Versioned prompt content
        │   └── metadata.json        # Prompt metadata
        ├── branches/
        │   └── {branch}.json        # Branch metadata
        ├── evaluations/{name}/
        │   └── {eval_id}.json       # Evaluation results
        └── chains/
            └── {name}.json          # Chain definitions
    """

    def __init__(
        self,
        region: str = 'us-east-1',
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        cloudfront_domain: Optional[str] = None,
        enable_versioning: bool = True,
        enable_encryption: bool = True,
        storage_class: str = 'STANDARD',
        lifecycle_days: int = 90,
    ):
        """
        Initialize S3/MinIO storage backend

        Args:
            region: AWS region (e.g., 'us-east-1', 'eu-west-1')
            endpoint_url: Custom endpoint URL (for MinIO, e.g., 'http://localhost:9000')
            aws_access_key_id: AWS access key (or MinIO access key)
            aws_secret_access_key: AWS secret key (or MinIO secret key)
            max_retries: Maximum number of retry attempts
            timeout: Connection timeout in seconds
            cloudfront_domain: CloudFront domain for CDN reads (optional)
            enable_versioning: Enable S3 object versioning
            enable_encryption: Enable server-side encryption
            storage_class: S3 storage class (STANDARD, STANDARD_IA, GLACIER, etc.)
            lifecycle_days: Days before transitioning to cheaper storage
        """
        super().__init__(
            name="s3",
            description="S3/MinIO object storage with versioning and CDN"
        )

        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install boto3"
            )

        self.region = region
        self.endpoint_url = endpoint_url
        self.cloudfront_domain = cloudfront_domain
        self.enable_versioning = enable_versioning
        self.enable_encryption = enable_encryption
        self.storage_class = storage_class
        self.lifecycle_days = lifecycle_days

        # Configure boto3 with retries and timeouts
        boto_config = BotoConfig(
            region_name=region,
            retries={'max_attempts': max_retries, 'mode': 'adaptive'},
            connect_timeout=timeout,
            read_timeout=timeout,
            max_pool_connections=50,
        )

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=boto_config,
        )

        # Initialize S3 resource for higher-level operations
        self.s3_resource = boto3.resource(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=boto_config,
        )

        self.bucket_name = None
        self.bucket = None
        self.current_branch = 'main'

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize S3 bucket

        Args:
            storage_path: S3 bucket name or s3://bucket-name format

        Example:
            storage.init_storage('my-promptly-bucket')
            storage.init_storage('s3://my-promptly-bucket')
        """
        # Parse bucket name from path
        if storage_path.startswith('s3://'):
            parsed = urlparse(storage_path)
            self.bucket_name = parsed.netloc
        else:
            self.bucket_name = storage_path

        self.bucket = self.s3_resource.Bucket(self.bucket_name)

        # Create bucket if it doesn't exist
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                except ClientError as create_error:
                    raise RuntimeError(f"Failed to create bucket: {create_error}")
            else:
                raise RuntimeError(f"Failed to access bucket: {e}")

        # Enable versioning if requested
        if self.enable_versioning:
            try:
                self.s3_client.put_bucket_versioning(
                    Bucket=self.bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
            except ClientError as e:
                print(f"Warning: Failed to enable versioning: {e}")

        # Enable server-side encryption if requested
        if self.enable_encryption:
            try:
                self.s3_client.put_bucket_encryption(
                    Bucket=self.bucket_name,
                    ServerSideEncryptionConfiguration={
                        'Rules': [{
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            },
                            'BucketKeyEnabled': True
                        }]
                    }
                )
            except ClientError as e:
                print(f"Warning: Failed to enable encryption: {e}")

        # Configure lifecycle policies for archiving old versions
        self._configure_lifecycle_policy()

        # Initialize main branch
        self._ensure_branch_exists('main')

    def _configure_lifecycle_policy(self) -> None:
        """Configure lifecycle policies for automatic archiving"""
        try:
            lifecycle_config = {
                'Rules': [
                    {
                        'ID': 'archive-old-versions',
                        'Status': 'Enabled',
                        'Prefix': 'prompts/',
                        'NoncurrentVersionTransitions': [
                            {
                                'NoncurrentDays': self.lifecycle_days,
                                'StorageClass': 'STANDARD_IA'  # Infrequent Access
                            },
                            {
                                'NoncurrentDays': self.lifecycle_days * 2,
                                'StorageClass': 'GLACIER'  # Long-term archive
                            }
                        ],
                        'NoncurrentVersionExpiration': {
                            'NoncurrentDays': self.lifecycle_days * 4  # Delete after 1 year
                        }
                    }
                ]
            }
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
        except ClientError as e:
            print(f"Warning: Failed to configure lifecycle policy: {e}")

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def _ensure_branch_exists(self, branch_name: str) -> None:
        """Ensure branch metadata exists"""
        branch_key = f"branches/{branch_name}.json"
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=branch_key)
        except ClientError:
            # Branch doesn't exist, create it
            branch_data = {
                'name': branch_name,
                'created_at': datetime.now().isoformat(),
                'head_commit': 'init',
            }
            self._put_object(
                branch_key,
                json.dumps(branch_data, indent=2),
                content_type='application/json',
                metadata={'branch': branch_name}
            )

    def _put_object(
        self,
        key: str,
        body: str,
        content_type: str = 'application/json',
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Put object to S3 with metadata and tags

        Returns:
            Version ID of the uploaded object
        """
        put_args = {
            'Bucket': self.bucket_name,
            'Key': key,
            'Body': body.encode('utf-8'),
            'ContentType': content_type,
            'StorageClass': self.storage_class,
        }

        if metadata:
            put_args['Metadata'] = metadata

        if self.enable_encryption:
            put_args['ServerSideEncryption'] = 'AES256'

        # Upload object
        response = self.s3_client.put_object(**put_args)
        version_id = response.get('VersionId', 'null')

        # Add tags if provided
        if tags:
            try:
                tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
                self.s3_client.put_object_tagging(
                    Bucket=self.bucket_name,
                    Key=key,
                    Tagging={'TagSet': tag_set}
                )
            except ClientError as e:
                print(f"Warning: Failed to add tags: {e}")

        return version_id

    def _get_object(
        self,
        key: str,
        version_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get object from S3 with optional version"""
        try:
            get_args = {'Bucket': self.bucket_name, 'Key': key}
            if version_id:
                get_args['VersionId'] = version_id

            response = self.s3_client.get_object(**get_args)
            content = response['Body'].read().decode('utf-8')
            return {
                'content': content,
                'metadata': response.get('Metadata', {}),
                'version_id': response.get('VersionId'),
                'last_modified': response.get('LastModified'),
            }
        except ClientError:
            return None

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save a prompt to S3

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

        # Get current version number
        version = self._get_next_version(name, branch)

        # Store prompt content
        prompt_key = f"prompts/{branch}/{name}/v{version}.json"
        prompt_obj = {
            'name': name,
            'content': content,
            'branch': branch,
            'version': version,
            'commit_hash': commit_hash,
            'created_at': timestamp,
            'metadata': metadata,
        }

        # Upload with metadata and tags
        version_id = self._put_object(
            prompt_key,
            json.dumps(prompt_obj, indent=2),
            metadata={
                'name': name,
                'branch': branch,
                'version': str(version),
                'commit-hash': commit_hash,
            },
            tags={
                'type': 'prompt',
                'branch': branch,
                'name': name,
            }
        )

        # Update branch head
        self._update_branch_head(branch, commit_hash)

        return commit_hash

    def _get_next_version(self, name: str, branch: str) -> int:
        """Get the next version number for a prompt"""
        prefix = f"prompts/{branch}/{name}/"
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' not in response:
                return 1

            # Count existing versions
            versions = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')]
            return len(versions) + 1
        except ClientError:
            return 1

    def _update_branch_head(self, branch_name: str, commit_hash: str) -> None:
        """Update branch head commit"""
        branch_key = f"branches/{branch_name}.json"
        branch_obj = self._get_object(branch_key)

        if branch_obj:
            branch_data = json.loads(branch_obj['content'])
            branch_data['head_commit'] = commit_hash
            branch_data['updated_at'] = datetime.now().isoformat()
        else:
            branch_data = {
                'name': branch_name,
                'created_at': datetime.now().isoformat(),
                'head_commit': commit_hash,
            }

        self._put_object(
            branch_key,
            json.dumps(branch_data, indent=2),
            metadata={'branch': branch_name}
        )

    def get_prompt(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None,
        commit_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a prompt from S3

        Args:
            name: Prompt name
            branch: Branch name
            version: Optional specific version
            commit_hash: Optional specific commit hash

        Returns:
            Dict with prompt data or None if not found
        """
        if commit_hash:
            # Search for prompt by commit hash
            return self._get_prompt_by_commit(name, branch, commit_hash)

        if version is None:
            # Get latest version
            version = self._get_latest_version(name, branch)
            if version is None:
                return None

        prompt_key = f"prompts/{branch}/{name}/v{version}.json"
        obj = self._get_object(prompt_key)

        if not obj:
            return None

        prompt_data = json.loads(obj['content'])
        prompt_data['s3_version_id'] = obj['version_id']
        return prompt_data

    def _get_latest_version(self, name: str, branch: str) -> Optional[int]:
        """Get the latest version number for a prompt"""
        prefix = f"prompts/{branch}/{name}/"
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' not in response:
                return None

            versions = []
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    # Extract version number from key
                    filename = obj['Key'].split('/')[-1]
                    if filename.startswith('v') and filename.endswith('.json'):
                        ver_str = filename[1:-5]  # Remove 'v' and '.json'
                        try:
                            versions.append(int(ver_str))
                        except ValueError:
                            continue

            return max(versions) if versions else None
        except ClientError:
            return None

    def _get_prompt_by_commit(
        self,
        name: str,
        branch: str,
        commit_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get prompt by commit hash"""
        prefix = f"prompts/{branch}/{name}/"
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' not in response:
                return None

            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    obj_data = self._get_object(obj['Key'])
                    if obj_data:
                        prompt_data = json.loads(obj_data['content'])
                        if prompt_data.get('commit_hash') == commit_hash:
                            prompt_data['s3_version_id'] = obj_data['version_id']
                            return prompt_data
            return None
        except ClientError:
            return None

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """
        List all prompts on a branch

        Args:
            branch: Branch name

        Returns:
            List of prompt metadata dictionaries
        """
        prefix = f"prompts/{branch}/"
        prompts = {}

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.json'):
                    # Extract prompt name from key
                    parts = key.split('/')
                    if len(parts) >= 3:
                        prompt_name = parts[2]

                        # Get latest version for this prompt
                        if prompt_name not in prompts:
                            obj_data = self._get_object(key)
                            if obj_data:
                                prompt_data = json.loads(obj_data['content'])
                                prompts[prompt_name] = {
                                    'name': prompt_data['name'],
                                    'branch': prompt_data['branch'],
                                    'version': prompt_data['version'],
                                    'commit_hash': prompt_data['commit_hash'],
                                    'created_at': prompt_data['created_at'],
                                }

            return list(prompts.values())
        except ClientError as e:
            print(f"Error listing prompts: {e}")
            return []

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """
        Create a new branch

        Args:
            branch_name: Name of the new branch
            from_branch: Source branch to branch from
        """
        # Copy branch metadata
        source_key = f"branches/{from_branch}.json"
        source_obj = self._get_object(source_key)

        if not source_obj:
            raise ValueError(f"Source branch '{from_branch}' not found")

        source_data = json.loads(source_obj['content'])

        # Create new branch
        new_branch_data = {
            'name': branch_name,
            'created_at': datetime.now().isoformat(),
            'head_commit': source_data['head_commit'],
            'parent_branch': from_branch,
        }

        branch_key = f"branches/{branch_name}.json"
        self._put_object(
            branch_key,
            json.dumps(new_branch_data, indent=2),
            metadata={'branch': branch_name}
        )

        # Copy all prompts from source branch (shallow copy - references only)
        # In a real implementation, you might want to use S3 batch copy
        # For now, we just create the branch metadata

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        return self.current_branch

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        # Verify branch exists
        branch_key = f"branches/{branch_name}.json"
        if not self._get_object(branch_key):
            raise ValueError(f"Branch '{branch_name}' not found")

        self.current_branch = branch_name

    def get_commit_history(
        self,
        name: Optional[str] = None,
        branch: str = "main",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get commit history

        Args:
            name: Optional prompt name to filter by
            branch: Branch name
            limit: Maximum number of commits to return

        Returns:
            List of commit dictionaries
        """
        if name:
            # Get history for specific prompt
            prefix = f"prompts/{branch}/{name}/"
        else:
            # Get history for all prompts on branch
            prefix = f"prompts/{branch}/"

        commits = []
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=limit * 2  # Get more to filter later
            )

            if 'Contents' not in response:
                return []

            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    obj_data = self._get_object(obj['Key'])
                    if obj_data:
                        prompt_data = json.loads(obj_data['content'])
                        commits.append({
                            'commit_hash': prompt_data['commit_hash'],
                            'name': prompt_data['name'],
                            'version': prompt_data['version'],
                            'branch': prompt_data['branch'],
                            'created_at': prompt_data['created_at'],
                            'metadata': prompt_data.get('metadata', {}),
                        })

            # Sort by timestamp (newest first)
            commits.sort(key=lambda x: x['created_at'], reverse=True)

            return commits[:limit]
        except ClientError as e:
            print(f"Error getting commit history: {e}")
            return []

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """
        Save evaluation results

        Args:
            eval_data: Dictionary containing evaluation data
        """
        prompt_name = eval_data['prompt_name']
        eval_id = hashlib.sha256(
            json.dumps(eval_data, sort_keys=True).encode()
        ).hexdigest()[:12]

        eval_key = f"evaluations/{prompt_name}/{eval_id}.json"
        eval_obj = {
            **eval_data,
            'eval_id': eval_id,
            'created_at': datetime.now().isoformat(),
        }

        self._put_object(
            eval_key,
            json.dumps(eval_obj, indent=2),
            metadata={'prompt-name': prompt_name},
            tags={'type': 'evaluation', 'prompt': prompt_name}
        )

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """
        Save a prompt chain

        Args:
            chain_data: Dictionary containing chain data
        """
        name = chain_data['name']
        chain_key = f"chains/{name}.json"

        chain_obj = {
            **chain_data,
            'updated_at': datetime.now().isoformat(),
        }

        self._put_object(
            chain_key,
            json.dumps(chain_obj, indent=2),
            metadata={'chain-name': name},
            tags={'type': 'chain', 'name': name}
        )

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a prompt chain

        Args:
            name: Chain name

        Returns:
            Dict with chain data or None if not found
        """
        chain_key = f"chains/{name}.json"
        obj = self._get_object(chain_key)

        if not obj:
            return None

        return json.loads(obj['content'])

    def close(self) -> None:
        """Close S3 connection"""
        # boto3 manages connections automatically
        pass

    # Additional S3-specific methods

    def generate_presigned_url(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access

        Args:
            name: Prompt name
            branch: Branch name
            version: Optional specific version (defaults to latest)
            expiration: URL expiration in seconds

        Returns:
            Presigned URL or None if prompt not found
        """
        if version is None:
            version = self._get_latest_version(name, branch)
            if version is None:
                return None

        prompt_key = f"prompts/{branch}/{name}/v{version}.json"

        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': prompt_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")
            return None

    def get_cloudfront_url(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None
    ) -> Optional[str]:
        """
        Get CloudFront CDN URL for fast reads

        Args:
            name: Prompt name
            branch: Branch name
            version: Optional specific version (defaults to latest)

        Returns:
            CloudFront URL or None if CDN not configured
        """
        if not self.cloudfront_domain:
            return None

        if version is None:
            version = self._get_latest_version(name, branch)
            if version is None:
                return None

        prompt_key = f"prompts/{branch}/{name}/v{version}.json"
        return f"https://{self.cloudfront_domain}/{prompt_key}"

    def enable_cross_region_replication(
        self,
        destination_bucket: str,
        destination_region: str,
        role_arn: str
    ) -> None:
        """
        Enable cross-region replication for disaster recovery

        Args:
            destination_bucket: Destination bucket name
            destination_region: Destination region
            role_arn: IAM role ARN with replication permissions
        """
        try:
            replication_config = {
                'Role': role_arn,
                'Rules': [
                    {
                        'ID': 'replicate-all',
                        'Status': 'Enabled',
                        'Priority': 1,
                        'Filter': {'Prefix': ''},
                        'Destination': {
                            'Bucket': f'arn:aws:s3:::{destination_bucket}',
                            'ReplicationTime': {
                                'Status': 'Enabled',
                                'Time': {'Minutes': 15}
                            },
                            'Metrics': {
                                'Status': 'Enabled',
                                'EventThreshold': {'Minutes': 15}
                            }
                        },
                        'DeleteMarkerReplication': {'Status': 'Enabled'}
                    }
                ]
            }
            self.s3_client.put_bucket_replication(
                Bucket=self.bucket_name,
                ReplicationConfiguration=replication_config
            )
            print(f"Cross-region replication enabled to {destination_bucket} ({destination_region})")
        except ClientError as e:
            raise RuntimeError(f"Failed to enable replication: {e}")

    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Dict with storage metrics
        """
        stats = {
            'bucket_name': self.bucket_name,
            'region': self.region,
            'versioning_enabled': self.enable_versioning,
            'encryption_enabled': self.enable_encryption,
            'storage_class': self.storage_class,
            'total_objects': 0,
            'total_size_bytes': 0,
            'prompts_count': 0,
            'chains_count': 0,
            'evaluations_count': 0,
        }

        try:
            # Count objects by prefix
            for obj_type, prefix in [
                ('prompts_count', 'prompts/'),
                ('chains_count', 'chains/'),
                ('evaluations_count', 'evaluations/'),
            ]:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                if 'Contents' in response:
                    stats[obj_type] = len(response['Contents'])
                    stats['total_objects'] += len(response['Contents'])
                    stats['total_size_bytes'] += sum(
                        obj['Size'] for obj in response['Contents']
                    )

        except ClientError as e:
            print(f"Error getting statistics: {e}")

        return stats
