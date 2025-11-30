"""
Test security features for AR API server.

Tests input validation for file uploads and rate limiting for vision endpoints.
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from io import BytesIO
from collections import deque
from PIL import Image
import numpy as np

from fastapi import HTTPException, UploadFile, Request
from fastapi.datastructures import Headers


# Import the security components
from HoloLoom.server.ar_api import (
    validate_image_upload,
    RateLimiter,
    MAX_FILE_SIZE,
    ALLOWED_IMAGE_FORMATS
)


class TestImageUploadValidation:
    """Test image upload validation security."""

    @pytest.fixture
    def create_mock_upload(self):
        """Factory fixture to create mock UploadFile objects."""
        def _create(content_type='image/png', size=1024, content=None):
            mock_file = MagicMock(spec=UploadFile)
            mock_file.content_type = content_type

            # Create a mock file object with seek/tell behavior
            mock_file_obj = MagicMock()
            mock_file_obj.tell.return_value = size
            mock_file_obj.seek = MagicMock()

            # If content provided, use it; otherwise create valid PNG
            if content is None:
                # Create a valid small PNG image
                img = Image.new('RGB', (10, 10), color='red')
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                content = buffer.getvalue()

            # Make read return the content
            mock_file.read = AsyncMock(return_value=content)
            mock_file.seek = AsyncMock()
            mock_file.file = mock_file_obj

            return mock_file

        return _create

    @pytest.mark.asyncio
    async def test_valid_image_formats(self, create_mock_upload):
        """Test that valid image formats are accepted."""
        valid_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif']

        for format_type in valid_formats:
            # Create appropriate image content for each format
            img = Image.new('RGB', (10, 10), color='blue')
            buffer = BytesIO()

            # Map content type to PIL format
            pil_format = {
                'image/jpeg': 'JPEG',
                'image/jpg': 'JPEG',
                'image/png': 'PNG',
                'image/webp': 'WEBP',
                'image/gif': 'GIF'
            }[format_type]

            img.save(buffer, format=pil_format)
            content = buffer.getvalue()

            mock_file = create_mock_upload(
                content_type=format_type,
                size=len(content),
                content=content
            )

            # Should not raise
            await validate_image_upload(mock_file)

    @pytest.mark.asyncio
    async def test_invalid_image_format(self, create_mock_upload):
        """Test that invalid formats are rejected."""
        invalid_formats = [
            'text/plain',
            'application/pdf',
            'image/svg+xml',
            'application/octet-stream',
            'video/mp4'
        ]

        for format_type in invalid_formats:
            mock_file = create_mock_upload(content_type=format_type)

            with pytest.raises(HTTPException) as exc_info:
                await validate_image_upload(mock_file)

            assert exc_info.value.status_code == 400
            assert "Invalid file format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_file_size_limit(self, create_mock_upload):
        """Test that files exceeding size limit are rejected."""
        # Create file larger than limit
        oversized = MAX_FILE_SIZE + 1
        mock_file = create_mock_upload(size=oversized)

        with pytest.raises(HTTPException) as exc_info:
            await validate_image_upload(mock_file)

        assert exc_info.value.status_code == 413
        assert "File too large" in exc_info.value.detail
        assert "10MB" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_file_at_size_limit(self, create_mock_upload):
        """Test that file exactly at size limit is accepted."""
        # Create valid image at max size
        img = Image.new('RGB', (1000, 1000), color='green')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        content = buffer.getvalue()

        mock_file = create_mock_upload(
            size=MAX_FILE_SIZE,
            content=content
        )

        # Should not raise
        await validate_image_upload(mock_file)

    @pytest.mark.asyncio
    async def test_malformed_image_rejected(self, create_mock_upload):
        """Test that malformed images are rejected even with correct content type."""
        # Create file with PNG content type but invalid content
        invalid_content = b"This is not a valid image file"
        mock_file = create_mock_upload(
            content_type='image/png',
            size=len(invalid_content),
            content=invalid_content
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_image_upload(mock_file)

        assert exc_info.value.status_code == 400
        assert "Invalid image file" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_corrupted_image_rejected(self, create_mock_upload):
        """Test that corrupted images are rejected."""
        # Create a valid PNG header but corrupted data
        # PNG signature + corrupted IHDR chunk
        corrupted_png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100

        mock_file = create_mock_upload(
            content_type='image/png',
            size=len(corrupted_png),
            content=corrupted_png
        )

        with pytest.raises(HTTPException) as exc_info:
            await validate_image_upload(mock_file)

        assert exc_info.value.status_code == 400
        assert "Invalid image file" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_file_pointer_reset(self, create_mock_upload):
        """Test that file pointer is reset after validation."""
        img = Image.new('RGB', (10, 10), color='yellow')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        content = buffer.getvalue()

        mock_file = create_mock_upload(content=content)

        await validate_image_upload(mock_file)

        # Verify seek(0) was called to reset pointer
        mock_file.seek.assert_called_with(0)

    @pytest.mark.asyncio
    async def test_empty_file_rejected(self, create_mock_upload):
        """Test that empty files are rejected."""
        mock_file = create_mock_upload(size=0, content=b'')

        with pytest.raises(HTTPException) as exc_info:
            await validate_image_upload(mock_file)

        assert exc_info.value.status_code == 400


class TestRateLimiter:
    """Test rate limiting functionality."""

    @pytest.fixture
    def create_mock_request(self):
        """Factory fixture to create mock Request objects."""
        def _create(client_ip='192.168.1.1'):
            mock_request = MagicMock(spec=Request)
            mock_client = MagicMock()
            mock_client.host = client_ip
            mock_request.client = mock_client
            return mock_request

        return _create

    @pytest.mark.asyncio
    async def test_rate_limit_initialization(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(max_requests=5, window_seconds=30)

        assert limiter.max_requests == 5
        assert limiter.window_seconds == 30
        assert isinstance(limiter.requests, dict)

    @pytest.mark.asyncio
    async def test_requests_under_limit(self, create_mock_request):
        """Test that requests under the limit are allowed."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        mock_request = create_mock_request('10.0.0.1')

        # First 3 requests should succeed
        for _ in range(3):
            await limiter.check_rate_limit(mock_request)  # Should not raise

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, create_mock_request):
        """Test that exceeding rate limit raises 429 error."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        mock_request = create_mock_request('10.0.0.2')

        # First 2 requests OK
        await limiter.check_rate_limit(mock_request)
        await limiter.check_rate_limit(mock_request)

        # Third request should fail
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail
        assert "2 requests per 60s" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_sliding_window(self, create_mock_request):
        """Test that sliding window works correctly."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)  # 1 second window
        mock_request = create_mock_request('10.0.0.3')

        # Add 2 requests
        await limiter.check_rate_limit(mock_request)
        await limiter.check_rate_limit(mock_request)

        # Should be at limit
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(mock_request)

        # Wait for window to expire
        await asyncio.sleep(1.1)

        # Should work again
        await limiter.check_rate_limit(mock_request)  # Should not raise

    @pytest.mark.asyncio
    async def test_per_ip_tracking(self, create_mock_request):
        """Test that rate limits are tracked per IP."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)

        # Different IPs
        request1 = create_mock_request('10.0.0.4')
        request2 = create_mock_request('10.0.0.5')

        # Each IP gets its own limit
        await limiter.check_rate_limit(request1)
        await limiter.check_rate_limit(request2)

        # Second request from first IP should fail
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(request1)

        # But second IP can still make another request after window
        # (though it would also fail since we set limit to 1)
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(request2)

    @pytest.mark.asyncio
    async def test_unknown_client_ip(self, create_mock_request):
        """Test handling of requests with no client info."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)

        # Request with no client
        mock_request = MagicMock(spec=Request)
        mock_request.client = None

        # Should use "unknown" as IP
        await limiter.check_rate_limit(mock_request)

        # Second request should fail
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit(mock_request)

    @pytest.mark.asyncio
    async def test_request_queue_cleanup(self, create_mock_request):
        """Test that old requests are cleaned up from queue."""
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0.0

            limiter = RateLimiter(max_requests=10, window_seconds=10)
            mock_request = create_mock_request('10.0.0.6')

            # Add 5 requests at time 0
            for _ in range(5):
                await limiter.check_rate_limit(mock_request)

            # Move to time 15 (outside window)
            mock_time.return_value = 15.0

            # Add new request - old ones should be cleaned up
            await limiter.check_rate_limit(mock_request)

            # Check that queue only has 1 request (the new one)
            assert len(limiter.requests['10.0.0.6']) == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, create_mock_request):
        """Test rate limiter with concurrent requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        mock_request = create_mock_request('10.0.0.7')

        async def make_request():
            try:
                await limiter.check_rate_limit(mock_request)
                return True
            except HTTPException:
                return False

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Exactly 5 should succeed
        assert sum(results) == 5
        assert results.count(False) == 5

    def test_rate_limiter_data_structure(self):
        """Test the internal data structure of rate limiter."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # Check it uses defaultdict with deque
        assert isinstance(limiter.requests, dict)

        # Access an IP should create a deque
        queue = limiter.requests['test_ip']
        assert isinstance(queue, deque)

    @pytest.mark.asyncio
    async def test_global_vision_rate_limiter(self):
        """Test that global vision_rate_limiter is configured correctly."""
        from HoloLoom.server.ar_api import vision_rate_limiter

        # Should be configured with conservative limits
        assert vision_rate_limiter.max_requests == 10
        assert vision_rate_limiter.window_seconds == 60

    @pytest.mark.asyncio
    async def test_rate_limit_boundary_conditions(self, create_mock_request):
        """Test boundary conditions for rate limiting."""
        # Test with 0 requests allowed
        limiter = RateLimiter(max_requests=0, window_seconds=60)
        mock_request = create_mock_request('10.0.0.8')

        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_rate_limit(mock_request)
        assert exc_info.value.status_code == 429

        # Test with very short window
        limiter = RateLimiter(max_requests=100, window_seconds=0.001)
        await asyncio.sleep(0.002)  # Wait for window to pass
        await limiter.check_rate_limit(mock_request)  # Should work

    @pytest.mark.asyncio
    async def test_rate_limit_exact_window_edge(self, create_mock_request):
        """Test requests at exact window boundaries."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 100.0

            limiter = RateLimiter(max_requests=1, window_seconds=10)
            mock_request = create_mock_request('10.0.0.9')

            # First request at t=100
            await limiter.check_rate_limit(mock_request)

            # Move to exactly window edge (t=110)
            mock_time.return_value = 110.0

            # Should fail (still in window)
            with pytest.raises(HTTPException):
                await limiter.check_rate_limit(mock_request)

            # Move just past window (t=110.001)
            mock_time.return_value = 110.001

            # Should work now
            await limiter.check_rate_limit(mock_request)