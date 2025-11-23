"""
City Services Department

Handles:
- Maintenance requests (streets, lighting, parks)
- Building permits
- Noise ordinance enforcement
- Public services coordination

Simpler than HOA - focuses on infrastructure and permits.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from HoloLoom.departments.base import BaseDepartment
from HoloLoom.departments.protocol import DepartmentRequest, DepartmentResponse

logger = logging.getLogger(__name__)


@dataclass
class ServiceRequest:
    """City service request."""
    request_id: str
    request_type: str  # "maintenance", "permit", "noise_ordinance"
    resident_id: str
    description: str
    status: str = "pending"  # "pending", "approved", "denied", "completed"
    timestamp: float = 0.0


class CityServicesDepartment(BaseDepartment):
    """
    City Services Department for public services and permits.

    Simpler workflow than HOA:
    - File service request
    - Review and approve/deny
    - Track completion
    """

    def __init__(self):
        """Initialize City Services Department."""
        super().__init__(name="CityServices", version="1.0.0")
        self.requests: List[ServiceRequest] = []

    async def process(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Process city services request.

        Supported request types:
        - "file_request": File a service request
        - "review_request": Review and approve/deny request
        - "get_status": Get status of a request
        - "get_statistics": Get department statistics
        """
        request_type = request.request_type

        if request_type == "file_request":
            return await self._file_request(request)
        elif request_type == "review_request":
            return await self._review_request(request)
        elif request_type == "get_status":
            return await self._get_status(request)
        elif request_type == "get_statistics":
            return await self._get_statistics(request)
        else:
            return DepartmentResponse(
                success=False,
                error=f"Unknown request type: {request_type}"
            )

    async def _file_request(self, request: DepartmentRequest) -> DepartmentResponse:
        """File a service request."""
        data = request.data

        resident_id = data.get("resident_id")
        request_type = data.get("request_type")
        description = data.get("description")

        if not all([resident_id, request_type, description]):
            return DepartmentResponse(
                success=False,
                error="Missing required fields"
            )

        service_request = ServiceRequest(
            request_id=f"CS-{len(self.requests)+1:04d}",
            request_type=request_type,
            resident_id=resident_id,
            description=description,
            timestamp=data.get("timestamp", 0.0)
        )

        self.requests.append(service_request)

        logger.info(f"Filed service request: {service_request.request_id}")

        return DepartmentResponse(
            success=True,
            data={
                "request_id": service_request.request_id,
                "message": f"Request filed. ID: {service_request.request_id}"
            }
        )

    async def _review_request(self, request: DepartmentRequest) -> DepartmentResponse:
        """Review and approve/deny request."""
        request_id = request.data.get("request_id")
        action = request.data.get("action", "approved")  # "approved" or "denied"

        service_request = next(
            (r for r in self.requests if r.request_id == request_id),
            None
        )

        if not service_request:
            return DepartmentResponse(
                success=False,
                error=f"Request not found: {request_id}"
            )

        service_request.status = action

        logger.info(f"Reviewed {request_id}: {action}")

        return DepartmentResponse(
            success=True,
            data={
                "request_id": request_id,
                "status": action
            }
        )

    async def _get_status(self, request: DepartmentRequest) -> DepartmentResponse:
        """Get status of a request."""
        request_id = request.data.get("request_id")

        service_request = next(
            (r for r in self.requests if r.request_id == request_id),
            None
        )

        if not service_request:
            return DepartmentResponse(
                success=False,
                error=f"Request not found: {request_id}"
            )

        return DepartmentResponse(
            success=True,
            data={
                "request_id": request_id,
                "status": service_request.status,
                "type": service_request.request_type,
                "description": service_request.description
            }
        )

    async def _get_statistics(self, request: DepartmentRequest) -> DepartmentResponse:
        """Get department statistics."""
        total = len(self.requests)
        by_status = {}
        by_type = {}

        for r in self.requests:
            by_status[r.status] = by_status.get(r.status, 0) + 1
            by_type[r.request_type] = by_type.get(r.request_type, 0) + 1

        return DepartmentResponse(
            success=True,
            data={
                "total_requests": total,
                "by_status": by_status,
                "by_type": by_type
            }
        )
