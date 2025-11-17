"""
SDK Exceptions
"""


class PromptlySDKError(Exception):
    """Base exception for Promptly SDK"""
    pass


class AuthenticationError(PromptlySDKError):
    """Authentication failed"""
    pass


class PromptNotFoundError(PromptlySDKError):
    """Prompt not found"""
    pass


class BranchNotFoundError(PromptlySDKError):
    """Branch not found"""
    pass


class RateLimitError(PromptlySDKError):
    """Rate limit exceeded"""
    pass


class APIError(PromptlySDKError):
    """General API error"""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
