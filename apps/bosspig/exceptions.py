"""
BossPig Exception Classes

Provides comprehensive error handling with actionable error messages.

Created: 2025-11-22
Status: Week 11 Production Hardening
"""


class BossPigError(Exception):
    """Base exception for all BossPig errors."""

    def __init__(self, message: str, suggestion: str = None):
        """
        Initialize BossPig error.

        Args:
            message: Error description
            suggestion: Actionable suggestion for fixing the error
        """
        self.message = message
        self.suggestion = suggestion

        full_message = message
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"

        super().__init__(full_message)


class BossPigConfigError(BossPigError):
    """
    Configuration-related errors.

    Examples:
    - Missing config file
    - Invalid JSON
    - Required field missing
    """
    pass


class BossPigFileError(BossPigError):
    """
    File reading/writing errors.

    Examples:
    - File not found
    - Permission denied
    - Invalid file format
    """
    pass


class BossPigAnalysisError(BossPigError):
    """
    Analysis/detection errors.

    Examples:
    - Regex compilation failure
    - NLP model unavailable
    - Invalid input text
    """
    pass


class BossPigTimeoutError(BossPigError):
    """
    Timeout errors for very large documents.

    Raised when analysis exceeds configured timeout.
    """
    pass


class BossPigValidationError(BossPigError):
    """
    Input validation errors.

    Examples:
    - Empty text
    - Invalid document type
    - Unsupported file format
    """
    pass


# ============================================================================
# Error Factory Functions
# ============================================================================

def config_not_found(config_path: str) -> BossPigConfigError:
    """Factory for config file not found error."""
    return BossPigConfigError(
        f"Configuration file not found: {config_path}",
        "Create a configuration file at this path or specify a different path using the config_path parameter."
    )


def invalid_config_json(config_path: str, error: str) -> BossPigConfigError:
    """Factory for invalid JSON config error."""
    return BossPigConfigError(
        f"Invalid JSON in configuration file: {config_path}\n{error}",
        "Check the JSON syntax using a validator (e.g., jsonlint.com). Common issues: missing commas, trailing commas, unescaped quotes."
    )


def missing_config_field(config_path: str, field: str) -> BossPigConfigError:
    """Factory for missing required config field error."""
    return BossPigConfigError(
        f"Required field '{field}' missing in configuration: {config_path}",
        f"Add the '{field}' field to your configuration file. See docs/CONFIGURATION.md for examples."
    )


def file_not_found(file_path: str) -> BossPigFileError:
    """Factory for file not found error."""
    return BossPigFileError(
        f"File not found: {file_path}",
        "Check that the file path is correct and the file exists. Use absolute paths to avoid confusion."
    )


def file_permission_denied(file_path: str) -> BossPigFileError:
    """Factory for permission denied error."""
    return BossPigFileError(
        f"Permission denied: {file_path}",
        "Check that you have read permissions for this file. On Unix systems: chmod +r <file>"
    )


def invalid_document_type(doc_type: str, valid_types: list) -> BossPigValidationError:
    """Factory for invalid document type error."""
    return BossPigValidationError(
        f"Invalid document type: '{doc_type}'",
        f"Valid document types are: {', '.join(valid_types)}. Specify one of these using the document_type parameter."
    )


def empty_text_input() -> BossPigValidationError:
    """Factory for empty text input error."""
    return BossPigValidationError(
        "Cannot analyze empty text",
        "Provide non-empty text to analyze. If reading from a file, check that the file is not empty."
    )


def invalid_file_encoding(file_path: str) -> BossPigFileError:
    """Factory for invalid file encoding error."""
    return BossPigFileError(
        f"File has invalid encoding: {file_path}",
        "The file cannot be decoded as UTF-8. Try converting the file to UTF-8 encoding or specify a different encoding."
    )


def analysis_failed(error_msg: str) -> BossPigAnalysisError:
    """Factory for general analysis failure error."""
    return BossPigAnalysisError(
        f"Analysis failed: {error_msg}",
        "Check the error details above. If the issue persists, please report it as a bug."
    )


def analysis_timeout(timeout_seconds: int, doc_size: int) -> BossPigTimeoutError:
    """Factory for analysis timeout error."""
    return BossPigTimeoutError(
        f"Analysis timed out after {timeout_seconds} seconds (document size: {doc_size} words)",
        f"Try splitting the document into smaller chunks or increasing the timeout. Very large documents (>50,000 words) may need special handling."
    )


def regex_compilation_failed(pattern: str, error: str) -> BossPigAnalysisError:
    """Factory for regex compilation error."""
    return BossPigAnalysisError(
        f"Failed to compile regex pattern: {pattern}\n{error}",
        "Check the regex pattern syntax. Common issues: unescaped special characters, unbalanced parentheses."
    )


def nlp_model_unavailable(model_name: str) -> BossPigAnalysisError:
    """Factory for NLP model unavailable error."""
    return BossPigAnalysisError(
        f"NLP model not available: {model_name}",
        f"Install the required spaCy model: python -m spacy download {model_name}"
    )


if __name__ == "__main__":
    # Example usage
    try:
        raise config_not_found("/path/to/missing/config.json")
    except BossPigConfigError as e:
        print(f"Error: {e}")
        print(f"\nCaught BossPigConfigError:")
        print(f"  Message: {e.message}")
        print(f"  Suggestion: {e.suggestion}")
