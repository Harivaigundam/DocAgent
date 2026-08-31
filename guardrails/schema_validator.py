from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_schema(data: dict, required_fields: list[str]) -> ValidationResult:
    """
    Validate that data contains all required fields with valid values.
    
    Args:
        data: Dictionary to validate
        required_fields: List of field names that must be present and non-None
    
    Returns:
        ValidationResult with validation status and any errors found
    """
    errors = []
    warnings = []
    
    for field_name in required_fields:
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")
        elif data[field_name] is None:
            errors.append(f"Required field is None: {field_name}")
    
    if "total" in data and data["total"] is not None:
        if not isinstance(data["total"], (int, float)):
            errors.append("Field 'total' must be a number")
        elif data["total"] < 0:
            errors.append("Field 'total' must be non-negative")
    
    is_valid = len(errors) == 0
    
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
