"""Tests for PII detection and redaction."""
from guardrails.pii_detector import detect_pii, redact_pii, PIIMatch


class TestDetectPII:
    def test_detect_ssn(self):
        text = "My SSN is 123-45-6789"
        matches = detect_pii(text)
        assert len(matches) == 1
        assert matches[0].type == "ssn"
        assert matches[0].value == "123-45-6789"
        assert matches[0].start == 10
        assert matches[0].end == 21

    def test_detect_credit_card(self):
        text = "Card number is 4111-1111-1111-1111"
        matches = detect_pii(text)
        assert len(matches) == 1
        assert matches[0].type == "credit_card"
        assert matches[0].value == "4111-1111-1111-1111"

    def test_detect_email(self):
        text = "Contact me at user@example.com for details"
        matches = detect_pii(text)
        assert len(matches) == 1
        assert matches[0].type == "email"
        assert matches[0].value == "user@example.com"

    def test_detect_phone(self):
        text = "Call me at (555) 123-4567"
        matches = detect_pii(text)
        assert len(matches) >= 1
        assert any(m.type == "phone" for m in matches)

    def test_detect_no_pii(self):
        text = "This document contains no personal information"
        matches = detect_pii(text)
        assert len(matches) == 0

    def test_detect_multiple_pii(self):
        text = "SSN: 123-45-6789, email: test@test.com, card: 4111111111111111"
        matches = detect_pii(text)
        types = [m.type for m in matches]
        assert "ssn" in types
        assert "email" in types
        assert "credit_card" in types
        assert len(matches) >= 3

    def test_multiple_emails(self):
        text = "Send to a@b.com or c@d.com"
        matches = detect_pii(text)
        email_matches = [m for m in matches if m.type == "email"]
        assert len(email_matches) == 2


class TestRedactPII:
    def test_redact_ssn(self):
        text = "SSN: 123-45-6789"
        redacted = redact_pii(text)
        assert "123-45-6789" not in redacted
        assert "[REDACTED_SSN]" in redacted

    def test_redact_email(self):
        text = "Email: user@example.com"
        redacted = redact_pii(text)
        assert "user@example.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_redact_multiple(self):
        text = "SSN: 123-45-6789, Email: test@test.com"
        redacted = redact_pii(text)
        assert "[REDACTED_SSN]" in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_redact_no_pii(self):
        text = "No PII here"
        redacted = redact_pii(text)
        assert redacted == text

    def test_redact_preserves_non_pii(self):
        text = "Invoice #12345 for $100"
        redacted = redact_pii(text)
        assert "Invoice" in redacted
        assert "$100" in redacted
