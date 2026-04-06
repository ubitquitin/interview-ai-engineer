"""
Unit tests for Pydantic models.

Fast, lightweight tests with no external dependencies.
Validates schema definitions and basic model behavior.
"""

import pytest
from pydantic import ValidationError
from src.models import (
    Deficiency,
    ProcessEntity,
    ExtractionOutput,
    RiskLevel,
    RiskItem,
    ComplianceOutput,
    WarningLetterMetadata,
)


class TestDeficiency:
    """Test suite for Deficiency model."""

    def test_valid_deficiency(self):
        """Test creation of valid deficiency.

        Verifies:
            - All required fields are accepted
            - Field values are properly assigned
        """
        deficiency = Deficiency(
            title="Inadequate Cleaning Validation",
            cfr_reference="21 CFR 211.67",
            description="Failed to establish written procedures for cleaning",
            evidence="Observed residue on equipment surface",
            required_action="Establish and validate cleaning procedures",
        )
        assert deficiency.title == "Inadequate Cleaning Validation"
        assert deficiency.cfr_reference == "21 CFR 211.67"

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError.

        Verifies:
            - ValidationError raised when required fields are missing
            - Pydantic validation is enforced
        """
        with pytest.raises(ValidationError):
            Deficiency(title="Test", cfr_reference="21 CFR 211.67")


class TestProcessEntity:
    """Test suite for ProcessEntity model."""

    def test_valid_process_entity(self):
        """Test creation of valid process entity.

        Verifies:
            - All entity fields properly initialized
            - Type and name fields stored correctly
        """
        entity = ProcessEntity(
            name="Batch Record System",
            type="system",
            description="Electronic system for tracking batch operations",
        )
        assert entity.name == "Batch Record System"
        assert entity.type == "system"

    def test_entity_types(self):
        """Test various entity types are accepted.

        Verifies:
            - All standard entity types (process, system, material, control) are valid
            - Type field correctly stores provided value
        """
        types = ["process", "system", "material", "control"]
        for entity_type in types:
            entity = ProcessEntity(
                name=f"Test {entity_type}",
                type=entity_type,
                description=f"A test {entity_type}",
            )
            assert entity.type == entity_type


class TestExtractionOutput:
    """Test suite for ExtractionOutput model."""

    def test_valid_extraction_output(self):
        """Test creation of valid extraction output with nested entities.

        Verifies:
            - Complex nested structure properly initialized
            - All list fields contain expected items
            - Nested entity objects accessible
        """
        output = ExtractionOutput(
            summary="Manufacturing process with quality controls",
            entities=[
                ProcessEntity(
                    name="Weighing Process",
                    type="process",
                    description="Weight verification for raw materials",
                )
            ],
            processes=["weighing", "mixing"],
            materials=["active ingredient", "excipient"],
            systems=["LIMS", "ERP"],
            controls=["SOP-001", "audit trail"],
        )
        assert len(output.entities) == 1
        assert "weighing" in output.processes
        assert "LIMS" in output.systems

    def test_empty_lists_allowed(self):
        """Test that empty lists are valid.

        Verifies:
            - Empty lists accepted for all collection fields
            - Model handles minimal data gracefully
        """
        output = ExtractionOutput(
            summary="Minimal process",
            entities=[],
            processes=[],
            materials=[],
            systems=[],
            controls=[],
        )
        assert len(output.entities) == 0
        assert len(output.processes) == 0


class TestRiskLevel:
    """Test suite for RiskLevel enum."""

    def test_risk_level_values(self):
        """Test all risk level enum values.

        Verifies:
            - All four risk levels (HIGH, MEDIUM, LOW, NONE) properly defined
            - Enum values match expected strings
        """
        assert RiskLevel.HIGH.value == "HIGH"
        assert RiskLevel.MEDIUM.value == "MEDIUM"
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.NONE.value == "NONE"

    def test_risk_level_comparison(self):
        """Test risk levels can be compared.

        Verifies:
            - Enum equality comparison works correctly
            - Different risk levels are distinguishable
        """
        assert RiskLevel.HIGH == RiskLevel.HIGH
        assert RiskLevel.HIGH != RiskLevel.LOW


class TestComplianceOutput:
    """Test suite for ComplianceOutput model."""

    def test_valid_compliance_output(self):
        """Test creation of valid compliance output.

        Verifies:
            - Full compliance assessment structure properly created
            - Nested RiskItem and Deficiency objects accessible
            - All risk and summary fields populated correctly
        """
        deficiency = Deficiency(
            title="Data Integrity Issue",
            cfr_reference="21 CFR 211.194",
            description="Inadequate audit trails",
            evidence="Missing timestamps in batch records",
            required_action="Implement comprehensive audit trail system",
        )

        risk = RiskItem(
            deficiency=deficiency,
            risk_level=RiskLevel.HIGH,
            reasoning="Input process lacks audit trail controls cited in CFR",
        )

        output = ComplianceOutput(
            risks=[risk],
            overall_risk=RiskLevel.HIGH,
            summary="Critical data integrity gaps identified",
        )

        assert len(output.risks) == 1
        assert output.overall_risk == RiskLevel.HIGH
        assert output.risks[0].deficiency.title == "Data Integrity Issue"

    def test_no_risks_identified(self):
        """Test compliance output with no risks.

        Verifies:
            - Empty risk list accepted
            - NONE risk level properly set
            - Model handles zero-risk scenario
        """
        output = ComplianceOutput(
            risks=[],
            overall_risk=RiskLevel.NONE,
            summary="No compliance issues identified",
        )
        assert len(output.risks) == 0
        assert output.overall_risk == RiskLevel.NONE


class TestWarningLetterMetadata:
    """Test suite for WarningLetterMetadata model."""

    def test_valid_metadata(self):
        """Test creation of valid warning letter metadata.

        Verifies:
            - All metadata fields properly initialized
            - URL validation works correctly
            - Optional fields accepted
        """
        metadata = WarningLetterMetadata(
            company_name="Test Pharma Inc.",
            issue_date="2024-01-15",
            url="https://www.fda.gov/inspections/warning-letters/test-pharma-inc",
            issuing_office="Office of Manufacturing Quality",
            subject="GMP Violations",
        )
        assert metadata.company_name == "Test Pharma Inc."
        assert "fda.gov" in str(metadata.url)

    def test_optional_fields(self):
        """Test that optional fields can be omitted.

        Verifies:
            - Optional fields (subject, ref_number) can be None
            - Model instantiation succeeds without optional fields
        """
        metadata = WarningLetterMetadata(
            company_name="Test Company",
            issue_date="2024-01-01",
            url="https://www.fda.gov/test",
            issuing_office="Test Office",
        )
        assert metadata.subject is None
        assert metadata.ref_number is None
