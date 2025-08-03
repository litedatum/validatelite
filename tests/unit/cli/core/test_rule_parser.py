"""
Unit tests for RuleParser
"""

import pytest

from cli.core.rule_parser import RuleParser
from cli.exceptions import RuleParsingError, ValidationError
from shared.enums import RuleCategory, RuleType, SeverityLevel


class TestRuleParser:
    """Test cases for RuleParser"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.parser = RuleParser()

    def test_parse_inline_rule_not_null(self) -> None:
        """Test parsing NOT_NULL rule"""
        rule_expr = "not_null(id)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.type == RuleType.NOT_NULL
        assert rule.target.entities[0].column == "id"
        assert rule.parameters == {}

    def test_parse_inline_rule_unique(self) -> None:
        """Test parsing UNIQUE rule"""
        rule_expr = "unique(email)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.type == RuleType.UNIQUE
        assert rule.target.entities[0].column == "email"
        assert rule.parameters == {}

    def test_parse_inline_rule_length(self) -> None:
        """Test parsing LENGTH rule"""
        rule_expr = "length(name,2,50)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.type == RuleType.LENGTH
        assert rule.target.entities[0].column == "name"
        assert rule.parameters["min_length"] == 2
        assert rule.parameters["max_length"] == 50

    def test_parse_inline_rule_range(self) -> None:
        """Test parsing RANGE rule"""
        rule_expr = "range(age,18,65)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.type == RuleType.RANGE
        assert rule.target.entities[0].column == "age"
        assert rule.parameters["min_value"] == 18.0
        assert rule.parameters["max_value"] == 65.0

    def test_parse_inline_rule_enum(self) -> None:
        """Test parsing ENUM rule"""
        rule_expr = "enum(status,active,inactive)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.type == RuleType.ENUM
        assert rule.target.entities[0].column == "status"
        assert rule.parameters["allowed_values"] == ["active", "inactive"]

    def test_parse_inline_rule_regex(self) -> None:
        """Test parsing REGEX rule"""
        rule_expr = "regex(email,^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.type == RuleType.REGEX
        assert rule.target.entities[0].column == "email"
        assert "pattern" in rule.parameters

    def test_enum_rule_parameter_type_conversion(self) -> None:
        """Test that enum rule parameters are converted to appropriate types"""
        # Test with numeric values
        rule_expr = "enum(gender,0,1)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.type == RuleType.ENUM
        assert rule.target.entities[0].column == "gender"
        assert rule.parameters["allowed_values"] == [
            0,
            1,
        ]  # Should be integers, not strings

        # Test with mixed types
        rule_expr = "enum(status,active,1,inactive,0)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.parameters["allowed_values"] == [
            "active",
            1,
            "inactive",
            0,
        ]  # Mixed types

        # Test with quoted strings
        rule_expr = "enum(category,'A','B','C')"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.parameters["allowed_values"] == ["A", "B", "C"]  # Should be strings

        # Test with negative numbers
        rule_expr = "enum(score,-1,0,1)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.parameters["allowed_values"] == [-1, 0, 1]  # Should be integers

        # Test with floats
        rule_expr = "enum(rate,0.0,0.5,1.0)"
        rule = self.parser._parse_inline_rule(rule_expr)

        assert rule.parameters["allowed_values"] == [0.0, 0.5, 1.0]  # Should be floats

    def test_invalid_rule_syntax(self) -> None:
        """Test handling of invalid rule syntax"""
        with pytest.raises(RuleParsingError):
            self.parser._parse_inline_rule("invalid_rule")

        with pytest.raises(
            ValidationError, match="Rule must specify at least a column name"
        ):
            self.parser._parse_inline_rule("not_null()")

    def test_unsupported_rule_type(self) -> None:
        """Test handling of unsupported rule type"""
        with pytest.raises(ValidationError, match="Unsupported rule type"):
            self.parser._parse_inline_rule("unsupported_type(column)")

    def test_enum_rule_missing_values(self) -> None:
        """Test ENUM rule with missing values"""
        with pytest.raises(
            ValidationError, match="Enum rule requires at least one allowed value"
        ):
            self.parser._parse_inline_rule("enum(column)")

    def test_length_rule_invalid_parameters(self) -> None:
        """Test LENGTH rule with invalid parameters"""
        with pytest.raises(ValidationError, match="Minimum length cannot be negative"):
            self.parser._parse_inline_rule("length(name,-1,50)")

        with pytest.raises(
            ValidationError, match="Maximum length cannot be less than minimum length"
        ):
            self.parser._parse_inline_rule("length(name,50,10)")

    def test_range_rule_invalid_parameters(self) -> None:
        """Test RANGE rule with invalid parameters"""
        with pytest.raises(
            ValidationError, match="Maximum value cannot be less than minimum value"
        ):
            self.parser._parse_inline_rule("range(age,100,50)")

    def test_parse_multiple_rules(self) -> None:
        """Test parsing multiple inline rules"""
        rules = [
            "not_null(id)",
            "unique(email)",
            "length(name,2,50)",
            "enum(gender,0,1)",
        ]

        parsed_rules = self.parser.parse_rules(inline_rules=rules)

        assert len(parsed_rules) == 4
        assert parsed_rules[0].type == RuleType.NOT_NULL
        assert parsed_rules[1].type == RuleType.UNIQUE
        assert parsed_rules[2].type == RuleType.LENGTH
        assert parsed_rules[3].type == RuleType.ENUM
        assert parsed_rules[3].parameters["allowed_values"] == [0, 1]

    def test_parse_rules_with_no_rules(self) -> None:
        """Test parsing with no rules provided"""
        with pytest.raises(ValidationError, match="No valid rules parsed"):
            self.parser.parse_rules()
