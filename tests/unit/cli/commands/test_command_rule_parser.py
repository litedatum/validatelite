"""
🧙‍♂️ Rule Parser TDD Tests - Modern Testing Architecture

Features:
- Schema Builder Pattern for zero boilerplate
- Contract Testing for Mock consistency
- Property-based Testing for edge cases
- Comprehensive boundary condition testing
- Exception flow coverage
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cli.core.rule_parser import RuleParser
from cli.exceptions import RuleParsingError, ValidationError
from shared.enums import RuleCategory, RuleType, SeverityLevel
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


class TestRuleParser:
    """Modern Rule Parser Test Suite - Testing Ghost's Architecture"""

    @pytest.fixture
    def parser(self) -> RuleParser:
        """Rule parser instance"""
        return RuleParser()

    @pytest.fixture
    def valid_rules_file(self) -> str:
        """Valid rules file fixture using Builder Pattern"""
        rules_data = {
            "version": "1.0",
            "rules": [
                {
                    "type": "not_null",
                    "column": "id",
                    "description": "Primary key validation",
                },
                {
                    "type": "length",
                    "column": "name",
                    "min": 2,
                    "max": 50,
                    "description": "Name length validation",
                },
                {
                    "type": "unique",
                    "column": "email",
                    "description": "Email uniqueness",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rules_data, f)
            temp_file = f.name

        return temp_file

    # === NORMAL FLOW TESTS ===

    def test_parse_single_inline_rule_success(self, parser: RuleParser) -> None:
        """Test single inline rule parsing: --rule 'not_null(id)'"""
        rule_expr = "not_null(id)"

        rules = parser.parse_rules(inline_rules=[rule_expr])

        assert len(rules) == 1
        rule = rules[0]
        assert rule.type == RuleType.NOT_NULL
        assert rule.target.entities[0].column == "id"
        assert rule.category == RuleCategory.COMPLETENESS

    def test_parse_multiple_inline_rules_success(self, parser: RuleParser) -> None:
        """Test multiple inline rules parsing"""
        rule_exprs = ["not_null(id)", "unique(email)", "length(name,2,50)"]

        rules = parser.parse_rules(inline_rules=rule_exprs)

        assert len(rules) == 3
        assert rules[0].type == RuleType.NOT_NULL
        assert rules[1].type == RuleType.UNIQUE
        assert rules[2].type == RuleType.LENGTH
        assert rules[2].parameters["min_length"] == 2
        assert rules[2].parameters["max_length"] == 50

    def test_parse_rules_file_success(
        self, parser: RuleParser, valid_rules_file: str
    ) -> None:
        """Test rules file parsing success"""
        rules = parser.parse_rules(rules_file=valid_rules_file)

        assert len(rules) == 3
        assert any(rule.type == RuleType.NOT_NULL for rule in rules)
        assert any(rule.type == RuleType.LENGTH for rule in rules)
        assert any(rule.type == RuleType.UNIQUE for rule in rules)

    def test_parse_complex_rule_parameters(self, parser: RuleParser) -> None:
        """Test complex rule parameter parsing"""
        rule_expr = 'regex(email,"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")'

        rules = parser.parse_rules(inline_rules=[rule_expr])

        assert len(rules) == 1
        rule = rules[0]
        assert rule.type == RuleType.REGEX
        assert "pattern" in rule.parameters
        assert "@" in rule.parameters["pattern"]

    def test_regex_pattern_escaping_fix(self, parser: RuleParser) -> None:
        """Test regex pattern escaping fix for command line"""
        # Verify the fix for command-line escaping issues.
        # The user provides a regular expression designed to validate email addresses.  The regex, used with the presumably existing function `regex(email, pattern)`, checks if the input `email` string matches the pattern  `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`. This pattern requires one or more alphanumeric characters, periods, underscores, percentage signs, plus or minus signs before the "@" symbol, followed by one or more alphanumeric characters, periods, or hyphens, followed by a period and two or more alphabetic characters.
        # Expected behavior: Correct handling of double escaping.
        rule_expr = 'regex(email,"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")'

        rules = parser.parse_rules(inline_rules=[rule_expr])

        assert len(rules) == 1
        rule = rules[0]
        pattern = rule.parameters["pattern"]

        # Verify correct handling of escape sequences/characters.
        assert "\\\\" not in pattern  # Double escaping should be avoided.
        assert "\\." in pattern  # Periods/full stops should be escaped.
        assert "{2,}" in pattern  # There should be a quantifier.

        # Verify that the regular expression is valid.
        import re

        re.compile(
            pattern
        )  # Exceptions should not be thrown.  Or, more contextually, No exceptions should be thrown.

    def test_regex_pattern_without_quotes(self, parser: RuleParser) -> None:
        """Test regex pattern without quotes"""
        # Test the regular expression without quotes.
        rule_expr = "regex(email,^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$)"

        rules = parser.parse_rules(inline_rules=[rule_expr])

        assert len(rules) == 1
        rule = rules[0]
        pattern = rule.parameters["pattern"]

        # Verify correct handling of escape sequences.
        assert "\\\\" not in pattern
        assert "\\." in pattern

    # === BOUNDARY CONDITION TESTS ===

    def test_empty_rule_list_handling(self, parser: RuleParser) -> None:
        """Test empty rule list handling - should raise ValidationError"""
        with pytest.raises(ValidationError, match="No valid rules parsed"):
            parser.parse_rules(inline_rules=[])

    def test_whitespace_rule_handling(self, parser: RuleParser) -> None:
        """Test whitespace in rule expressions"""
        rule_exprs = [
            "  not_null(id)  ",  # Leading/trailing spaces
            "unique( email )",  # Spaces in parameters
            "length(name, 2, 50)",  # Spaces after commas
        ]

        rules = parser.parse_rules(inline_rules=rule_exprs)

        assert len(rules) == 3
        assert all(rule.target.entities[0].column for rule in rules)

    def test_case_insensitive_rule_types(self, parser: RuleParser) -> None:
        """Test case insensitive rule type handling"""
        rule_exprs = [
            "NOT_NULL(id)",
            "not_null(id)",
            "Not_Null(id)",
            "UNIQUE(email)",
            "unique(email)",
        ]

        rules = parser.parse_rules(inline_rules=rule_exprs)

        assert len(rules) == 5
        assert all(rule.type in [RuleType.NOT_NULL, RuleType.UNIQUE] for rule in rules)

    @given(
        st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_based_column_names(
        self, parser: RuleParser, column_name: str
    ) -> None:
        """Property-based test for various column names"""
        # Skip problematic characters that would break rule syntax
        if any(char in column_name for char in ["(", ")", ",", '"', "'"]):
            return

        rule_expr = f"not_null({column_name.strip()})"

        try:
            rules = parser.parse_rules(inline_rules=[rule_expr])
            if rules:  # If parsing succeeded
                assert len(rules) == 1
                assert rules[0].target.entities[0].column == column_name.strip()
        except (ValidationError, RuleParsingError):
            # Expected for invalid column names
            pass

    # === ERROR HANDLING TESTS ===

    def test_invalid_rule_syntax_error(self, parser: RuleParser) -> None:
        """Test invalid rule syntax error handling"""
        # Test each invalid rule separately to avoid issues with loop execution
        with pytest.raises(RuleParsingError):
            parser.parse_rules(
                inline_rules=["not_null)"]
            )  # Missing opening parenthesis

        with pytest.raises(RuleParsingError):
            parser.parse_rules(
                inline_rules=["not_null(id"]
            )  # Missing closing parenthesis

        with pytest.raises(RuleParsingError):
            parser.parse_rules(inline_rules=["not_null()"])  # Missing column name

        with pytest.raises(RuleParsingError):
            parser.parse_rules(inline_rules=["(id)"])  # Missing rule type

        with pytest.raises(RuleParsingError):
            parser.parse_rules(inline_rules=[""])  # Empty rule

    def test_unsupported_rule_type_error(self, parser: RuleParser) -> None:
        """Test unsupported rule type error"""
        with pytest.raises(RuleParsingError):
            parser.parse_rules(inline_rules=["custom_rule(column)"])

    def test_rule_parameter_validation_errors(self, parser: RuleParser) -> None:
        """Test rule parameter validation errors"""
        # Test each invalid parameter rule separately
        with pytest.raises(RuleParsingError):
            parser.parse_rules(
                inline_rules=["length(name,-1,50)"]
            )  # Negative min length

        with pytest.raises(RuleParsingError):
            parser.parse_rules(inline_rules=["length(name,50,10)"])  # Min > Max

        with pytest.raises(RuleParsingError):
            parser.parse_rules(
                inline_rules=["range(age,150,100)"]
            )  # Min > Max for range

    def test_malformed_json_rules_file_error(self, parser: RuleParser) -> None:
        """Test malformed JSON rules file error"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json syntax}')  # Invalid JSON
            malformed_file = f.name

        try:
            with pytest.raises(ValidationError, match="Invalid JSON in rules file"):
                parser.parse_rules(rules_file=malformed_file)
        finally:
            Path(malformed_file).unlink(missing_ok=True)

    def test_missing_rules_file_error(self, parser: RuleParser) -> None:
        """Test missing rules file error"""
        nonexistent_file = "/path/to/nonexistent/rules.json"

        with pytest.raises(FileNotFoundError, match="Rules file not found"):
            parser.parse_rules(rules_file=nonexistent_file)

    def test_invalid_rules_file_schema_error(self, parser: RuleParser) -> None:
        """Test invalid rules file schema error"""
        invalid_schema = {
            "wrong_version": "2.0",  # Wrong field name
            "invalid_rules": [{"invalid": "structure"}],  # Wrong field name
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_schema, f)
            invalid_file = f.name

        try:
            with pytest.raises(
                ValidationError, match="Rules file must contain a 'rules' array"
            ):
                parser.parse_rules(rules_file=invalid_file)
        finally:
            Path(invalid_file).unlink(missing_ok=True)

    # === EDGE CASES AND SPECIAL SCENARIOS ===

    def test_unicode_column_names(self, parser: RuleParser) -> None:
        """Test Unicode column names support"""
        unicode_rules = [
            "not_null(用户名)",  # Chinese characters
            "unique(correo_electrónico)",  # Spanish with accents
            "length(название,1,100)",  # Cyrillic
        ]

        rules = parser.parse_rules(inline_rules=unicode_rules)

        assert len(rules) == 3
        assert rules[0].target.entities[0].column == "用户名"
        assert rules[1].target.entities[0].column == "correo_electrónico"
        assert rules[2].target.entities[0].column == "название"

    def test_special_characters_in_column_names(self, parser: RuleParser) -> None:
        """Test special characters in column names"""
        special_char_rules = [
            'not_null("user-name")',  # Hyphen in quotes
            'unique("user_email")',  # Underscore in quotes
            'length("user.name",1,50)',  # Dot in name
        ]

        rules = parser.parse_rules(inline_rules=special_char_rules)

        assert len(rules) == 3
        assert (
            rules[0].target.entities[0].column == "user-name"
        )  # Quotes should be stripped
        assert rules[1].target.entities[0].column == "user_email"
        assert rules[2].target.entities[0].column == "user.name"

    def test_extremely_long_rule_expressions(self, parser: RuleParser) -> None:
        """Test extremely long rule expressions"""
        # Very long column name
        long_column = "a" * 1000
        rule_expr = f"not_null({long_column})"

        # This should fail due to length limitations
        with pytest.raises(RuleParsingError, match="Invalid rule syntax"):
            parser.parse_rules(inline_rules=[rule_expr])

    def test_concurrent_rule_parsing(self, parser: RuleParser) -> None:
        """Test concurrent rule parsing safety"""
        import threading
        import time

        results = []
        errors = []

        def parse_rules_thread(thread_id: int) -> None:
            try:
                rules = parser.parse_rules(
                    inline_rules=[f"not_null(column_{thread_id})"]
                )
                results.append((thread_id, len(rules)))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(10):
            thread = threading.Thread(target=parse_rules_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Concurrent parsing errors: {errors}"
        assert len(results) == 10
        assert all(count == 1 for _, count in results)

    # === INTEGRATION WITH BUILDER PATTERN ===

    def test_integration_with_test_data_builder(self, parser: RuleParser) -> None:
        """Test integration with TestDataBuilder"""
        # Parse rules and validate they can be built properly
        rules = parser.parse_rules(inline_rules=["not_null(id)", "unique(email)"])

        # Verify rules can be enhanced using Builder pattern
        enhanced_rules = []
        for rule in rules:
            column = rule.target.entities[0].column
            if column is not None:  # Ensure column is not None
                enhanced_rule = (
                    TestDataBuilder.rule()
                    .with_name(rule.name)
                    .with_type(rule.type)
                    .with_target("test_db", "test_table", column)
                    .build()
                )
                enhanced_rules.append(enhanced_rule)

        assert len(enhanced_rules) == 2
        assert enhanced_rules[0].type == RuleType.NOT_NULL
        assert enhanced_rules[1].type == RuleType.UNIQUE

    def test_rule_parser_with_mock_contract(self, parser: RuleParser) -> None:
        """Test rule parser with mock contracts"""
        # This test demonstrates how Contract Testing would work
        # when RuleParser interacts with other components

        # Create a mock rule result
        mock_rule = TestDataBuilder.rule().as_not_null_rule().build()

        # Test with direct parsing instead of mocking non-existent validator
        rules = parser.parse_rules(inline_rules=["not_null(id)"])

        # Verify contract compliance
        assert len(rules) == 1
        assert rules[0].type == mock_rule.type

    # === PERFORMANCE TESTS ===

    def test_large_rules_file_performance(self, parser: RuleParser) -> None:
        """Test performance with large rules file"""
        # Create a large rules file with 1000 rules
        large_rules_data = {
            "version": "1.0",
            "rules": [
                {
                    "type": "not_null",
                    "column": f"column_{i}",
                    "description": f"Rule {i}",
                }
                for i in range(1000)
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(large_rules_data, f)
            large_file = f.name

        try:
            import time

            start_time = time.time()
            rules = parser.parse_rules(rules_file=large_file)
            end_time = time.time()

            # Should parse 1000 rules in reasonable time (< 1 second)
            assert len(rules) == 1000
            assert (
                end_time - start_time
            ) < 1.0, f"Parsing took {end_time - start_time:.2f}s, expected < 1.0s"
        finally:
            Path(large_file).unlink(missing_ok=True)

    def test_memory_usage_with_many_inline_rules(self, parser: RuleParser) -> None:
        """Test memory usage with many inline rules"""
        # Create 500 inline rules
        inline_rules = [f"not_null(column_{i})" for i in range(500)]

        import os

        import psutil

        process = psutil.Process(os.getpid())

        memory_before = process.memory_info().rss
        rules = parser.parse_rules(inline_rules=inline_rules)
        memory_after = process.memory_info().rss

        memory_increase = memory_after - memory_before

        assert len(rules) == 500
        # Memory increase should be reasonable (< 10MB for 500 rules)
        assert (
            memory_increase < 10 * 1024 * 1024
        ), f"Memory increase: {memory_increase / 1024 / 1024:.2f}MB"
