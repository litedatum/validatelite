#!/usr/bin/env python3
"""
Debug script to test SQLite desired_type validation
"""

import asyncio
import json
import tempfile
from pathlib import Path

from cli.app import cli_app
from click.testing import CliRunner

async def test_sqlite_validation():
    """Test SQLite validation with debug output"""
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        excel_path = tmp_path / "test_data.xlsx"
        schema_path = tmp_path / "test_schema.json"
        
        # Create test data
        import pandas as pd
        
        # Users table data
        users_data = {
            'user_id': [101, 102, 103, 104, 105, 106, 107],
            'name': [
                'Alice',           # ✓ Valid: length 5 <= 10
                'Bob',             # ✓ Valid: length 3 <= 10
                'Charlie',         # ✓ Valid: length 7 <= 10
                'David',           # ✓ Valid: length 5 <= 10
                'VeryLongName',    # ✗ Invalid: length 12 > 10
                'X',               # ✓ Valid: length 1 <= 10
                'TenCharName'      # ✗ Invalid: length 10 = 10 (should be valid)
            ],
            'age': [
                25,    # ✓ Valid: 2 digits
                30,    # ✓ Valid: 2 digits
                5,     # ✓ Valid: 1 digit
                99,    # ✓ Valid: 2 digits
                123,   # ✗ Invalid: 3 digits > 2
                8,     # ✓ Valid: 1 digit
                150    # ✗ Invalid: 3 digits > 2
            ],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com',
                     'david@test.com', 'eve@test.com', 'x@test.com', 'frank@test.com']
        }
        
        # Write to Excel file
        with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
            pd.DataFrame(users_data).to_excel(writer, sheet_name='users', index=False)
        
        # Create schema definition
        schema_definition = {
            "users": {
                "rules": [
                    { "field": "user_id", "type": "integer", "required": True },
                    { "field": "name", "type": "string", "required": True, "desired_type": "string(10)" },
                    { "field": "age", "type": "integer", "required": True, "desired_type": "integer(2)" },
                    { "field": "email", "type": "string", "required": True }
                ]
            }
        }
        
        with open(schema_path, 'w') as f:
            json.dump(schema_definition, f, indent=2)
        
        # Run validation
        runner = CliRunner()
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", str(excel_path), "--rules", str(schema_path), "--output", "json"]
        )
        
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        
        if result.exit_code == 0:
            payload = json.loads(result.output)
            print(f"Status: {payload.get('status')}")
            print(f"Fields: {json.dumps(payload.get('fields', []), indent=2)}")

if __name__ == "__main__":
    asyncio.run(test_sqlite_validation())
