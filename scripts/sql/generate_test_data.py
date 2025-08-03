#!/usr/bin/env python3
"""
Test data generation script for CI/CD pipeline.

This script generates test data for both MySQL and PostgreSQL databases
to be used in E2E and integration tests.
"""

import asyncio
import os
import random
import sys
from typing import List, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from tests.shared.utils.database_utils import (
    get_available_databases,
    get_mysql_test_url,
    get_postgresql_test_url,
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def generate_customer_data(count: int = 1000) -> List[Tuple]:
    """Generate test customer data."""
    names = [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eve",
        "Frank",
        "Grace",
        "Helen",
        "Ivy",
        "Jack",
        "Yang",
        "Fan",
        "Emy",
        "Tom",
        "Charles",
        "Huhansan",
    ]

    domains = ["example.com", "test.org", "mail.com", "sample.net"]

    customers = []
    for i in range(count):
        name = f"{random.choice(names)}{random.randint(1000, 9999)}"
        email = f"{name.lower()}{random.randint(100, 999)}@{random.choice(domains)}"
        age = random.randint(-50, 200)  # Include some invalid ages for testing
        gender = random.choice([0, 1, 3, None])  # Include invalid values

        customers.append((name, email, age, gender))

    return customers


async def insert_test_data(engine: AsyncEngine, customers: List[Tuple]) -> None:
    """Insert test data into the database."""
    async with engine.connect() as conn:
        # Insert customer data
        for name, email, age, gender in customers:
            await conn.execute(
                text(
                    """
                    INSERT INTO customers (name, email, age, gender, created_at)
                    VALUES (:name, :email, :age, :gender, CURRENT_TIMESTAMP)
                """
                ),
                {"name": name, "email": email, "age": age, "gender": gender},
            )

        await conn.commit()


async def setup_mysql_database() -> None:
    """Setup MySQL database with schema and test data."""
    # Get MySQL URL from environment variables
    db_url = get_mysql_test_url()

    # Create engine
    engine = create_async_engine(db_url, echo=False)

    try:
        # Read and execute schema
        schema_path = os.path.join(
            os.path.dirname(__file__), "mysql_customers_schema.sql"
        )
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        async with engine.connect() as conn:
            # Execute schema creation
            for statement in schema_sql.split(";"):
                if statement.strip():
                    await conn.execute(text(statement))
            await conn.commit()

        # Generate and insert test data
        customers = generate_customer_data(1000)
        await insert_test_data(engine, customers)

        print(
            f"✅ MySQL database setup completed. Inserted {len(customers)} customers."
        )

    finally:
        await engine.dispose()


async def setup_postgresql_database() -> None:
    """Setup PostgreSQL database with schema and test data."""
    # Get PostgreSQL URL from environment variables
    db_url = get_postgresql_test_url()

    # Create engine
    engine = create_async_engine(db_url, echo=False)

    try:
        # Read and execute schema
        schema_path = os.path.join(
            os.path.dirname(__file__), "postgresql_customers_schema.sql"
        )
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        async with engine.connect() as conn:
            # Execute schema creation
            for statement in schema_sql.split(";"):
                if statement.strip():
                    await conn.execute(text(statement))
            await conn.commit()

        # Generate and insert test data
        customers = generate_customer_data(1000)
        await insert_test_data(engine, customers)

        print(
            "✅ PostgreSQL database setup completed. "
            f"Inserted {len(customers)} customers."
        )

    finally:
        await engine.dispose()


async def main() -> None:
    """Main function to setup available databases."""
    print("🚀 Starting database setup for CI/CD pipeline...")

    # Get available databases
    available_databases = get_available_databases()
    print(f"📋 Available databases: {', '.join(available_databases)}")

    # Setup MySQL database if available
    if "mysql" in available_databases:
        print("📦 Setting up MySQL database...")
        try:
            await setup_mysql_database()
        except Exception as e:
            print(f"❌ MySQL setup failed: {e}")
            sys.exit(1)
    else:
        print("⏭️  Skipping MySQL setup (not configured)")

    # Setup PostgreSQL database if available
    if "postgresql" in available_databases:
        print("📦 Setting up PostgreSQL database...")
        try:
            await setup_postgresql_database()
        except Exception as e:
            print(f"❌ PostgreSQL setup failed: {e}")
            sys.exit(1)
    else:
        print("⏭️  Skipping PostgreSQL setup (not configured)")

    print("🎉 Database setup completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
