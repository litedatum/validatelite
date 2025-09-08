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
from typing import List, Tuple, cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from shared.enums.connection_types import ConnectionType
from tests.shared.utils.database_utils import (
    get_available_databases,
    get_db_url,
    get_mysql_connection_params,
    get_postgresql_connection_params,
)


def generate_order_data(
    customer_count: int = 1000, orders_per_customer: int = 3
) -> List[Tuple]:
    """Generate test data with patterns for consistent test results."""
    # Products and statuses designed to work with schema validation tests
    products = [
        "Laptop",
        "Phone",
        "Tablet",
        "Mouse",
        "Keyboard",
        "Monitor",
        "Headphones",
        "Camera",
    ]
    # All statuses are <= 50 characters to fit status VARCHAR(50)
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]

    orders = []

    # Generate orders for customer IDs 1 through customer_count
    for customer_id in range(1, customer_count + 1):
        # Generate random number of orders per customer (1 to orders_per_customer)
        num_orders = random.randint(1, orders_per_customer)

        for _ in range(num_orders):
            product_name = f"{random.choice(products)} {random.randint(100, 999)}"
            # Generate reasonable quantities (business-valid range)
            quantity = random.randint(1, 10)
            # Generate valid prices that fit DECIMAL(10,2) constraints
            # All prices must be valid to avoid database insertion failures
            price = round(random.uniform(10.0, 999.99), 2)

            status = random.choice(statuses)
            # Generate valid dates avoiding invalid combinations (e.g., Feb 30)
            import calendar
            from datetime import date

            year = 2024
            month = random.randint(1, 12)
            # Get the maximum valid day for this month/year
            max_day = calendar.monthrange(year, month)[1]
            day = random.randint(1, max_day)
            order_date = date(year, month, day)

            # Always use valid customer IDs to avoid foreign key constraint failures
            orders.append(
                (customer_id, product_name, quantity, price, status, order_date)
            )

    # Add some orders with specific patterns for testing schema validation
    # These patterns should all be DATABASE-VALID (insertable)
    # but may have BUSINESS-LOGIC issues
    from datetime import date

    test_date = date(2024, 1, 15)
    test_patterns = [
        # Pattern 1: Valid data for baseline comparison
        (1, "Baseline Product", 1, 99.99, "pending", test_date),
        (2, "Test Product Alpha", 2, 149.50, "processing", test_date),
        (3, "Test Product Beta", 1, 299.99, "shipped", test_date),
        # Pattern 2: Edge case quantities (valid for DB, but may be business-invalid)
        (4, "Edge Case Product", 1, 0.01, "pending", test_date),  # Minimal price
        (5, "Edge Case Product", 100, 999.99, "delivered", test_date),  # High quantity
        # Pattern 3: Long but valid product names and statuses
        (
            6,
            "A" * 200 + " Product",
            1,
            199.99,
            "pending",
            test_date,
        ),  # Long but valid product name
        (7, "Test Product", 1, 99.99, "processing", test_date),  # Standard valid data
        # Pattern 4: Various valid price patterns that fit DECIMAL(10,2)
        (
            8,
            "Precision Test Product",
            1,
            12345678.99,
            "pending",
            test_date,
        ),  # Max valid DECIMAL(10,2)
        (9, "Small Price Product", 1, 0.01, "delivered", test_date),  # Min valid price
        (10, "Round Price Product", 5, 100.00, "cancelled", test_date),  # Round number
    ]

    orders.extend(test_patterns)

    return orders


def generate_customer_data(count: int = 1000) -> List[Tuple]:
    """
    Generate test customer data with specific patterns to
    ensure test cases pass/fail consistently.
    """
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

    # Ensure we have specific test patterns for failing test cases
    test_patterns = [
        # Pattern 1: NULL emails (for not_null test)
        (
            f"{random.choice(names)}1001",
            None,
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}1002",
            None,
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}1003",
            None,
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        # Pattern 2: Invalid email formats (for regex test)
        (
            f"{random.choice(names)}2001",
            "invalid-email",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}2002",
            "test@",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}2003",
            "@example.com",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}2004",
            "test..test@example.com",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}2005",
            "test@example",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        # Pattern 3: Duplicate emails (for unique email test)
        (
            f"{random.choice(names)}3001",
            "duplicate@example.com",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}3002",
            "duplicate@example.com",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}3003",
            "duplicate@example.com",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        # Pattern 6: Duplicate names (for unique name test)
        (
            "DuplicateName",
            f"unique1@{random.choice(domains)}",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            "DuplicateName",
            f"unique2@{random.choice(domains)}",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        (
            "DuplicateName",
            f"unique3@{random.choice(domains)}",
            random.randint(18, 65),
            random.choice([0, 1]),
        ),
        # Pattern 4: Invalid ages (for range test)
        (
            f"{random.choice(names)}4001",
            f"{random.choice(names).lower()}4001@{random.choice(domains)}",
            -10,
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}4002",
            f"{random.choice(names).lower()}4002@{random.choice(domains)}",
            150,
            random.choice([0, 1]),
        ),
        (
            f"{random.choice(names)}4003",
            f"{random.choice(names).lower()}4003@{random.choice(domains)}",
            200,
            random.choice([0, 1]),
        ),
        # Pattern 5: Invalid gender values (for enum test)
        (
            f"{random.choice(names)}5001",
            f"{random.choice(names).lower()}5001@{random.choice(domains)}",
            random.randint(18, 65),
            3,
        ),
        (
            f"{random.choice(names)}5002",
            f"{random.choice(names).lower()}5002@{random.choice(domains)}",
            random.randint(18, 65),
            None,
        ),
        (
            f"{random.choice(names)}5003",
            f"{random.choice(names).lower()}5003@{random.choice(domains)}",
            random.randint(18, 65),
            5,
        ),
    ]

    # Add the test patterns first
    customers.extend(test_patterns)

    # Generate remaining random data
    remaining_count = count - len(test_patterns)
    for i in range(remaining_count):
        name = f"{random.choice(names)}{random.randint(5000, 9999)}"
        email = f"{name.lower()}{random.randint(100, 999)}@{random.choice(domains)}"
        age = random.randint(18, 65)  # Valid age range
        gender = random.choice([0, 1])  # Valid gender values

        customers.append((name, email, age, gender))

    return customers


async def insert_test_data(
    engine: AsyncEngine, customers: List[Tuple], orders: List[Tuple]
) -> None:
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

        # Insert order data
        for customer_id, product_name, quantity, price, status, order_date in orders:
            await conn.execute(
                text(
                    """
                    INSERT INTO orders (customer_id, product_name, quantity,
                                        price, status, order_date)
                    VALUES (:customer_id, :product_name, :quantity,
                            :price, :status, :order_date)
                """
                ),
                {
                    "customer_id": customer_id,
                    "product_name": product_name,
                    "quantity": quantity,
                    "price": price,
                    "status": status,
                    "order_date": order_date,
                },
            )

        await conn.commit()


async def setup_mysql_database() -> None:
    """Setup MySQL database with schema and test data."""
    # Get MySQL URL from environment variables
    connection_params = get_mysql_connection_params()

    db_url = get_db_url(
        db_type=ConnectionType.MYSQL,
        database=str(connection_params["database"]),
        username=str(connection_params["username"]),
        password=str(connection_params["password"]),
        host=str(connection_params["host"]),
        port=cast(int, connection_params["port"]),
    )
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
        orders = generate_order_data(1000, 3)
        await insert_test_data(engine, customers, orders)

        print(
            f"[SUCCESS] MySQL database setup completed. "
            f"Inserted {len(customers)} customers and {len(orders)} orders."
        )

    finally:
        await engine.dispose()


async def setup_postgresql_database() -> None:
    """Setup PostgreSQL database with schema and test data."""
    # Get PostgreSQL URL from environment variables
    connection_params = get_postgresql_connection_params()
    db_url = get_db_url(
        db_type=ConnectionType.POSTGRESQL,
        database=str(connection_params["database"]),
        username=str(connection_params["username"]),
        password=str(connection_params["password"]),
        host=str(connection_params["host"]),
        port=cast(int, connection_params["port"]),
    )

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
        orders = generate_order_data(1000, 3)
        await insert_test_data(engine, customers, orders)

        print(
            "[SUCCESS] PostgreSQL database setup completed. "
            f"Inserted {len(customers)} customers and {len(orders)} orders."
        )

    finally:
        await engine.dispose()


async def main() -> None:
    """Main function to setup available databases."""
    print("[INFO] Starting database setup for CI/CD pipeline...")

    # Get available databases
    available_databases = get_available_databases()
    print(f"[INFO] Available databases: {', '.join(available_databases)}")

    # Setup MySQL database if available
    if "mysql" in available_databases:
        print("[INFO] Setting up MySQL database...")
        try:
            await setup_mysql_database()
        except Exception as e:
            print(f"[ERROR] MySQL setup failed: {e}")
            sys.exit(1)
    else:
        print("[INFO] Skipping MySQL setup (not configured)")

    # Setup PostgreSQL database if available
    if "postgresql" in available_databases:
        print("[INFO] Setting up PostgreSQL database...")
        try:
            await setup_postgresql_database()
        except Exception as e:
            print(f"[ERROR] PostgreSQL setup failed: {e}")
            sys.exit(1)
    else:
        print("[INFO] Skipping PostgreSQL setup (not configured)")

    print("[SUCCESS] Database setup completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
