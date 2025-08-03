-- Drop table if exists to allow for clean recreation
DROP TABLE IF EXISTS customers;

-- Create customers table with proper PostgreSQL structure
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    age INTEGER,
    gender INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add comment for gender field
COMMENT ON COLUMN customers.gender IS '0=female, 1=male, 3=invalid';

-- Add some indexes for better performance during testing
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_age ON customers(age);
CREATE INDEX idx_customers_gender ON customers(gender);
