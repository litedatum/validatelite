-- Drop table if exists to allow for clean recreation
DROP TABLE IF EXISTS customers;

-- Create customers table with proper MySQL structure
CREATE TABLE customers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255),
    email VARCHAR(255),
    age INT,
    gender INT COMMENT '0=female, 1=male, 3=invalid',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add some indexes for better performance during testing
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_age ON customers(age);
CREATE INDEX idx_customers_gender ON customers(gender);
