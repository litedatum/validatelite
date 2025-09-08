-- Drop tables if exists to allow for clean recreation (orders first due to FK constraint)
DROP TABLE IF EXISTS orders;
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

-- Orders table will be created after customers table

-- Create orders table with proper PostgreSQL structure
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) NOT NULL,
    order_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- Add comment for status field
COMMENT ON COLUMN orders.status IS 'Order status: pending, processing, shipped, delivered, cancelled';

-- Add indexes for orders table
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_order_date ON orders(order_date);
