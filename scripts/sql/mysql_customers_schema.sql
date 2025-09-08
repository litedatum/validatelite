-- Drop tables if exists to allow for clean recreation (orders first due to FK constraint)
DROP TABLE IF EXISTS orders;
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

-- Orders table will be created after customers table

-- Create orders table with proper MySQL structure
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    quantity INT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) NOT NULL,
    order_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- Add indexes for orders table
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_order_date ON orders(order_date);
