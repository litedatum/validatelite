#!/bin/bash

# Setup script for local test databases
# This script helps set up MySQL and PostgreSQL databases for local testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MYSQL_PASSWORD=${MYSQL_ROOT_PASSWORD:-"test_password"}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-"test_password"}
MYSQL_PORT=${MYSQL_PORT:-3306}
POSTGRES_PORT=${POSTGRES_PORT:-5432}

echo -e "${GREEN}Setting up test databases for validatelite...${NC}"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi
}

# Function to check if port is available
check_port() {
    local port=$1
    local service=$2

    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $port is already in use. $service might not start properly.${NC}"
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to start MySQL
start_mysql() {
    echo -e "${GREEN}Starting MySQL container...${NC}"

    # Stop existing container if running
    docker stop validatelite-mysql 2>/dev/null || true
    docker rm validatelite-mysql 2>/dev/null || true

    # Start new container
    docker run -d \
        --name validatelite-mysql \
        -e MYSQL_ROOT_PASSWORD="$MYSQL_PASSWORD" \
        -e MYSQL_DATABASE=test_db \
        -p "$MYSQL_PORT:3306" \
        mysql:8.0 \
        --default-authentication-plugin=mysql_native_password

    echo -e "${GREEN}MySQL container started on port $MYSQL_PORT${NC}"
}

# Function to start PostgreSQL
start_postgres() {
    echo -e "${GREEN}Starting PostgreSQL container...${NC}"

    # Stop existing container if running
    docker stop validatelite-postgres 2>/dev/null || true
    docker rm validatelite-postgres 2>/dev/null || true

    # Start new container
    docker run -d \
        --name validatelite-postgres \
        -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        -e POSTGRES_DB=test_db \
        -p "$POSTGRES_PORT:5432" \
        postgres:15

    echo -e "${GREEN}PostgreSQL container started on port $POSTGRES_PORT${NC}"
}

# Function to wait for database to be ready
wait_for_database() {
    local service=$1
    local port=$2
    local password=$3

    echo -e "${YELLOW}Waiting for $service to be ready...${NC}"

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if [ "$service" = "MySQL" ]; then
            if docker exec validatelite-mysql mysqladmin ping -h localhost -u root -p"$password" --silent 2>/dev/null; then
                echo -e "${GREEN}$service is ready!${NC}"
                return 0
            fi
        elif [ "$service" = "PostgreSQL" ]; then
            if docker exec validatelite-postgres pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; then
                echo -e "${GREEN}$service is ready!${NC}"
                return 0
            fi
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Error: $service failed to start within 60 seconds${NC}"
    return 1
}

# Function to test database connections
test_connections() {
    echo -e "${GREEN}Testing database connections...${NC}"

    # Test MySQL
    if docker exec validatelite-mysql mysql -h localhost -u root -p"$MYSQL_PASSWORD" -e "SELECT 1;" test_db >/dev/null 2>&1; then
        echo -e "${GREEN}✓ MySQL connection successful${NC}"
    else
        echo -e "${RED}✗ MySQL connection failed${NC}"
        return 1
    fi

    # Test PostgreSQL
    if docker exec validatelite-postgres psql -h localhost -U postgres -d test_db -c "SELECT 1;" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PostgreSQL connection successful${NC}"
    else
        echo -e "${RED}✗ PostgreSQL connection failed${NC}"
        return 1
    fi
}

# Function to show connection information
show_connection_info() {
    echo -e "${GREEN}Database setup complete!${NC}"
    echo
    echo -e "${YELLOW}Connection Information:${NC}"
    echo "MySQL:"
    echo "  Host: localhost"
    echo "  Port: $MYSQL_PORT"
    echo "  User: root"
    echo "  Password: $MYSQL_PASSWORD"
    echo "  Database: test_db"
    echo "  URL: mysql://root:$MYSQL_PASSWORD@localhost:$MYSQL_PORT/test_db"
    echo
    echo "PostgreSQL:"
    echo "  Host: localhost"
    echo "  Port: $POSTGRES_PORT"
    echo "  User: postgres"
    echo "  Password: $POSTGRES_PASSWORD"
    echo "  Database: test_db"
    echo "  URL: postgresql://postgres:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/test_db"
    echo
    echo -e "${YELLOW}Environment variables for testing:${NC}"
    echo "export MYSQL_ROOT_PASSWORD=$MYSQL_PASSWORD"
    echo "export POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
    echo "export MYSQL_DB_URL=mysql://root:$MYSQL_PASSWORD@localhost:$MYSQL_PORT/test_db"
    echo "export POSTGRESQL_DB_URL=postgresql://postgres:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/test_db"
}

# Function to cleanup
cleanup() {
    echo -e "${YELLOW}Cleaning up containers...${NC}"
    docker stop validatelite-mysql validatelite-postgres 2>/dev/null || true
    docker rm validatelite-mysql validatelite-postgres 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Main script logic
main() {
    case "${1:-start}" in
        "start")
            check_docker
            check_port $MYSQL_PORT "MySQL"
            check_port $POSTGRES_PORT "PostgreSQL"

            start_mysql
            start_postgres

            wait_for_database "MySQL" $MYSQL_PORT $MYSQL_PASSWORD
            wait_for_database "PostgreSQL" $POSTGRES_PORT $POSTGRES_PASSWORD

            test_connections
            show_connection_info
            ;;
        "stop")
            cleanup
            ;;
        "restart")
            cleanup
            sleep 2
            main start
            ;;
        "status")
            echo -e "${YELLOW}Container Status:${NC}"
            docker ps --filter "name=validatelite-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status}"
            echo
            echo "Commands:"
            echo "  start   - Start test databases (default)"
            echo "  stop    - Stop and remove containers"
            echo "  restart - Restart containers"
            echo "  status  - Show container status"
            echo
            echo "Environment variables:"
            echo "  MYSQL_ROOT_PASSWORD - MySQL root password (default: test_password)"
            echo "  POSTGRES_PASSWORD   - PostgreSQL password (default: test_password)"
            echo "  MYSQL_PORT         - MySQL port (default: 3306)"
            echo "  POSTGRES_PORT      - PostgreSQL port (default: 5432)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
