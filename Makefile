.PHONY: install test coverage run docker-build docker-run clean

# Install dependencies
install:
	poetry install

# Run tests
test:
	poetry run python run_tests.py

# Run tests with coverage
coverage:
	poetry run pytest --cov=src tests/

# Start the application
run:
	poetry run python scripts/start_app.py

# Build Docker image
docker-build:
	docker build -t financial-ai-agent .

# Run Docker container
docker-run:
	docker-compose up

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name "coverage_html" -exec rm -r {} +
	find . -type f -name ".coverage" -delete
