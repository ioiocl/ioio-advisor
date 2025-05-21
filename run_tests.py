import pytest
import coverage
import os
import sys

def run_tests_with_coverage():
    """Run all tests with coverage reporting."""
    
    # Start coverage measurement
    cov = coverage.Coverage(
        source=["src"],
        omit=[
            "*/tests/*",
            "*/migrations/*",
            "*/__init__.py"
        ]
    )
    cov.start()
    
    # Run tests
    test_args = [
        "--verbose",
        "--asyncio-mode=auto",
        "-v",
        "tests/unit",
        "tests/integration",
        "tests/system"
    ]
    
    result = pytest.main(test_args)
    
    # Stop coverage measurement
    cov.stop()
    cov.save()
    
    # Generate reports
    print("\nCoverage Summary:")
    cov.report()
    
    # Generate HTML report
    html_dir = "coverage_html"
    cov.html_report(directory=html_dir)
    print(f"\nDetailed HTML coverage report generated in: {html_dir}/index.html")
    
    return result

if __name__ == "__main__":
    result = run_tests_with_coverage()
    sys.exit(result)
