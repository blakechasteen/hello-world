"""
DATAPIG Visual Dashboard Demo

Demonstrates Tufte-style HTML report generation for data quality analysis.

**Features Demonstrated**:
- Analyzing multiple datasets
- Generating quality reports
- Rendering small multiples
- Data density tables
- Complete dashboard with trends

**Output**: demos/output/datapig_dashboard.html

Run:
    PYTHONPATH=. python demos/demo_datapig_dashboard.py
"""

import time
import os
from pathlib import Path

# DATAPIG imports
from HoloLoom.datapig.detector import DataPigDetector
from HoloLoom.datapig.dashboard import QualityReport, render_quality_dashboard


def main():
    print("DATAPIG Visual Dashboard Demo")
    print("=" * 60)

    # Sample datasets with various quality issues
    datasets = [
        {
            "name": "users.csv",
            "data": [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "ssn": "123-45-6789"},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "ssn": "987-65-4321"},
                {"id": 1, "name": "Alice", "email": "alice@example.com", "ssn": "123-45-6789"},  # Duplicate
                {"id": 3, "name": "Charlie", "ssn": "555-12-3456"},  # Missing email (schema drift)
                {"id": 4, "name": "Diana", "email": "diana@example.com", "ssn": "111-22-3333"},
            ]
        },
        {
            "name": "orders.csv",
            "data": [
                {"order_id": 1, "user_id": 1, "amount": 100.50, "status": "shipped"},
                {"order_id": 2, "user_id": 2, "amount": 250.00, "status": "pending"},
                {"order_id": 3, "user_id": 999, "amount": 75.25, "status": "shipped"},  # Missing relation (user 999 doesn't exist)
                {"order_id": 4, "user_id": 1, "amount": 999999.99, "status": "pending"},  # Outlier amount
                {"order_id": 5, "user_id": 2, "amount": 50.00, "status": "SHIPPED"},  # Inconsistent format (capitalization)
            ]
        },
        {
            "name": "products.csv",
            "data": [
                {"product_id": 1, "name": "Widget A", "price": 19.99, "stock": 100},
                {"product_id": 2, "name": "Widget B", "price": 29.99, "stock": 50},
                {"product_id": 3, "name": "Widget A", "price": 19.99, "stock": 100},  # Fuzzy duplicate
                {"product_id": 4, "name": "Gadget", "price": 0.01, "stock": 1},  # Outlier price
            ]
        },
        {
            "name": "clean_data.csv",
            "data": [
                {"id": 1, "value": "A"},
                {"id": 2, "value": "B"},
                {"id": 3, "value": "C"},
                {"id": 4, "value": "D"},
                {"id": 5, "value": "E"},
            ]
        }
    ]

    # Analyze all datasets
    print("\nAnalyzing datasets...")
    reports = []

    for dataset_info in datasets:
        print(f"\n📊 {dataset_info['name']}")
        print(f"   Rows: {len(dataset_info['data'])}")

        detector = DataPigDetector()
        start_time = time.time()
        issues = detector.analyze_dataset(dataset_info["data"])
        duration_ms = (time.time() - start_time) * 1000

        print(f"   Issues found: {len(issues)}")
        print(f"   Duration: {duration_ms:.1f}ms")

        # Create quality report
        report = QualityReport(
            dataset_name=dataset_info["name"],
            timestamp=time.time(),
            issues=issues,
            dataset_size=len(dataset_info["data"]),
            duration_ms=duration_ms
        )

        # Show quality score
        score = report.get_quality_score()
        print(f"   Quality Score: {score * 100:.1f}%")

        # Show critical issues
        critical = report.get_critical_issues()
        if critical:
            print(f"   🚨 Critical Issues: {len(critical)}")
            for issue in critical[:2]:  # Show first 2
                print(f"      - {issue.message}")

        reports.append(report)

    # Generate dashboard
    print("\n" + "=" * 60)
    print("Generating Tufte-style dashboard...")

    html = render_quality_dashboard(
        reports,
        title="DATAPIG Quality Dashboard",
        subtitle="Production Data Analysis - Demo"
    )

    # Save to file
    output_dir = Path("demos/output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "datapig_dashboard.html"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Dashboard saved to: {output_path}")
    print(f"   File size: {len(html)} bytes")

    # Show summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print(f"   Total Datasets: {len(reports)}")
    print(f"   Total Issues: {sum(len(r.issues) for r in reports)}")
    print(f"   Average Quality: {sum(r.get_quality_score() for r in reports) / len(reports) * 100:.1f}%")

    # Show breakdown
    print("\nDataset Quality Breakdown:")
    for report in reports:
        score = report.get_quality_score()
        if score >= 0.9:
            status = "✅ EXCELLENT"
        elif score >= 0.75:
            status = "🟢 GOOD"
        elif score >= 0.5:
            status = "🟡 FAIR"
        else:
            status = "🔴 POOR"

        print(f"   {report.dataset_name:20s} {status:15s} ({score * 100:5.1f}%)")

    print("\n" + "=" * 60)
    print("Demo complete! Open the HTML file in your browser.")
    print(f"\nFile path: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
