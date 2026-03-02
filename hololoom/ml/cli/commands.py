"""
HoloLoom ML Training System - CLI Commands

Command-line interface for training, evaluating, and managing ML models.

Usage:
    hololoom-ml train <data> -t <target> [-m <model_type>]
    hololoom-ml evaluate <model_id>
    hololoom-ml predict <model_id> <data> [-o <output>]
    hololoom-ml list [--model-type <type>] [--json]
    hololoom-ml describe <model_id>
    hololoom-ml delete <model_id>
    hololoom-ml validate <data> [-t <target>]

Created: 2025-12-31
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for ML CLI."""
    parser = argparse.ArgumentParser(
        prog="hololoom-ml",
        description="HoloLoom ML Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    hololoom-ml train sales.csv -t revenue
    hololoom-ml train data.xlsx -t price -m ridge

  Evaluate a model:
    hololoom-ml evaluate linreg_001

  Make predictions:
    hololoom-ml predict linreg_001 new_data.csv
    hololoom-ml predict linreg_001 new_data.csv -o predictions.csv

  List models:
    hololoom-ml list
    hololoom-ml list --model-type linear_regression --json

  Describe a model:
    hololoom-ml describe linreg_001

  Validate data:
    hololoom-ml validate data.csv -t target_column
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # TRAIN command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("data", help="Path to training data (CSV, Excel, JSON, Parquet)")
    train_parser.add_argument("-t", "--target", required=True, help="Target column to predict")
    train_parser.add_argument(
        "-m",
        "--model-type",
        default="linear_regression",
        choices=["linear_regression", "ridge", "lasso"],
        help="Model type (default: linear_regression)",
    )
    train_parser.add_argument(
        "-f",
        "--features",
        nargs="+",
        help="Feature columns (default: all except target)",
    )
    train_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # EVALUATE command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("model_id", help="Model ID to evaluate")
    eval_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # PREDICT command
    pred_parser = subparsers.add_parser("predict", help="Make predictions")
    pred_parser.add_argument("model_id", help="Model ID to use")
    pred_parser.add_argument("data", help="Path to prediction data")
    pred_parser.add_argument("-o", "--output", help="Output file for predictions")
    pred_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # LIST command
    list_parser = subparsers.add_parser("list", help="List trained models")
    list_parser.add_argument(
        "--model-type",
        choices=["linear_regression", "ridge", "lasso"],
        help="Filter by model type",
    )
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--min-r2", type=float, help="Minimum R² score")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum results")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # DESCRIBE command
    desc_parser = subparsers.add_parser("describe", help="Describe a model")
    desc_parser.add_argument("model_id", help="Model ID to describe")
    desc_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # DELETE command
    del_parser = subparsers.add_parser("delete", help="Delete (archive) a model")
    del_parser.add_argument("model_id", help="Model ID to delete")
    del_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # VALIDATE command
    val_parser = subparsers.add_parser("validate", help="Validate data quality")
    val_parser.add_argument("data", help="Path to data file")
    val_parser.add_argument("-t", "--target", help="Target column (optional)")
    val_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # COMPARE command
    cmp_parser = subparsers.add_parser("compare", help="Compare models")
    cmp_parser.add_argument("model_ids", nargs="+", help="Model IDs to compare")
    cmp_parser.add_argument("--json", action="store_true", help="Output as JSON")

    return parser


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_metric(name: str, value, precision: int = 4) -> None:
    """Print a formatted metric."""
    if value is None:
        print(f"  {name}: N/A")
    elif isinstance(value, float):
        print(f"  {name}: {value:.{precision}f}")
    else:
        print(f"  {name}: {value}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"\n✓ {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"\n✗ Error: {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"  ⚠ {message}")


async def cmd_train(args: argparse.Namespace) -> int:
    """Execute train command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    skill = MLTrainerSkill()

    params = {
        "data_source": args.data,
        "target_column": args.target,
        "model_type": args.model_type,
    }
    if args.features:
        params["feature_columns"] = args.features

    print(f"Training {args.model_type} model on {args.data}...")
    print(f"Target column: {args.target}")

    result = await skill.execute(SkillInput(
        operation=MLOperation.TRAIN,
        parameters=params,
    ))

    if args.json:
        output = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if result.success else 1

    if result.success:
        print_success(result.message)
        print_header("Training Results")

        data = result.data
        print(f"  Model ID: {data.get('model_id')}")
        print(f"  Model Type: {data.get('model_type')}")
        print(f"  Training Time: {data.get('training_duration_ms', 0):.1f} ms")
        print(f"  Features: {data.get('n_features')}")
        print(f"  Samples: {data.get('n_samples')}")

        print("\n  Metrics:")
        metrics = data.get("metrics", {})
        print_metric("    R² Score", metrics.get("r2_score"))
        print_metric("    MSE", metrics.get("mse"))
        print_metric("    RMSE", metrics.get("rmse"))
        print_metric("    MAE", metrics.get("mae"))

        if data.get("feature_importance"):
            print("\n  Feature Importance:")
            for name, importance in list(data["feature_importance"].items())[:10]:
                print(f"    {name}: {importance:.4f}")

        if result.suggestions:
            print("\n  Next Steps:")
            for suggestion in result.suggestions:
                print(f"    → {suggestion}")

        return 0
    else:
        print_error(result.message)

        if result.data.get("issues"):
            print("\n  Issues Found:")
            for issue in result.data["issues"]:
                print(f"    - {issue.get('type')}: {issue.get('message')}")

        if result.suggestions:
            print("\n  Suggestions:")
            for suggestion in result.suggestions:
                print(f"    → {suggestion}")

        return 1


async def cmd_evaluate(args: argparse.Namespace) -> int:
    """Execute evaluate command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    skill = MLTrainerSkill()

    result = await skill.execute(SkillInput(
        operation=MLOperation.EVALUATE,
        parameters={"model_id": args.model_id},
    ))

    if args.json:
        output = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if result.success else 1

    if result.success:
        print_header(f"Evaluation: {args.model_id}")

        data = result.data
        print_metric("R² Score", data.get("r2_score"))
        print_metric("Adjusted R²", data.get("adjusted_r2"))
        print_metric("MSE", data.get("mse"))
        print_metric("RMSE", data.get("rmse"))
        print_metric("MAE", data.get("mae"))

        if data.get("cv_mean") is not None:
            print(f"\n  Cross-Validation:")
            print_metric("    CV Mean", data.get("cv_mean"))
            print_metric("    CV Std", data.get("cv_std"))

        if data.get("residuals_stats"):
            print(f"\n  Residuals:")
            stats = data["residuals_stats"]
            print_metric("    Mean", stats.get("mean"))
            print_metric("    Std", stats.get("std"))

        return 0
    else:
        print_error(result.message)
        return 1


async def cmd_predict(args: argparse.Namespace) -> int:
    """Execute predict command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    skill = MLTrainerSkill()

    params = {
        "model_id": args.model_id,
        "data_source": args.data,
    }
    if args.output:
        params["output_path"] = args.output

    print(f"Making predictions with {args.model_id} on {args.data}...")

    result = await skill.execute(SkillInput(
        operation=MLOperation.PREDICT,
        parameters=params,
    ))

    if args.json:
        output = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if result.success else 1

    if result.success:
        print_success(result.message)

        data = result.data
        print(f"\n  Predictions: {data.get('n_predictions')}")

        stats = data.get("prediction_stats", {})
        print(f"\n  Statistics:")
        print_metric("    Mean", stats.get("mean"))
        print_metric("    Std", stats.get("std"))
        print_metric("    Min", stats.get("min"))
        print_metric("    Max", stats.get("max"))

        if data.get("output_path"):
            print(f"\n  Saved to: {data['output_path']}")

        sample = data.get("predictions_sample", [])
        if sample:
            print(f"\n  Sample (first {len(sample)}):")
            for i, pred in enumerate(sample):
                print(f"    [{i}] {pred:.4f}")

        return 0
    else:
        print_error(result.message)
        return 1


async def cmd_list(args: argparse.Namespace) -> int:
    """Execute list command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    skill = MLTrainerSkill()

    params = {"limit": args.limit}
    if args.model_type:
        params["model_type"] = args.model_type
    if args.status:
        params["status"] = args.status
    if args.min_r2:
        params["min_r2"] = args.min_r2

    result = await skill.execute(SkillInput(
        operation=MLOperation.LIST_MODELS,
        parameters=params,
    ))

    if args.json:
        output = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if result.success else 1

    if result.success:
        models = result.data.get("models", [])
        total = result.data.get("total", 0)

        if not models:
            print("\nNo models found.")
            if result.suggestions:
                print("\nSuggestion:", result.suggestions[0])
            return 0

        print_header(f"Models ({total} found)")

        # Table header
        print(f"  {'ID':<20} {'Type':<18} {'Status':<10} {'R²':<10}")
        print(f"  {'-'*20} {'-'*18} {'-'*10} {'-'*10}")

        for model in models:
            model_id = model.get("model_id", "")[:20]
            model_type = model.get("model_type", "")[:18]
            status = model.get("status", "")[:10]
            r2 = model.get("r2_score")
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"

            print(f"  {model_id:<20} {model_type:<18} {status:<10} {r2_str:<10}")

        return 0
    else:
        print_error(result.message)
        return 1


async def cmd_describe(args: argparse.Namespace) -> int:
    """Execute describe command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    skill = MLTrainerSkill()

    result = await skill.execute(SkillInput(
        operation=MLOperation.DESCRIBE_MODEL,
        parameters={"model_id": args.model_id},
    ))

    if args.json:
        output = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if result.success else 1

    if result.success:
        data = result.data
        print_header(f"Model: {args.model_id}")

        print(f"  Type: {data.get('model_type')}")
        print(f"  Status: {data.get('status')}")
        print(f"  Created: {data.get('timestamp')}")
        print(f"  Training Time: {data.get('training_duration_ms', 0):.1f} ms")
        print(f"  Data Hash: {data.get('data_hash', 'N/A')}")

        if data.get("hyperparameters"):
            print(f"\n  Hyperparameters:")
            for key, value in data["hyperparameters"].items():
                print(f"    {key}: {value}")

        if data.get("feature_names"):
            print(f"\n  Features ({len(data['feature_names'])}):")
            for name in data["feature_names"][:10]:
                print(f"    - {name}")
            if len(data["feature_names"]) > 10:
                print(f"    ... and {len(data['feature_names']) - 10} more")

        if data.get("evaluation"):
            print(f"\n  Evaluation:")
            eval_data = data["evaluation"]
            print_metric("    R² Score", eval_data.get("r2_score"))
            print_metric("    RMSE", eval_data.get("rmse"))
            print_metric("    MAE", eval_data.get("mae"))

        return 0
    else:
        print_error(result.message)
        return 1


async def cmd_delete(args: argparse.Namespace) -> int:
    """Execute delete command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    if not args.force:
        confirm = input(f"Delete model {args.model_id}? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    skill = MLTrainerSkill()

    result = await skill.execute(SkillInput(
        operation=MLOperation.DELETE_MODEL,
        parameters={"model_id": args.model_id},
    ))

    if result.success:
        print_success(f"Model {args.model_id} archived")
        return 0
    else:
        print_error(result.message)
        return 1


async def cmd_validate(args: argparse.Namespace) -> int:
    """Execute validate command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    skill = MLTrainerSkill()

    params = {"data_source": args.data}
    if args.target:
        params["target_column"] = args.target

    print(f"Validating data in {args.data}...")

    result = await skill.execute(SkillInput(
        operation=MLOperation.VALIDATE_DATA,
        parameters=params,
    ))

    if args.json:
        output = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if result.success else 1

    if result.success:
        data = result.data
        is_valid = data.get("is_valid")

        if is_valid:
            print_success("Data validation PASSED")
        else:
            print_error("Data validation FAILED")

        print(f"\n  Samples: {data.get('n_samples')}")
        print(f"  Features: {data.get('n_features')}")

        issues = data.get("issues", [])
        if issues:
            print(f"\n  Issues ({len(issues)}):")
            for issue in issues:
                severity = issue.get("severity", "warning")
                icon = "✗" if severity == "error" else "⚠"
                blocking = " [BLOCKING]" if issue.get("blocking") else ""
                col = f" ({issue['column']})" if issue.get("column") else ""
                print(f"    {icon} {issue['type']}{col}: {issue['message']}{blocking}")

        recommendations = data.get("recommendations", [])
        if recommendations:
            print(f"\n  Recommendations:")
            for rec in recommendations:
                print(f"    → {rec}")

        return 0 if is_valid else 1
    else:
        print_error(result.message)
        return 1


async def cmd_compare(args: argparse.Namespace) -> int:
    """Execute compare command."""
    from hololoom.ml.skills.ml_trainer_skill import MLTrainerSkill, SkillInput, MLOperation

    skill = MLTrainerSkill()

    result = await skill.execute(SkillInput(
        operation=MLOperation.COMPARE_MODELS,
        parameters={"model_ids": args.model_ids},
    ))

    if args.json:
        output = {
            "success": result.success,
            "message": result.message,
            "data": result.data,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if result.success else 1

    if result.success:
        data = result.data
        comparisons = data.get("comparisons", [])

        print_header("Model Comparison")

        # Table header
        print(f"  {'Model':<20} {'Type':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
        print(f"  {'-'*20} {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

        for comp in comparisons:
            if comp.get("error"):
                print(f"  {comp['model_id']:<20} ERROR: {comp['error']}")
                continue

            model_id = comp.get("model_id", "")[:20]
            model_type = comp.get("model_type", "")[:15]
            r2 = comp.get("r2_score")
            rmse = comp.get("rmse")
            mae = comp.get("mae")

            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
            mae_str = f"{mae:.4f}" if mae is not None else "N/A"

            # Mark best model
            best = data.get("best_model")
            marker = " ★" if model_id == best else ""

            print(f"  {model_id:<20} {model_type:<15} {r2_str:<10} {rmse_str:<10} {mae_str:<10}{marker}")

        if data.get("best_model"):
            print(f"\n  Best Model: {data['best_model']} (R² = {data.get('best_r2', 0):.4f})")

        return 0
    else:
        print_error(result.message)
        return 1


def cli() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Map commands to handlers
    handlers = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "list": cmd_list,
        "describe": cmd_describe,
        "delete": cmd_delete,
        "validate": cmd_validate,
        "compare": cmd_compare,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print_error(f"Unknown command: {args.command}")
        return 1

    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as e:
        print_error(str(e))
        return 1


def main() -> None:
    """Entry point for console script."""
    sys.exit(cli())


if __name__ == "__main__":
    main()
