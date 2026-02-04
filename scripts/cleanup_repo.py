#!/usr/bin/env python3
"""
Repository Cleanup Script

Automated cleanup of temporary files, old test results, and build artifacts.
Safe to run regularly during development.
"""

import os
import glob
import shutil
import argparse
from typing import List
from pathlib import Path


def cleanup_python_cache():
    """Remove Python cache files and directories"""
    print("🧹 Cleaning Python cache files...")

    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            shutil.rmtree(cache_dir)
            print(f"   Removed: {cache_dir}")

    # Remove .pyc files
    pyc_files = glob.glob('**/*.pyc', recursive=True)
    for file in pyc_files:
        os.remove(file)
        print(f"   Removed: {file}")

    print(f"   ✅ Python cache cleanup complete")


def cleanup_temp_reports():
    """Clean up temporary report files that may be left in root directory"""
    print(f"🧪 Cleaning temporary report files...")

    temp_patterns = [
        'integration_test_report_*.json',
        'pipeline_report_*.json',
    ]

    removed_count = 0
    for pattern in temp_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            os.remove(match)
            print(f"   Removed: {match}")
            removed_count += 1

    print(f"   ✅ Temporary reports cleanup complete ({removed_count} items)")


def cleanup_build_artifacts():
    """Remove build artifacts and temporary files"""
    print("🔨 Cleaning build artifacts...")

    artifacts = [
        'build/',
        'dist/',
        '*.egg-info/',
        '.pytest_cache/',
        'benchmarks_output/',
        'demo_outputs/',
        'profile_results/',
        'tools/',  # Old tools directory (consolidated into scripts/)
    ]

    removed_count = 0
    for pattern in artifacts:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            if os.path.isdir(match):
                shutil.rmtree(match)
                print(f"   Removed directory: {match}")
            else:
                os.remove(match)
                print(f"   Removed file: {match}")
            removed_count += 1

    print(f"   ✅ Build artifacts cleanup complete ({removed_count} items)")


def cleanup_root_temp_files():
    """Remove temporary files that ended up in root directory"""
    print("📁 Cleaning root temporary files...")

    temp_patterns = [
        '*.benchmark',
        '*.profiling',
        'temp_*',
    ]

    removed_count = 0
    for pattern in temp_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            os.remove(match)
            print(f"   Removed: {match}")
            removed_count += 1

    print(f"   ✅ Root cleanup complete ({removed_count} items)")


def cleanup_gpu_cache():
    """Clean GPU-related cache directories"""
    print("🎮 Cleaning GPU cache...")

    gpu_cache_dirs = [
        os.path.expanduser('~/.triton/cache'),
        os.path.expanduser('~/.cache/triton'),
        '.torch_compile_cache/',
    ]

    for cache_dir in gpu_cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"   Removed: {cache_dir}")
            except PermissionError:
                print(f"   ⚠️ Permission denied: {cache_dir}")

    print(f"   ✅ GPU cache cleanup complete")


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description='Clean up repository artifacts and temporary files')
    parser.add_argument('--skip-gpu-cache', action='store_true',
                       help='Skip GPU cache cleanup')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without actually doing it')

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN - showing what would be cleaned:")
        print("   (Use without --dry-run to actually perform cleanup)")
        print()

    print("🧹 Repository Cleanup Script")
    print("=" * 40)

    # Change to repository root
    script_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(script_dir)
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")
    print()

    if not args.dry_run:
        # Run cleanup functions
        cleanup_python_cache()
        cleanup_temp_reports()
        cleanup_build_artifacts()
        cleanup_root_temp_files()

        if not args.skip_gpu_cache:
            cleanup_gpu_cache()
    else:
        # Dry run - just list what would be cleaned
        print("Would clean:")
        print("  - Python cache files (__pycache__, *.pyc)")
        print("  - Temporary report files")
        print("  - Build artifacts (build/, dist/, *.egg-info/)")
        print("  - Root temporary files")
        if not args.skip_gpu_cache:
            print("  - GPU cache directories")

    print()
    print("✨ Repository cleanup complete!")
    print()
    print("💡 Tips:")
    print("  - Run this script regularly during development")
    print("  - Use --dry-run to see what would be cleaned")
    print("  - GPU cache cleanup can be skipped with --skip-gpu-cache")


if __name__ == "__main__":
    main()