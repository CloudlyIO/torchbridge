"""
Benchmark Database for KernelPyTorch Cloud Testing.

This module provides a lightweight database for storing and querying
benchmark results across cloud platforms.

Features:
- Store benchmark records with metadata
- Query by platform, hardware, date range
- Compare results across platforms
- Detect performance regressions

Version: 0.3.7
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkRecord:
    """A single benchmark result record."""
    # Identification
    id: Optional[int] = None
    run_id: str = ""  # Unique run identifier
    timestamp: datetime = field(default_factory=datetime.now)

    # Platform info
    cloud_provider: str = ""  # "aws" or "gcp"
    instance_type: str = ""   # e.g., "p4d.24xlarge", "a2-highgpu-1g"
    region: str = ""          # e.g., "us-west-2", "us-central1-a"
    hardware_type: str = ""   # "nvidia", "amd", "tpu"
    gpu_model: str = ""       # e.g., "H100", "A100", "MI300", "TPU v5e"

    # Benchmark info
    benchmark_name: str = ""
    benchmark_suite: str = ""  # e.g., "nvidia_backend", "tpu_backend"

    # Results
    latency_ms: float = 0.0
    throughput: float = 0.0
    memory_mb: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0

    # Cost
    duration_seconds: float = 0.0
    cost_usd: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "cloud_provider": self.cloud_provider,
            "instance_type": self.instance_type,
            "region": self.region,
            "hardware_type": self.hardware_type,
            "gpu_model": self.gpu_model,
            "benchmark_name": self.benchmark_name,
            "benchmark_suite": self.benchmark_suite,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "memory_mb": self.memory_mb,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkRecord":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            id=data.get("id"),
            run_id=data.get("run_id", ""),
            timestamp=timestamp,
            cloud_provider=data.get("cloud_provider", ""),
            instance_type=data.get("instance_type", ""),
            region=data.get("region", ""),
            hardware_type=data.get("hardware_type", ""),
            gpu_model=data.get("gpu_model", ""),
            benchmark_name=data.get("benchmark_name", ""),
            benchmark_suite=data.get("benchmark_suite", ""),
            latency_ms=data.get("latency_ms", 0.0),
            throughput=data.get("throughput", 0.0),
            memory_mb=data.get("memory_mb", 0.0),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            tests_skipped=data.get("tests_skipped", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ComparisonResult:
    """Result of comparing benchmarks across platforms."""
    benchmark_name: str
    platform_a: str
    platform_b: str
    latency_ratio: float  # platform_b / platform_a
    throughput_ratio: float
    memory_ratio: float
    cost_ratio: float
    records_a: int
    records_b: int

    def __str__(self) -> str:
        return (
            f"{self.benchmark_name}: {self.platform_a} vs {self.platform_b}\n"
            f"  Latency ratio: {self.latency_ratio:.2f}x\n"
            f"  Throughput ratio: {self.throughput_ratio:.2f}x\n"
            f"  Memory ratio: {self.memory_ratio:.2f}x\n"
            f"  Cost ratio: {self.cost_ratio:.2f}x"
        )


# ============================================================================
# Benchmark Database
# ============================================================================

class BenchmarkDatabase:
    """
    SQLite-based benchmark database for cloud testing results.

    Example:
        >>> db = BenchmarkDatabase("benchmarks.db")
        >>> record = BenchmarkRecord(
        ...     run_id="test-123",
        ...     cloud_provider="aws",
        ...     instance_type="p4d.24xlarge",
        ...     benchmark_name="flash_attention",
        ...     latency_ms=1.5,
        ... )
        >>> db.insert(record)
        >>> results = db.query(cloud_provider="aws")
    """

    def __init__(self, db_path: str = "benchmarks.db"):
        """
        Initialize benchmark database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                cloud_provider TEXT,
                instance_type TEXT,
                region TEXT,
                hardware_type TEXT,
                gpu_model TEXT,
                benchmark_name TEXT,
                benchmark_suite TEXT,
                latency_ms REAL,
                throughput REAL,
                memory_mb REAL,
                tests_passed INTEGER,
                tests_failed INTEGER,
                tests_skipped INTEGER,
                duration_seconds REAL,
                cost_usd REAL,
                metadata TEXT
            )
        """)

        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cloud_provider ON benchmarks(cloud_provider)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hardware_type ON benchmarks(hardware_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmark_name ON benchmarks(benchmark_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmarks(timestamp)
        """)

        conn.commit()
        conn.close()

    def insert(self, record: BenchmarkRecord) -> int:
        """
        Insert a benchmark record.

        Args:
            record: BenchmarkRecord to insert

        Returns:
            Inserted record ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO benchmarks (
                run_id, timestamp, cloud_provider, instance_type, region,
                hardware_type, gpu_model, benchmark_name, benchmark_suite,
                latency_ms, throughput, memory_mb, tests_passed, tests_failed,
                tests_skipped, duration_seconds, cost_usd, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.run_id,
            record.timestamp.isoformat() if record.timestamp else datetime.now().isoformat(),
            record.cloud_provider,
            record.instance_type,
            record.region,
            record.hardware_type,
            record.gpu_model,
            record.benchmark_name,
            record.benchmark_suite,
            record.latency_ms,
            record.throughput,
            record.memory_mb,
            record.tests_passed,
            record.tests_failed,
            record.tests_skipped,
            record.duration_seconds,
            record.cost_usd,
            json.dumps(record.metadata),
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Inserted benchmark record {record_id}")
        return record_id

    def insert_many(self, records: List[BenchmarkRecord]) -> List[int]:
        """Insert multiple benchmark records."""
        return [self.insert(r) for r in records]

    def query(
        self,
        cloud_provider: Optional[str] = None,
        hardware_type: Optional[str] = None,
        benchmark_name: Optional[str] = None,
        benchmark_suite: Optional[str] = None,
        instance_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[BenchmarkRecord]:
        """
        Query benchmark records.

        Args:
            cloud_provider: Filter by cloud provider ("aws", "gcp")
            hardware_type: Filter by hardware ("nvidia", "amd", "tpu")
            benchmark_name: Filter by benchmark name
            benchmark_suite: Filter by benchmark suite
            instance_type: Filter by instance type
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum records to return

        Returns:
            List of matching BenchmarkRecords
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM benchmarks WHERE 1=1"
        params = []

        if cloud_provider:
            query += " AND cloud_provider = ?"
            params.append(cloud_provider)
        if hardware_type:
            query += " AND hardware_type = ?"
            params.append(hardware_type)
        if benchmark_name:
            query += " AND benchmark_name = ?"
            params.append(benchmark_name)
        if benchmark_suite:
            query += " AND benchmark_suite = ?"
            params.append(benchmark_suite)
        if instance_type:
            query += " AND instance_type = ?"
            params.append(instance_type)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert rows to records
        records = []
        for row in rows:
            records.append(BenchmarkRecord(
                id=row[0],
                run_id=row[1],
                timestamp=datetime.fromisoformat(row[2]) if row[2] else None,
                cloud_provider=row[3],
                instance_type=row[4],
                region=row[5],
                hardware_type=row[6],
                gpu_model=row[7],
                benchmark_name=row[8],
                benchmark_suite=row[9],
                latency_ms=row[10] or 0.0,
                throughput=row[11] or 0.0,
                memory_mb=row[12] or 0.0,
                tests_passed=row[13] or 0,
                tests_failed=row[14] or 0,
                tests_skipped=row[15] or 0,
                duration_seconds=row[16] or 0.0,
                cost_usd=row[17] or 0.0,
                metadata=json.loads(row[18]) if row[18] else {},
            ))

        return records

    def get_latest(
        self,
        benchmark_name: str,
        hardware_type: Optional[str] = None,
    ) -> Optional[BenchmarkRecord]:
        """Get the latest benchmark result."""
        results = self.query(
            benchmark_name=benchmark_name,
            hardware_type=hardware_type,
            limit=1,
        )
        return results[0] if results else None

    def get_statistics(
        self,
        benchmark_name: str,
        cloud_provider: Optional[str] = None,
        hardware_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get statistics for a benchmark.

        Returns:
            Dictionary with avg, min, max, stddev for latency/throughput
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT
                AVG(latency_ms) as avg_latency,
                MIN(latency_ms) as min_latency,
                MAX(latency_ms) as max_latency,
                AVG(throughput) as avg_throughput,
                COUNT(*) as count
            FROM benchmarks
            WHERE benchmark_name = ?
        """
        params = [benchmark_name]

        if cloud_provider:
            query += " AND cloud_provider = ?"
            params.append(cloud_provider)
        if hardware_type:
            query += " AND hardware_type = ?"
            params.append(hardware_type)

        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()

        return {
            "avg_latency_ms": row[0] or 0.0,
            "min_latency_ms": row[1] or 0.0,
            "max_latency_ms": row[2] or 0.0,
            "avg_throughput": row[3] or 0.0,
            "count": row[4] or 0,
        }


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_platforms(
    db: BenchmarkDatabase,
    benchmark_name: str,
    platform_a: str,
    platform_b: str,
) -> ComparisonResult:
    """
    Compare benchmark results between two platforms.

    Args:
        db: BenchmarkDatabase instance
        benchmark_name: Benchmark to compare
        platform_a: First platform (e.g., "aws", "gcp")
        platform_b: Second platform

    Returns:
        ComparisonResult with ratios
    """
    records_a = db.query(benchmark_name=benchmark_name, cloud_provider=platform_a)
    records_b = db.query(benchmark_name=benchmark_name, cloud_provider=platform_b)

    if not records_a or not records_b:
        return ComparisonResult(
            benchmark_name=benchmark_name,
            platform_a=platform_a,
            platform_b=platform_b,
            latency_ratio=0.0,
            throughput_ratio=0.0,
            memory_ratio=0.0,
            cost_ratio=0.0,
            records_a=len(records_a),
            records_b=len(records_b),
        )

    # Calculate averages
    avg_latency_a = sum(r.latency_ms for r in records_a) / len(records_a)
    avg_latency_b = sum(r.latency_ms for r in records_b) / len(records_b)
    avg_throughput_a = sum(r.throughput for r in records_a) / len(records_a)
    avg_throughput_b = sum(r.throughput for r in records_b) / len(records_b)
    avg_memory_a = sum(r.memory_mb for r in records_a) / len(records_a)
    avg_memory_b = sum(r.memory_mb for r in records_b) / len(records_b)
    avg_cost_a = sum(r.cost_usd for r in records_a) / len(records_a)
    avg_cost_b = sum(r.cost_usd for r in records_b) / len(records_b)

    return ComparisonResult(
        benchmark_name=benchmark_name,
        platform_a=platform_a,
        platform_b=platform_b,
        latency_ratio=avg_latency_b / avg_latency_a if avg_latency_a > 0 else 0.0,
        throughput_ratio=avg_throughput_b / avg_throughput_a if avg_throughput_a > 0 else 0.0,
        memory_ratio=avg_memory_b / avg_memory_a if avg_memory_a > 0 else 0.0,
        cost_ratio=avg_cost_b / avg_cost_a if avg_cost_a > 0 else 0.0,
        records_a=len(records_a),
        records_b=len(records_b),
    )


def query_benchmarks(
    db_path: str = "benchmarks.db",
    **kwargs,
) -> List[BenchmarkRecord]:
    """
    Convenience function to query benchmarks.

    Args:
        db_path: Path to database file
        **kwargs: Query parameters (passed to BenchmarkDatabase.query)

    Returns:
        List of matching BenchmarkRecords
    """
    db = BenchmarkDatabase(db_path)
    return db.query(**kwargs)
