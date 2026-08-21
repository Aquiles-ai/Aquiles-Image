from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from aquilesimage.models import ServerConfigs

BENCH_TOOL_VERSION = "0.1.0"

SIZE_MAP: Dict[str, Tuple[int, int]] = {
    "1024x1024": (1024, 1024),
    "1536x1024": (1536, 1024),
    "1024x1536": (1024, 1536),
    "256x256":   (256,  256),
    "512x512":   (512,  512),
    "1792x1024": (1792, 1024),
    "1024x1792": (1024, 1792),
    "2048x2048": (2048, 2048),
}


class UniformProfile(BaseModel):
    type: Literal["uniform"] = "uniform"
    size: str = "1024x1024"
    n: int = Field(default=1, ge=1)

    @field_validator("size")
    @classmethod
    def _valid_size(cls, v: str) -> str:
        if v not in SIZE_MAP:
            raise ValueError(f"size '{v}' is not a supported size. Allowed: {sorted(SIZE_MAP)}")
        return v


class MixedProfile(BaseModel):
    type: Literal["mixed"] = "mixed"
    sizes: Dict[str, float]
    n: Union[int, Tuple[int, int]] = Field(default=1)

    @field_validator("sizes")
    @classmethod
    def _valid_sizes(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("sizes distribution cannot be empty")
        for size, weight in v.items():
            if size not in SIZE_MAP:
                raise ValueError(f"size '{size}' is not a supported size. Allowed: {sorted(SIZE_MAP)}")
            if weight <= 0:
                raise ValueError(f"weight for size '{size}' must be > 0")
        return v

    @field_validator("n")
    @classmethod
    def _valid_n(cls, v):
        if isinstance(v, int):
            if v < 1:
                raise ValueError("n must be >= 1")
            return v
        low, high = v
        if low < 1 or high < low:
            raise ValueError(f"invalid n range ({low}, {high}): require 1 <= low <= high")
        return v


Profile = Annotated[Union[UniformProfile, MixedProfile], Field(discriminator="type")]


class DeviceInfo(BaseModel):
    id: str
    name: str
    vram_total_gb: float
    vram_free_gb: float


class TimingSummary(BaseModel):
    mean_ms: Optional[float] = None
    median_ms: Optional[float] = None
    std_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None


class MetricsBlock(BaseModel):
    successful: int = 0
    failed: int = 0
    rejected_429: int = 0
    duration_s: float = 0.0
    target_request_rate: Optional[float] = None
    achieved_request_rate: Optional[float] = None
    requests_per_s: Optional[float] = None
    images_per_s: Optional[float] = None
    e2el_ms: TimingSummary = Field(default_factory=TimingSummary)
    warnings: List[str] = Field(default_factory=list)


class RequestRecord(BaseModel):
    request_id: str
    size: Optional[str] = None
    n_images: int = 1
    status_code: Optional[int] = None
    e2el_ms: Optional[float] = None
    error: Optional[str] = None
    submitted_at: float = 0.0
    completed_at: float = 0.0


class BenchReport(BaseModel):
    bench: Dict[str, Any]
    server: Dict[str, Any] = Field(default_factory=dict)
    load: Dict[str, Any] = Field(default_factory=dict)
    metrics: MetricsBlock = Field(default_factory=MetricsBlock)
    requests: List[RequestRecord] = Field(default_factory=list)


class BenchConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5500
    api_key: Optional[str] = None

    num_prompts: int = Field(default=100, ge=1)
    request_rate: Optional[float] = Field(
        default=None,
        description="Requests per second. None means unlimited (burst).",
    )
    max_concurrency: Optional[int] = Field(default=None, ge=1)

    profile: Profile = Field(default_factory=UniformProfile)
    warmup: int = Field(default=3, ge=0)
    seed: Optional[int] = None
    timeout_s: float = Field(default=600.0, gt=0)

    label: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)
    result_dir: str = "./bench_results"
    result_filename: Optional[str] = None
    save_detailed: bool = False

    @model_validator(mode="after")
    def _sanity(self) -> "BenchConfig":
        if self.request_rate is not None and self.request_rate <= 0:
            raise ValueError("request_rate must be > 0 or None for burst mode")
        if self.num_prompts < self.warmup:
            raise ValueError("warmup cannot exceed num_prompts")
        return self

    def collect_warnings(self) -> List[str]:
        warnings: List[str] = []
        if self.api_key is not None:
            warnings.append(
                "api_key is stored in plaintext inside this config file"
            )
        if isinstance(self.profile, MixedProfile):
            total = sum(self.profile.sizes.values())
            if abs(total - 1.0) > 0.01:
                warnings.append(
                    f"mixed profile weights sum to {total:.4f}, expected ~1.0; "
                    "they will be normalized"
                )
        if self.max_concurrency is not None and self.request_rate is not None:
            warnings.append(
                "request_rate and max_concurrency are both set; the effective "
                "rate may be lower than requested if the server cannot keep up"
            )
        return warnings

    def save_config(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=4)
        return target

    @classmethod
    def uniform(cls, size: str = "1024x1024", **kwargs) -> "BenchConfig":
        return cls(profile=UniformProfile(size=size), **kwargs)

    @classmethod
    def mixed(cls, sizes: Dict[str, float], **kwargs) -> "BenchConfig":
        return cls(profile=MixedProfile(sizes=sizes), **kwargs)
