from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from aquilesimage.cli import branding as ui
from aquilesimage.bench.models import (
    BENCH_TOOL_VERSION,
    BenchConfig,
    BenchReport,
    MetricsBlock,
    MixedProfile,
    RequestRecord,
    TimingSummary,
)

READY_CHECK_TIMEOUT_S = 60.0

PROMPT_SUBJECTS = [
    "a red fox", "an old lighthouse", "a samurai", "a vintage car",
    "a neon city street", "a snowy mountain", "a bowl of ramen",
    "an astronaut", "a tropical beach", "a steam train", "a glass of wine",
    "a medieval castle", "a jellyfish", "a sunflower field", "a robot barista",
]
PROMPT_STYLES = [
    "cinematic", "photorealistic", "watercolor", "cyberpunk", "minimalist",
    "baroque", "isometric", "noir", "surrealist", "documentary",
]


class BenchError(RuntimeError):
    pass


def _percentile(sorted_values: List[float], pct: float) -> float:
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def summarize_latencies(values_ms: List[float]) -> TimingSummary:
    if not values_ms:
        return TimingSummary()
    ordered = sorted(values_ms)
    return TimingSummary(
        mean_ms=statistics.fmean(ordered),
        median_ms=statistics.median(ordered),
        std_ms=statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        p99_ms=_percentile(ordered, 99),
    )


def _make_prompt(rng: random.Random) -> str:
    subject = rng.choice(PROMPT_SUBJECTS)
    style = rng.choice(PROMPT_STYLES)
    return f"a {style} image of {subject}, highly detailed, sharp focus"


def sample_spec(profile: Any, rng: random.Random, request_id: str) -> Dict[str, Any]:
    if isinstance(profile, MixedProfile):
        sizes = list(profile.sizes.keys())
        weights = list(profile.sizes.values())
        size = rng.choices(sizes, weights=weights, k=1)[0]
    else:
        size = profile.size
    n = profile.n if isinstance(profile.n, int) else rng.randint(profile.n[0], profile.n[1])
    return {"request_id": request_id, "size": size, "n": n, "prompt": _make_prompt(rng)}


def build_request_plan(config: BenchConfig, rng: random.Random) -> List[Dict[str, Any]]:
    return [sample_spec(config.profile, rng, f"bench-{i}") for i in range(config.num_prompts)]


def build_arrival_times(config: BenchConfig, rng: random.Random) -> List[float]:
    if config.request_rate is None:
        return [0.0] * config.num_prompts
    times: List[float] = []
    t = 0.0
    for _ in range(config.num_prompts):
        t += rng.expovariate(config.request_rate)
        times.append(t)
    return times


class BenchRunner:
    def __init__(self, config: BenchConfig, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.config = config
        self.transport = transport
        self.base_url = config.base_url
        self.headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}

    async def arun(self) -> BenchReport:
        warnings: List[str] = list(self.config.collect_warnings())
        rng = random.Random(self.config.seed)

        limits = self.config.max_concurrency or 1024
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=httpx.Timeout(self.config.timeout_s),
            limits=httpx.Limits(max_connections=limits, max_keepalive_connections=limits),
            transport=self.transport,
        ) as client:
            health = await self._ready_check(client)
            devices = health.get("devices") or []
            model_name = await self._resolve_model(client)

            configs_snapshot = await self._safe_get_json(client, "/v1/configs")
            stats_before = await self._safe_get_json(client, "/v1/stats")
            warnings.extend(self._dynamic_warnings(configs_snapshot))

            await self._warmup(client, model_name, rng, warnings)

            plan = build_request_plan(self.config, rng)
            arrivals = build_arrival_times(self.config, rng)

            rate_label = "burst" if self.config.request_rate is None else f"{self.config.request_rate}/s"
            ui.info(f"target={self.base_url} model={model_name} "
                    f"prompts={self.config.num_prompts} rate={rate_label} "
                    f"profile={self.config.profile.type}")

            records = await self._execute(client, model_name, plan, arrivals)

            stats_after = await self._safe_get_json(client, "/v1/stats")

        successful, rejected, failed = self._classify(records)
        start_wall = min(r.submitted_at for r in records)
        end_wall = max(r.completed_at for r in records)
        duration_s = end_wall - start_wall
        send_window_s = max(r.submitted_at for r in records) - start_wall
        total_images = sum(r.n_images for r in successful)
        latencies = [r.e2el_ms for r in successful if r.e2el_ms is not None]

        if rejected:
            warnings.append(
                f"{len(rejected)} requests were rejected with HTTP 429; "
                "the server's max_concurrent_infer is saturating at this load"
            )

        report = BenchReport(
            bench={
                "tool_version": BENCH_TOOL_VERSION,
                "timestamp": int(start_wall),
                "duration_s": round(duration_s, 3),
                "label": self.config.label,
                "metadata": self.config.metadata,
                "config": self.config.model_dump(mode="json"),
            },
            server={
                "configs": configs_snapshot,
                "devices": devices,
                "stats_before": stats_before,
                "stats_after": stats_after,
            },
            load={
                "num_prompts": self.config.num_prompts,
                "profile": self.config.profile.model_dump(mode="json"),
                "target_request_rate": self.config.request_rate,
                "max_concurrency": self.config.max_concurrency,
                "send_window_s": round(send_window_s, 3),
            },
            metrics=MetricsBlock(
                successful=len(successful),
                failed=len(failed),
                rejected_429=len(rejected),
                duration_s=round(duration_s, 3),
                target_request_rate=self.config.request_rate,
                achieved_request_rate=round(len(records) / duration_s, 3) if duration_s > 0 else None,
                requests_per_s=round(len(successful) / duration_s, 3) if duration_s > 0 else None,
                images_per_s=round(total_images / duration_s, 3) if duration_s > 0 else None,
                e2el_ms=summarize_latencies(latencies),
                warnings=warnings,
            ),
            requests=[r for r in records if r is not None] if self.config.save_detailed else [],
        )

        result_path = self.save_report(report)
        print_summary(report, result_path)
        return report

    def run(self) -> BenchReport:
        return asyncio.run(self.arun())

    def save_report(self, report: BenchReport) -> Path:
        result_dir = Path(self.config.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        rate_tag = "burst" if self.config.request_rate is None else f"{self.config.request_rate}rps"
        filename = self.config.result_filename or (
            f"{self.config.label or 'bench'}-{rate_tag}-{int(time.time())}.json"
        )
        path = result_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=4)
        return path

    def _dynamic_warnings(self, configs_snapshot: Optional[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        if configs_snapshot is None:
            return out
        max_infer = configs_snapshot.get("max_concurrent_infer")
        if (
            isinstance(max_infer, int)
            and self.config.max_concurrency is not None
            and self.config.max_concurrency > max_infer
        ):
            out.append(
                f"max_concurrency ({self.config.max_concurrency}) exceeds the server's "
                f"max_concurrent_infer ({max_infer}); expect HTTP 429 rejections"
            )
        mode = configs_snapshot.get("mode")
        if mode == "piecewise" and isinstance(self.config.profile, MixedProfile):
            out.append(
                "server runs in piecewise mode with a mixed size profile; shapes outside "
                "the precompiled matrix will trigger runtime compilation and skew p95/p99"
            )
        return out

    async def _ready_check(self, client: httpx.AsyncClient) -> Dict[str, Any]:
        deadline = time.time() + READY_CHECK_TIMEOUT_S
        last_error: Optional[str] = None
        while True:
            try:
                resp = await client.get("/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        return data
                    last_error = f"status={data.get('status')}"
            except httpx.HTTPError as e:
                last_error = str(e)
            if time.time() >= deadline:
                raise BenchError(
                    f"server at {self.base_url} did not become ready within "
                    f"{READY_CHECK_TIMEOUT_S:.0f}s (last: {last_error})"
                )
            await asyncio.sleep(1.0)

    async def _resolve_model(self, client: httpx.AsyncClient) -> str:
        data = await self._safe_get_json(client, "/v1/models")
        if data and data.get("data"):
            return data["data"][0]["id"]
        raise BenchError(f"could not resolve model name from {self.base_url}/v1/models")

    async def _safe_get_json(self, client: httpx.AsyncClient, path: str) -> Optional[Dict[str, Any]]:
        try:
            resp = await client.get(path)
            if resp.status_code == 200:
                return resp.json()
            return None
        except httpx.HTTPError:
            return None

    async def _warmup(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        rng: random.Random,
        warnings: List[str],
    ) -> None:
        for i in range(self.config.warmup):
            spec = sample_spec(self.config.profile, rng, f"warmup-{i}")
            try:
                resp = await client.post("/images/generations", json=self._payload(spec, model_name))
                if resp.status_code != 200:
                    warnings.append(f"warmup request {i} returned HTTP {resp.status_code}")
            except httpx.HTTPError as e:
                warnings.append(f"warmup request {i} failed: {e}")

    @staticmethod
    def _payload(spec: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        return {
            "model": model_name,
            "prompt": spec["prompt"],
            "n": spec["n"],
            "size": spec["size"],
        }

    async def _execute(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        plan: List[Dict[str, Any]],
        arrivals: List[float],
    ) -> List[Optional[RequestRecord]]:
        records: List[Optional[RequestRecord]] = [None] * len(plan)
        semaphore = asyncio.Semaphore(self.config.max_concurrency or len(plan))
        progress_step = max(1, len(plan) // 10)
        completed_count = 0

        loop = asyncio.get_running_loop()
        t0 = loop.time()

        progress = None
        if ui.RICH:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            progress = Progress(
                SpinnerColumn(style="magenta"),
                TextColumn("[bold magenta]bench[/bold magenta]"),
                BarColumn(bar_width=None, style="dim magenta", complete_style="magenta"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )

        async def fire(idx: int) -> None:
            nonlocal completed_count
            delay = arrivals[idx]
            if delay > 0:
                await asyncio.sleep(delay)
            spec = plan[idx]
            submitted_at = time.time()
            async with semaphore:
                req_t0 = time.perf_counter()
                status_code: Optional[int] = None
                error: Optional[str] = None
                try:
                    resp = await client.post("/images/generations", json=self._payload(spec, model_name))
                    status_code = resp.status_code
                    if status_code != 200:
                        error = f"HTTP {status_code}"
                except httpx.HTTPError as e:
                    error = str(e)
                latency_ms = (time.perf_counter() - req_t0) * 1000.0

            records[idx] = RequestRecord(
                request_id=spec["request_id"],
                size=spec["size"],
                n_images=spec["n"],
                status_code=status_code,
                e2el_ms=latency_ms if status_code == 200 else None,
                error=error,
                submitted_at=submitted_at,
                completed_at=time.time(),
            )
            completed_count += 1
            if progress is not None:
                progress.update(task_id, completed=completed_count)
            elif completed_count % progress_step == 0 or completed_count == len(plan):
                ui.info(f"{completed_count}/{len(plan)} completed")

        if progress is not None:
            with progress:
                task_id = progress.add_task("run", total=len(plan))
                await asyncio.gather(*(asyncio.create_task(fire(i)) for i in range(len(plan))))
        else:
            await asyncio.gather(*(asyncio.create_task(fire(i)) for i in range(len(plan))))
        return records

    @staticmethod
    def _classify(records: List[Optional[RequestRecord]]) -> Tuple[
        List[RequestRecord], List[RequestRecord], List[RequestRecord]
    ]:
        successful: List[RequestRecord] = []
        rejected: List[RequestRecord] = []
        failed: List[RequestRecord] = []
        for r in records:
            if r is None:
                continue
            if r.status_code == 200 and r.error is None:
                successful.append(r)
            elif r.status_code == 429:
                rejected.append(r)
            else:
                failed.append(r)
        return successful, rejected, failed


def print_summary(report: BenchReport, result_path: Path) -> None:
    m = report.metrics
    configs = report.server.get("configs") or {}

    if not ui.RICH:
        print("============ Aquiles-Image Bench Result ============")
        print(f"Label:                   {report.bench.get('label')}")
        print(f"Successful requests:     {m.successful}")
        print(f"Failed requests:         {m.failed}")
        print(f"Rejected (429):          {m.rejected_429}")
        print(f"Benchmark duration (s):  {m.duration_s}")
        print(f"Request throughput:      {m.requests_per_s} req/s")
        print(f"Output image throughput: {m.images_per_s} img/s")
        e = m.e2el_ms
        if e.mean_ms is not None:
            print(f"Mean/Median E2EL (ms):   {e.mean_ms:.2f} / {e.median_ms:.2f}")
            print(f"P50/P95/P99 E2EL (ms):   {e.p50_ms:.2f} / {e.p95_ms:.2f} / {e.p99_ms:.2f}")
        for w in m.warnings:
            print(f"warning: {w}")
        print(f"Result saved to:         {result_path}")
        return

    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = ui.get_console()

    head = Table(show_header=False, box=None, pad_edge=False)
    head.add_column()
    head.add_column(justify="right")
    subtitle = f"label: {report.bench.get('label')}"
    if configs:
        subtitle += f" · mode={configs.get('mode')} · max_batch_size={configs.get('max_batch_size')}"
    head.add_row(
        Text("aquiles-image bench", style="bold magenta"),
        Text(subtitle, style="grey62"),
    )
    console.print(Panel(head, box=box.HEAVY, border_style="magenta"))

    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column("metric", style="grey62")
    t.add_column("value", justify="right")
    t.add_row("requests",
              f"[green]{m.successful}[/green] ok · [red]{m.failed}[/red] fail · "
              f"[yellow]{m.rejected_429}[/yellow] 429")
    t.add_row("duration", f"{m.duration_s:.1f}s")
    t.add_row("throughput",
              f"[bold]{m.images_per_s}[/bold] img/s ({m.requests_per_s} req/s)")
    e = m.e2el_ms
    if e.mean_ms is not None:
        grad = ["bright_green", "green", "yellow", "bright_red", "red"]
        for i, name in enumerate(["mean", "median", "p50", "p95", "p99"]):
            value = getattr(e, f"{name}_ms")
            t.add_row(f"e2el {name}", Text(f"{value:>8.1f} ms", style=grad[i]))
    console.print(t)

    for w in m.warnings:
        ui.warn(w)
    ui.info(f"Result saved to {result_path}")


def run_bench(config: BenchConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> BenchReport:
    return BenchRunner(config, transport=transport).run()


async def arun_bench(config: BenchConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> BenchReport:
    return await BenchRunner(config, transport=transport).arun()
