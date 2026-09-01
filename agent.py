import base64
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlparse

import psutil
import requests
import websocket


DEFAULT_CONTROL_PLANE_URL = (
    "http://private-ai-prod-alb-2007774676.ap-south-1.elb.amazonaws.com"
)
VLLM_CHAT_COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
CREDENTIALS_PATH = os.environ.get(
    "AI_NODE_CREDENTIALS_PATH",
    "/opt/ai-node/credentials.json"
)
HF_CACHE_HOST = os.environ.get("HF_CACHE_DIR", "/opt/ai-node/hf-cache")
N8N_CONTAINER_PORT = 5678
N8N_STATE_DIR = os.environ.get("N8N_AGENT_STATE_DIR", "/opt/ai-node/n8n-state")


def load_credentials(path: str = CREDENTIALS_PATH) -> dict:
    if not os.path.isfile(path):
        print(
            f"Missing credentials file: {path}\n"
            "Re-run the installer with an enrollment key so permanent "
            "machine credentials are written before starting the agent."
        )
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Failed to read credentials from {path}: {error}")
        sys.exit(1)

    machine_id = data.get("machine_id")
    agent_token = data.get("agent_token")

    if not machine_id or not agent_token:
        print(
            f"Credentials file {path} must include machine_id and agent_token"
        )
        sys.exit(1)

    return data


credentials = load_credentials()

CONTROL_PLANE_URL = (
    os.environ.get("CONTROL_PLANE_URL")
    or credentials.get("control_plane_url")
    or DEFAULT_CONTROL_PLANE_URL
)
MACHINE_ID = credentials["machine_id"]
AGENT_TOKEN = credentials["agent_token"]

websocket_send_lock = threading.Lock()
active_command_lock = threading.Lock()
active_command_id = None


def get_gpu_stats():
    command = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,utilization.gpu,"
        "memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    gpus = []

    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]

        name = parts[0]
        temperature = float(parts[1])
        utilization = float(parts[2])
        memory_used = int(float(parts[3]))
        memory_total = int(float(parts[4]))
        power_draw = float(parts[5]) if parts[5] not in ("", "[N/A]") else None

        gpu = {
            "name": name,
            "temperature": temperature,
            "utilization": utilization,
            "memory_used": memory_used,
            "memory_total": memory_total,
        }
        if power_draw is not None:
            gpu["power_draw"] = power_draw

        gpus.append(gpu)

    return gpus


def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, OSError):
        return None

    if not temps:
        return None

    preferred_keys = ("coretemp", "k10temp", "cpu_thermal", "acpitz")
    for key in preferred_keys:
        entries = temps.get(key)
        if entries:
            values = [entry.current for entry in entries if entry.current is not None]
            if values:
                return round(sum(values) / len(values), 1)

    for entries in temps.values():
        values = [entry.current for entry in entries if entry.current is not None]
        if values:
            return round(sum(values) / len(values), 1)

    return None


def get_cpu_stats():
    return {
        "utilization": round(psutil.cpu_percent(interval=0.2), 1),
        "temperature": get_cpu_temperature(),
        "cores": psutil.cpu_count(logical=True),
    }


def get_memory_stats():
    memory = psutil.virtual_memory()
    return {
        "used": int(memory.used),
        "total": int(memory.total),
        "percent": round(memory.percent, 1),
    }


def get_storage_stats():
    storage = []
    seen_devices = set()

    for partition in psutil.disk_partitions(all=False):
        if partition.fstype in ("", "tmpfs", "devtmpfs", "squashfs", "overlay"):
            continue
        if not partition.mountpoint:
            continue
        if partition.device in seen_devices:
            continue

        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except (PermissionError, OSError):
            continue

        seen_devices.add(partition.device)
        storage.append({
            "mount": partition.mountpoint,
            "device": partition.device,
            "used": int(usage.used),
            "total": int(usage.total),
            "free": int(usage.free),
            "percent": round(usage.percent, 1),
        })

    storage.sort(key=lambda item: (item["mount"] != "/", item["mount"]))
    return storage


def get_lan_ip():
    try:
        result = subprocess.run(
            ["ip", "-4", "route", "get", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.split()
            if "src" in parts:
                return parts[parts.index("src") + 1]
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except Exception:
        pass

    return None


def collect_telemetry():
    return {
        "gpus": get_gpu_stats(),
        "cpu": get_cpu_stats(),
        "memory": get_memory_stats(),
        "storage": get_storage_stats(),
    }


def send_heartbeat():
    telemetry = collect_telemetry()
    lan_ip = get_lan_ip()
    body = {
        "machine_id": MACHINE_ID,
        "gpus": telemetry["gpus"],
        "cpu": telemetry["cpu"],
        "memory": telemetry["memory"],
        "storage": telemetry["storage"],
    }
    if lan_ip:
        body["lan_ip"] = lan_ip

    response = requests.post(
        f"{CONTROL_PLANE_URL}/machines/heartbeat",
        headers={
            "Authorization": f"Bearer {AGENT_TOKEN}",
            "Content-Type": "application/json"
        },
        json=body,
        timeout=5
    )

    response.raise_for_status()

    print("Heartbeat:", response.json())
    print("Telemetry:", {
        "gpus": telemetry["gpus"],
        "cpu": telemetry["cpu"],
        "memory": telemetry["memory"],
        "storage": telemetry["storage"],
        "lan_ip": lan_ip,
    })

def poll_commands():
    global active_command_id

    with active_command_lock:
        if active_command_id is not None:
            return

    try:
        response = requests.get(
            f"{CONTROL_PLANE_URL}/machines/{MACHINE_ID}/commands/next",
            headers={
                "Authorization": f"Bearer {AGENT_TOKEN}"
            },
            timeout=10
        )

        response.raise_for_status()

        data = response.json()
        command = data.get("command")

        if not command:
            return

        command_type = command.get("type")
        command_id = command.get("commandId")
        print(f"Received command: {command_type}")

        if command_type == "PING":
            complete_command(command_id, "COMPLETED", "PONG")
            return

        if command_type == "INFERENCE":
            try:
                inference_response = requests.post(
                    VLLM_CHAT_COMPLETIONS_URL,
                    json=command.get("payload") or {},
                    timeout=300
                )
                inference_response.raise_for_status()
                complete_command(
                    command_id,
                    "COMPLETED",
                    inference_response.json()
                )
            except Exception as error:
                print(f"Inference error: {error}")
                complete_command(command_id, "FAILED", str(error), error=str(error))
            return

        if command_type in (
            "INSTALL_LLM",
            "OFFLOAD_LLM",
            "ONLOAD_LLM",
            "DELETE_LLM",
            "INSTALL_APP",
            "OFFLOAD_APP",
            "ONLOAD_APP",
            "DELETE_APP",
            "DEPLOY_WORKFLOW",
            "REMOVE_WORKFLOW",
            "RESTART_MACHINE",
            "SHUTDOWN_MACHINE",
        ):
            with active_command_lock:
                active_command_id = command_id
            worker = threading.Thread(
                target=handle_queued_command,
                args=(command,),
                daemon=True,
                name=f"cmd-{command_id}"
            )
            worker.start()
            return

        print(f"Unknown command type: {command_type}")
        complete_command(
            command_id,
            "FAILED",
            f"Unknown command type: {command_type}",
            error=f"Unknown command type: {command_type}"
        )
    except Exception as error:
        print(f"Command polling error: {error}")


class InstallError(Exception):
    """Raised for install/onload failures with a taxonomy error_code (see
    private-ai-cp's shared/constants.py INSTALL_ERROR_TAXONOMY) so the
    backend can render a specific, actionable message instead of a raw
    exception string. `detail` is the raw diagnostic (docker logs, etc.) —
    stored as errorDetail, not shown to the user by default.
    """

    def __init__(self, error_code, message, detail=None):
        self.error_code = error_code
        self.detail = detail
        super().__init__(message)


class InstallCancelled(Exception):
    """Raised when a cancel signal was observed and honored mid-install —
    handle_queued_command completes this as CANCELLED, not FAILED."""


def report_progress(
    command_id,
    phase,
    message,
    percent=None,
    log_line=None
):
    body = {
        "phase": phase,
        "message": message,
        "percent": percent,
    }
    if log_line:
        body["log_line"] = log_line

    try:
        response = requests.post(
            f"{CONTROL_PLANE_URL}/machines/{MACHINE_ID}/commands/{command_id}/progress",
            headers={
                "Authorization": f"Bearer {AGENT_TOKEN}",
                "Content-Type": "application/json"
            },
            json=body,
            timeout=15
        )
        response.raise_for_status()
        print(f"Progress [{phase}]: {message}")
        return response.json()
    except Exception as error:
        print(f"Progress report error: {error}")
        return None


def raise_if_cancelled(progress_response):
    """report_progress already round-trips to the control plane every few
    seconds during long phases — piggyback the cancel check on that instead
    of a separate poll, so honoring a cancel request is prompt without extra
    load."""
    if progress_response and progress_response.get("cancel_requested"):
        raise InstallCancelled()


def run_subprocess(command, timeout=None):
    print("Running:", " ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        joined = " ".join(command)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {joined}\n{stderr}"
        )
    return result


def _extract_container_id(run_result):
    """`docker run -d` prints the new container's full ID as the last
    stdout line on success."""
    if not run_result or not run_result.stdout:
        return None
    lines = [line.strip() for line in run_result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    return candidate if re.fullmatch(r"[0-9a-f]{12,64}", candidate) else None


def docker_container_state(container_name):
    """Returns {"status": "running"|"exited"|..., "exit_code": int} or None
    if the container doesn't exist / inspect fails — used to tell "still
    starting" apart from "already crashed" during health-check polling,
    instead of waiting out the full timeout on a dead container.
    """
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}|{{.State.ExitCode}}", container_name],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return None
    output = (result.stdout or "").strip()
    if "|" not in output:
        return None
    status, _, exit_code = output.partition("|")
    try:
        return {"status": status, "exit_code": int(exit_code)}
    except ValueError:
        return {"status": status, "exit_code": None}


def docker_container_logs(container_name, tail=100):
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_name],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return (result.stdout or "") + (result.stderr or "")


def docker_image_present(container_image):
    result = subprocess.run(
        ["docker", "image", "inspect", container_image],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0


def _pull_progress_percent(line):
    lower = line.lower()
    if "downloading" in lower:
        return 35
    if "extracting" in lower or "extract" in lower:
        return 50
    if "pull complete" in lower or "digest:" in lower or "status: downloaded" in lower:
        return 60
    if "waiting" in lower or "verifying" in lower:
        return 55
    return 25


_LAYER_PROGRESS_RE = re.compile(
    r"^[0-9a-f]{12}:\s*(?:Downloading|Extracting)\s*\[.*?\]\s*"
    r"([\d.]+)\s*([kKmMgG]?B)\s*/\s*([\d.]+)\s*([kKmMgG]?B)"
)
_SIZE_MULTIPLIERS = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}


def _parse_size(value, unit):
    return float(value) * _SIZE_MULTIPLIERS.get(unit.lower(), 1)


def _real_pull_percent(line, layer_bytes):
    """Docker's non-TTY pull output still prints periodic per-layer
    "Downloading [===>   ] 4.2MB/9.7MB" lines — parse and sum them for a
    real byte-based percent instead of the fixed-keyword guess, when the
    format is there to parse. Returns None (caller falls back to the
    keyword-based estimate) when it isn't.
    """
    match = _LAYER_PROGRESS_RE.match(line)
    if not match:
        return None
    layer_id = line.split(":", 1)[0]
    downloaded = _parse_size(match.group(1), match.group(2))
    total = _parse_size(match.group(3), match.group(4))
    if total <= 0:
        return None
    layer_bytes[layer_id] = (downloaded, total)
    total_downloaded = sum(d for d, _ in layer_bytes.values())
    total_size = sum(t for _, t in layer_bytes.values())
    if total_size <= 0:
        return None
    fraction = min(1.0, total_downloaded / total_size)
    # This phase occupies roughly 25-65% of the overall install progress.
    return 25 + fraction * 40, total_downloaded, total_size


def docker_pull_with_progress(command_id, container_image, timeout=3600):
    """Pull image while streaming layer progress to control plane."""
    if docker_image_present(container_image):
        report_progress(
            command_id,
            "DOWNLOADING",
            f"Using cached image {container_image}",
            percent=65,
            log_line=f"docker image inspect {container_image} (cached)",
        )
        return

    command = ["docker", "pull", container_image]
    print("Running:", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_report = 0.0
    last_line = ""
    last_percent = 25
    layer_bytes = {}
    deadline = time.time() + timeout
    stop_heartbeat = threading.Event()

    def pull_heartbeat():
        nonlocal last_line, last_percent
        while not stop_heartbeat.wait(30):
            if process.poll() is not None:
                return
            message = last_line or (
                f"Still pulling {container_image} — large GPU image layers can "
                "take 30–60 min with no new log lines while extracting"
            )
            progress_response = report_progress(
                command_id,
                "DOWNLOADING",
                message[:500],
                percent=last_percent,
                log_line=message[:500],
            )
            if progress_response and progress_response.get("cancel_requested"):
                process.kill()

    heartbeat = threading.Thread(
        target=pull_heartbeat,
        daemon=True,
        name=f"pull-heartbeat-{command_id}",
    )
    heartbeat.start()

    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line)
            last_line = line

            now = time.time()
            byte_progress = _real_pull_percent(line, layer_bytes)
            if byte_progress:
                last_percent, downloaded, total = byte_progress
                display_message = (
                    f"Downloading model image — "
                    f"{downloaded / (1024 * 1024):.0f} MB / {total / (1024 * 1024):.0f} MB"
                )
            else:
                last_percent = _pull_progress_percent(line)
                display_message = line[:500]

            if now - last_report >= 5 or "Pull complete" in line or "Status:" in line:
                progress_response = report_progress(
                    command_id,
                    "DOWNLOADING",
                    display_message,
                    percent=last_percent,
                    log_line=line[:500],
                )
                last_report = now
                if progress_response and progress_response.get("cancel_requested"):
                    process.kill()
                    raise InstallCancelled()

            if now > deadline:
                process.kill()
                raise InstallError(
                    "DOWNLOAD_FAILED",
                    f"docker pull timed out after {timeout}s: {container_image}",
                )

        return_code = process.wait(timeout=30)
    except (InstallCancelled, InstallError):
        process.kill()
        raise
    except Exception:
        process.kill()
        raise
    finally:
        stop_heartbeat.set()

    if return_code != 0:
        raise InstallError(
            "DOWNLOAD_FAILED",
            f"Failed to download the model image (exit code {return_code}).",
            detail=f"docker pull {container_image}\n{last_line}",
        )

    report_progress(
        command_id,
        "DOWNLOADING",
        f"Pulled {container_image}",
        percent=65,
        log_line=f"docker pull {container_image} (complete)",
    )


def docker_available():
    run_subprocess(["docker", "info"], timeout=30)


def ensure_hf_cache_dir():
    os.makedirs(HF_CACHE_HOST, exist_ok=True)


def hf_hub_cache_dir(model_path: str) -> str:
    """Hugging Face Hub cache folder for a repo id (org/name)."""
    slug = model_path.strip().replace("/", "--")
    return os.path.join(HF_CACHE_HOST, "hub", f"models--{slug}")


def remove_hf_cache(model_path: str) -> bool:
    """Delete downloaded model weights from the host HF cache. Returns True if removed."""
    cache_dir = hf_hub_cache_dir(model_path)
    existed = os.path.isdir(cache_dir)
    hub_rel = os.path.relpath(cache_dir, HF_CACHE_HOST)
    # Cache files are owned by root (written by the vLLM container). Use a
    # short-lived container so deletion works when the agent runs unprivileged.
    run_subprocess([
        "docker", "run", "--rm",
        "-v", f"{HF_CACHE_HOST}:/cache:rw",
        "alpine:3",
        "rm", "-rf", f"/cache/{hub_rel}",
    ], timeout=300)
    return existed


def vllm_docker_run_args(container_name, container_image, model_path, port, util):
    ensure_hf_cache_dir()
    return [
        "docker", "run", "-d",
        "--gpus", "all",
        "-p", f"{port}:8000",
        "--name", container_name,
        "--restart", "unless-stopped",
        "-v", f"{HF_CACHE_HOST}:/root/.cache/huggingface",
        container_image,
        "--model", model_path,
        "--served-model-name", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", util,
        "--max-model-len", "2048",
    ]


def cpu_kv_cache_gb(ram_allocated_mb):
    """KV cache must share RAM with model weights — don't use the full budget."""
    total_gb = max(1.0, ram_allocated_mb / 1024.0)
    return max(2, min(16, int(total_gb * 0.35)))


def vllm_cpu_docker_run_args(container_name, container_image, model_path, port, ram_allocated_mb):
    ensure_hf_cache_dir()
    kv_cache_gb = cpu_kv_cache_gb(ram_allocated_mb)
    return [
        "docker", "run", "-d",
        "--cap-add", "SYS_NICE",
        "--security-opt", "seccomp=unconfined",
        "--ipc=host",
        "--shm-size", "4g",
        "-p", f"{port}:8000",
        "--name", container_name,
        "--restart", "unless-stopped",
        "-v", f"{HF_CACHE_HOST}:/root/.cache/huggingface",
        "-e", f"VLLM_CPU_KVCACHE_SPACE={kv_cache_gb}",
        "-e", "VLLM_CPU_OMP_THREADS_BIND=auto",
        container_image,
        "--model", model_path,
        "--served-model-name", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "bfloat16",
        "--max-model-len", "512",
        "--max-num-seqs", "4",
        "--max-num-batched-tokens", "512",
        "--trust-remote-code",
    ]


def docker_run_args_for_payload(payload):
    runtime = payload.get("runtime") or "gpu"
    container_name = payload.get("containerName")
    container_image = payload.get("containerImage")
    model_path = payload.get("modelPath")
    port = payload.get("port")
    if not all([container_name, container_image, model_path, port]):
        raise ValueError("Missing fields for docker run (container/image/model/port)")

    if runtime == "cpu":
        ram_mb = payload.get("ramAllocatedMb") or 4096
        return vllm_cpu_docker_run_args(
            container_name, container_image, model_path, port, ram_mb
        )

    util = resolve_gpu_memory_utilization(payload)
    return vllm_docker_run_args(
        container_name, container_image, model_path, port, util
    )


def install_health_timeout_seconds(payload):
    runtime = payload.get("runtime") or "gpu"
    # GPU first boot often downloads multi-GB weights then loads VRAM; 5m was too short.
    return 600 if runtime == "cpu" else 900


def revalidate_resources_before_run(payload):
    """Last-second guard immediately before `docker run` — resources can
    change between the control plane's preflight check and now (e.g. a
    second install raced this one and consumed VRAM in between). The
    control plane stays the authoritative decision-maker; this is a final
    safety net, not a second admission-control system, so it fails open
    (returns quietly) whenever it can't get a confident reading rather than
    blocking an install over a telemetry hiccup.
    """
    runtime = payload.get("runtime") or "gpu"
    if runtime == "cpu":
        required_mb = payload.get("ramAllocatedMb")
        if not required_mb:
            return
        free_mb = psutil.virtual_memory().available / (1024 * 1024)
        if free_mb < float(required_mb):
            raise InstallError(
                "INSUFFICIENT_RAM",
                f"Not enough free RAM right before starting: need {required_mb} MB, have {int(free_mb)} MB.",
            )
    else:
        required_mb = payload.get("vramAllocatedMb")
        if not required_mb:
            return
        try:
            gpus = get_gpu_stats()
        except Exception:
            return
        free_mb = sum(max(0, g["memory_total"] - g["memory_used"]) for g in gpus)
        if free_mb < float(required_mb):
            raise InstallError(
                "INSUFFICIENT_VRAM",
                f"Not enough free VRAM right before starting: need {required_mb} MB, have {int(free_mb)} MB.",
            )


def handle_install_llm(command_id, payload):
    container_name = payload.get("containerName")
    container_image = payload.get("containerImage")
    model_path = payload.get("modelPath")
    port = payload.get("port")
    runtime = payload.get("runtime") or "gpu"

    if not all([container_name, container_image, model_path, port]):
        raise ValueError("INSTALL_LLM payload missing required fields")

    report_progress(command_id, "VALIDATING", "Checking Docker", percent=5)
    docker_available()

    report_progress(
        command_id,
        "DOWNLOADING",
        f"Pulling {container_image} (this can take 20–40 min on first run)",
        percent=25,
        log_line=f"docker pull {container_image}"
    )
    pull_attempts = 3
    last_error = None
    for attempt in range(1, pull_attempts + 1):
        try:
            docker_pull_with_progress(command_id, container_image, timeout=3600)
            last_error = None
            break
        except InstallCancelled:
            raise
        except Exception as error:
            last_error = error
            if attempt >= pull_attempts:
                raise
            report_progress(
                command_id,
                "DOWNLOADING",
                f"Pull failed (attempt {attempt}/{pull_attempts}): {error}. Retrying…",
                percent=25,
                log_line=str(error)[:500],
            )
            time.sleep(5)
    if last_error:
        raise last_error

    report_progress(command_id, "PREPARING", "Preparing container", percent=68)
    # Remove any leftover container with the same name.
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True
    )
    revalidate_resources_before_run(payload)

    report_progress(
        command_id,
        "STARTING",
        f"Starting {container_name} on port {port} ({runtime})",
        percent=70,
        log_line=f"docker run {container_name} runtime={runtime}"
    )
    run_result = run_subprocess(
        docker_run_args_for_payload(payload),
        timeout=120,
    )
    container_id = _extract_container_id(run_result)

    report_progress(
        command_id,
        "LOADING_MODEL",
        "Container started — waiting for the model to load",
        percent=80,
    )

    served_model_id = wait_for_model_health(
        command_id,
        port,
        container_name,
        timeout_seconds=install_health_timeout_seconds(payload),
    )
    result = {
        "port": port,
        "container_name": container_name,
        "model_path": model_path,
        "runtime": runtime,
    }
    if container_id:
        result["container_id"] = container_id
    if served_model_id:
        result["served_model_id"] = served_model_id
    if runtime == "cpu":
        result["ram_allocated_mb"] = payload.get("ramAllocatedMb")
    else:
        result["gpu_memory_utilization"] = resolve_gpu_memory_utilization(payload)
    complete_command(command_id, "COMPLETED", result)


def handle_offload_llm(command_id, payload):
    container_name = payload.get("containerName")
    if not container_name:
        raise ValueError("OFFLOAD_LLM payload missing containerName")

    report_progress(
        command_id,
        "stop_container",
        f"Stopping {container_name}",
        percent=50,
        log_line=f"docker stop {container_name}"
    )
    result = subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "no such container" not in stderr and "is not running" not in stderr:
            raise RuntimeError(
                f"docker stop failed: {(result.stderr or result.stdout).strip()}"
            )

    complete_command(
        command_id,
        "COMPLETED",
        {"container_name": container_name, "action": "offloaded"}
    )


def handle_delete_llm(command_id, payload):
    container_name = payload.get("containerName")
    model_path = payload.get("modelPath")
    if not container_name:
        raise ValueError("DELETE_LLM payload missing containerName")

    report_progress(
        command_id,
        "stop_container",
        f"Stopping {container_name}",
        percent=20,
        log_line=f"docker stop {container_name}"
    )
    subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True,
        text=True
    )

    report_progress(
        command_id,
        "remove_container",
        f"Removing {container_name}",
        percent=45,
        log_line=f"docker rm -fv {container_name}"
    )
    result = subprocess.run(
        ["docker", "rm", "-fv", container_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "no such container" not in stderr:
            raise RuntimeError(
                f"docker rm failed: {(result.stderr or result.stdout).strip()}"
            )

    cache_removed = False
    if model_path:
        report_progress(
            command_id,
            "remove_cache",
            f"Removing model files for {model_path}",
            percent=75,
            log_line=f"rm -rf {hf_hub_cache_dir(model_path)}"
        )
        try:
            cache_removed = remove_hf_cache(model_path)
        except OSError as err:
            raise RuntimeError(
                f"Failed to remove model cache for {model_path}: {err}"
            ) from err

    complete_command(
        command_id,
        "COMPLETED",
        {
            "container_name": container_name,
            "model_path": model_path,
            "cache_removed": cache_removed,
            "action": "deleted",
        }
    )


def resolve_gpu_memory_utilization(payload):
    raw = payload.get("gpuMemoryUtilization")
    if raw is None:
        return "0.30"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return "0.30"
    value = max(0.05, min(0.95, value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def fetch_served_model_id(port):
    response = requests.get(
        f"http://127.0.0.1:{port}/v1/models",
        timeout=10,
    )
    response.raise_for_status()
    models = response.json().get("data") or []
    if not models:
        return None
    return models[0].get("id")


_served_model_by_port = {}


def resolve_vllm_model_id(port, requested_model=None):
    cached = _served_model_by_port.get(port)
    if cached:
        return cached
    try:
        served = fetch_served_model_id(port)
    except Exception as error:
        print(f"Could not resolve served model id on port {port}: {error}")
        served = None
    if served:
        _served_model_by_port[int(port)] = served
        return served
    return requested_model


def smoke_test_inference(port, model_id):
    """One real, tiny completion request — READY should mean "a request
    would actually succeed," not just "the container exists and /v1/models
    responds." Raises InstallError(MODEL_LOAD_FAILED) on a genuine failure.
    """
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": model_id or "default",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=30,
        )
    except Exception as error:
        raise InstallError(
            "MODEL_LOAD_FAILED",
            "The model server is up but a test inference request failed.",
            detail=str(error),
        ) from error

    if response.status_code != 200:
        raise InstallError(
            "MODEL_LOAD_FAILED",
            f"The model server is up but a test inference request failed (HTTP {response.status_code}).",
            detail=(response.text or "")[:2000],
        )


def wait_for_model_health(command_id, port, container_name, timeout_seconds=300):
    started = time.time()
    report_progress(
        command_id,
        "HEALTH_CHECKING",
        "Waiting for model server to become ready…",
        percent=90
    )
    deadline = time.time() + timeout_seconds
    last_error = None
    last_health_report = 0.0
    served_model_id = None
    while time.time() < deadline:
        try:
            response = requests.get(
                f"http://127.0.0.1:{port}/v1/models",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("data") or []
                if models:
                    served_model_id = models[0].get("id")
                    if served_model_id:
                        _served_model_by_port[int(port)] = served_model_id

                report_progress(
                    command_id,
                    "HEALTH_CHECKING",
                    "Model server responded — confirming it can actually serve a request",
                    percent=93,
                )
                smoke_test_inference(port, served_model_id)

                report_progress(
                    command_id,
                    "HEALTH_CHECKING",
                    "Model server is healthy",
                    percent=95,
                    log_line="health check + smoke test ok"
                )
                return served_model_id
            last_error = f"HTTP {response.status_code}"
        except InstallError:
            raise
        except Exception as error:
            last_error = str(error)

        # The container may have crashed rather than "still starting" — don't
        # wait out the whole timeout to discover that.
        state = docker_container_state(container_name) if container_name else None
        if state and state.get("status") not in ("running", "created"):
            logs = docker_container_logs(container_name)
            raise InstallError(
                "CONTAINER_FAILED",
                f"The container exited unexpectedly "
                f"(status: {state.get('status')}, exit code: {state.get('exit_code')}).",
                detail=logs[-4000:],
            )

        now = time.time()
        if now - last_health_report >= 10:
            elapsed = int(now - started)
            progress_response = report_progress(
                command_id,
                "HEALTH_CHECKING",
                f"Waiting for model server to become ready… ({elapsed}s)",
                percent=90,
                log_line=f"health poll: {last_error or 'starting'}"
            )
            raise_if_cancelled(progress_response)
            last_health_report = now
        time.sleep(5)

    raise InstallError(
        "HEALTH_CHECK_TIMEOUT",
        f"Model server did not become healthy within {timeout_seconds}s: {last_error}",
        detail=docker_container_logs(container_name)[-4000:] if container_name else None,
    )


def docker_run_vllm(command_id, payload):
    container_name = payload.get("containerName")
    port = payload.get("port")
    runtime = payload.get("runtime") or "gpu"

    if not container_name or port is None:
        raise ValueError("Missing fields for docker run (container/port)")

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True
    )

    revalidate_resources_before_run(payload)

    report_progress(
        command_id,
        "STARTING",
        f"Starting {container_name} on port {port} ({runtime})",
        percent=70,
        log_line=f"docker run {container_name} runtime={runtime}"
    )
    run_result = run_subprocess(
        docker_run_args_for_payload(payload),
        timeout=120,
    )
    return _extract_container_id(run_result)


def handle_onload_llm(command_id, payload):
    container_name = payload.get("containerName")
    port = payload.get("port")
    if not container_name or port is None:
        raise ValueError("ONLOAD_LLM payload missing containerName or port")

    report_progress(
        command_id,
        "STARTING",
        f"Starting {container_name}",
        percent=40,
        log_line=f"docker start {container_name}"
    )
    container_id = None
    result = subprocess.run(
        ["docker", "start", container_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").lower()
        if "no such container" in stderr:
            report_progress(
                command_id,
                "STARTING",
                "Container missing; recreating with docker run",
                percent=50,
                log_line="fallback docker run"
            )
            container_id = docker_run_vllm(command_id, payload)
        else:
            raise InstallError(
                "CONTAINER_FAILED",
                "Could not start the container.",
                detail=(result.stderr or result.stdout).strip(),
            )

    report_progress(
        command_id,
        "LOADING_MODEL",
        "Container started — waiting for the model to load",
        percent=80,
    )

    served_model_id = wait_for_model_health(
        command_id,
        port,
        container_name,
        timeout_seconds=install_health_timeout_seconds(payload),
    )
    onload_result = {
        "port": port,
        "container_name": container_name,
        "action": "onloaded",
    }
    if container_id:
        onload_result["container_id"] = container_id
    if served_model_id:
        onload_result["served_model_id"] = served_model_id
    complete_command(
        command_id,
        "COMPLETED",
        onload_result,
    )


def app_docker_run_args(container_name, container_image, port, data_volume_name):
    """Generic app container: no CLI flags appended, unlike vLLM's run args —
    the image runs as-is. Only n8n uses this today, hence the fixed internal
    port; if a second app type is added, make N8N_CONTAINER_PORT a payload
    field instead of a module constant.
    """
    return [
        "docker", "run", "-d",
        "-p", f"{port}:{N8N_CONTAINER_PORT}",
        "--name", container_name,
        "--restart", "unless-stopped",
        "-v", f"{data_volume_name}:/home/node/.n8n",
        # Every hop to this container is plain HTTP (agent -> 127.0.0.1, and
        # direct LAN access for manual debugging) — no TLS in front of it.
        # n8n's default Secure-flagged auth cookie never gets sent back over
        # HTTP, which silently breaks session auth (login "succeeds" but the
        # cookie is dropped on the next request) without this.
        "-e", "N8N_SECURE_COOKIE=false",
        container_image,
    ]


def wait_for_app_health(command_id, port, timeout_seconds=180):
    started = time.time()
    report_progress(
        command_id,
        "health_check",
        "Waiting for n8n to become ready…",
        percent=90
    )
    deadline = time.time() + timeout_seconds
    last_error = None
    last_health_report = 0.0
    while time.time() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/", timeout=5)
            if response.status_code < 400:
                report_progress(
                    command_id,
                    "health_check",
                    "n8n is healthy",
                    percent=95,
                    log_line="health check ok"
                )
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as error:
            last_error = str(error)

        now = time.time()
        if now - last_health_report >= 10:
            elapsed = int(now - started)
            report_progress(
                command_id,
                "health_check",
                f"Waiting for n8n to become ready… ({elapsed}s)",
                percent=90,
                log_line=f"health poll: {last_error or 'starting'}"
            )
            last_health_report = now
        time.sleep(5)

    raise TimeoutError(f"n8n did not become healthy: {last_error}")


_container_name_by_port = {}


def handle_install_app(command_id, payload):
    container_name = payload.get("containerName")
    container_image = payload.get("containerImage")
    port = payload.get("port")
    data_volume_name = payload.get("dataVolumeName")

    if not all([container_name, container_image, port, data_volume_name]):
        raise ValueError("INSTALL_APP payload missing required fields")

    report_progress(command_id, "validate", "Checking Docker", percent=5)
    docker_available()

    report_progress(
        command_id,
        "pull_image",
        f"Pulling {container_image}",
        percent=25,
        log_line=f"docker pull {container_image}"
    )
    docker_pull_with_progress(command_id, container_image, timeout=1200)

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True
    )

    report_progress(
        command_id,
        "start_container",
        f"Starting {container_name} on port {port}",
        percent=70,
        log_line=f"docker run {container_name}"
    )
    run_subprocess(
        app_docker_run_args(container_name, container_image, port, data_volume_name),
        timeout=120,
    )

    wait_for_app_health(command_id, port, timeout_seconds=180)
    _container_name_by_port[int(port)] = container_name

    complete_command(
        command_id,
        "COMPLETED",
        {
            "port": port,
            "container_name": container_name,
            "data_volume_name": data_volume_name,
        }
    )


def handle_offload_app(command_id, payload):
    container_name = payload.get("containerName")
    if not container_name:
        raise ValueError("OFFLOAD_APP payload missing containerName")

    report_progress(
        command_id,
        "stop_container",
        f"Stopping {container_name}",
        percent=50,
        log_line=f"docker stop {container_name}"
    )
    result = subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "no such container" not in stderr and "is not running" not in stderr:
            raise RuntimeError(
                f"docker stop failed: {(result.stderr or result.stdout).strip()}"
            )

    complete_command(
        command_id,
        "COMPLETED",
        {"container_name": container_name, "action": "offloaded"}
    )


def handle_onload_app(command_id, payload):
    container_name = payload.get("containerName")
    port = payload.get("port")
    if not container_name or port is None:
        raise ValueError("ONLOAD_APP payload missing containerName or port")

    report_progress(
        command_id,
        "start_container",
        f"Starting {container_name}",
        percent=40,
        log_line=f"docker start {container_name}"
    )
    result = subprocess.run(
        ["docker", "start", container_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").lower()
        if "no such container" in stderr:
            container_image = payload.get("containerImage")
            data_volume_name = payload.get("dataVolumeName")
            if not container_image or not data_volume_name:
                raise ValueError(
                    "Container missing and cannot recreate: missing "
                    "containerImage/dataVolumeName"
                )
            report_progress(
                command_id,
                "start_container",
                "Container missing; recreating with docker run",
                percent=50,
                log_line="fallback docker run"
            )
            run_subprocess(
                app_docker_run_args(container_name, container_image, port, data_volume_name),
                timeout=120,
            )
        else:
            raise RuntimeError(
                f"docker start failed: {(result.stderr or result.stdout).strip()}"
            )

    wait_for_app_health(command_id, port, timeout_seconds=180)
    _container_name_by_port[int(port)] = container_name

    complete_command(
        command_id,
        "COMPLETED",
        {"port": port, "container_name": container_name, "action": "onloaded"}
    )


def handle_delete_app(command_id, payload):
    container_name = payload.get("containerName")
    data_volume_name = payload.get("dataVolumeName")
    if not container_name:
        raise ValueError("DELETE_APP payload missing containerName")

    report_progress(
        command_id,
        "stop_container",
        f"Stopping {container_name}",
        percent=20,
        log_line=f"docker stop {container_name}"
    )
    subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True,
        text=True
    )

    report_progress(
        command_id,
        "remove_container",
        f"Removing {container_name}",
        percent=45,
        log_line=f"docker rm -fv {container_name}"
    )
    result = subprocess.run(
        ["docker", "rm", "-fv", container_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "no such container" not in stderr:
            raise RuntimeError(
                f"docker rm failed: {(result.stderr or result.stdout).strip()}"
            )

    # Unlike the shared HF cache (a read-mostly cache safe to prune by
    # sub-path), this volume is exclusive to one n8n instance — removing it
    # is real, irreversible data loss, gated by confirm=true at the API layer.
    volume_removed = False
    if data_volume_name:
        report_progress(
            command_id,
            "remove_volume",
            f"Removing volume {data_volume_name}",
            percent=80,
            log_line=f"docker volume rm {data_volume_name}"
        )
        vol_result = subprocess.run(
            ["docker", "volume", "rm", data_volume_name],
            capture_output=True,
            text=True
        )
        if vol_result.returncode == 0:
            volume_removed = True
        else:
            stderr = (vol_result.stderr or "").lower()
            if "no such volume" not in stderr:
                raise RuntimeError(
                    f"docker volume rm failed: "
                    f"{(vol_result.stderr or vol_result.stdout).strip()}"
                )

    complete_command(
        command_id,
        "COMPLETED",
        {
            "container_name": container_name,
            "data_volume_name": data_volume_name,
            "volume_removed": volume_removed,
            "action": "deleted",
        }
    )


def _n8n_owner_state_path(container_name):
    return os.path.join(N8N_STATE_DIR, f"{container_name}.json")


_n8n_api_key_by_port = {}


def _n8n_write_state(state_path, state):
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
    os.chmod(state_path, 0o600)


def _n8n_forget_api_key(port, container_name):
    """Drop both the in-memory and on-disk cached key so the next
    _n8n_ensure_api_key call re-provisions instead of returning a value
    already known to be invalid (e.g. after a 401, or after the key was
    deleted from n8n's own UI).
    """
    _n8n_api_key_by_port.pop(int(port), None)
    state_path = _n8n_owner_state_path(container_name)
    if not os.path.isfile(state_path):
        return
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    if state.pop("apiKey", None) is not None:
        _n8n_write_state(state_path, state)


def _n8n_ensure_api_key(port, container_name):
    """Bootstrap a non-interactive owner account + Public API key for this
    n8n instance, on first use, and cache/reuse it afterward — in memory for
    the life of the process, and on disk (alongside the owner login) so it
    survives an agent restart without re-provisioning.

    The on-disk cache isn't just an optimization: n8n only ever returns a
    key's raw secret value in the response to the call that created it —
    `GET /rest/api-keys` (listing) returns metadata only (label, scopes,
    expiry), never the value. So once the in-memory cache is lost and there's
    no on-disk copy, a previously-created key is unrecoverable by value, even
    though n8n still shows it exists — the only way forward is to delete
    that stale entry and create a new one, which is what happens below when
    neither cache has a value.

    NOTE — verification spike: this targets the REST shape used by n8n's own
    setup wizard (`/rest/owner/setup`, `/rest/login`, `/rest/api-keys`) as of
    the version pinned by the control plane's N8N_CONTAINER_IMAGE. This is
    not a documented/stable public contract — confirm it against that exact
    image tag before relying on it in production. If it changes, only this
    function needs updating; nothing else in the workflow/credential flow
    depends on the exact bootstrap mechanism.
    """
    cached = _n8n_api_key_by_port.get(int(port))
    if cached:
        return cached

    os.makedirs(N8N_STATE_DIR, exist_ok=True)
    state_path = _n8n_owner_state_path(container_name)
    base_url = f"http://127.0.0.1:{port}"

    if os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    else:
        # Suffix guarantees n8n's password-complexity check (upper+lower+digit)
        # passes regardless of what token_urlsafe happens to generate.
        state = {
            "email": f"owner-{uuid.uuid4().hex[:12]}@local.invalid",
            "password": secrets.token_urlsafe(24) + "aA1",
        }
        _n8n_write_state(state_path, state)

    if state.get("apiKey"):
        _n8n_api_key_by_port[int(port)] = state["apiKey"]
        return state["apiKey"]

    session = requests.Session()

    # First run creates the owner account — and n8n logs the caller in as a
    # side effect of a successful setup, via the same session's cookie. Only
    # call /rest/login explicitly when setup did NOT just succeed (i.e. the
    # instance was already set up from an earlier run); calling login a
    # second time right after a fresh setup risks stepping on/replacing that
    # already-valid session, which is the likely cause of a 401 on the
    # api-keys call further down despite setup+login both reporting success.
    setup_response = session.post(
        f"{base_url}/rest/owner/setup",
        json={
            "email": state["email"],
            "firstName": "Private",
            "lastName": "AI",
            "password": state["password"],
        },
        timeout=15,
    )

    if not setup_response.ok:
        if "already" not in setup_response.text.lower():
            print(
                f"n8n owner setup returned {setup_response.status_code} "
                f"(attempting login anyway): {setup_response.text[:500]}"
            )
        login_response = session.post(
            f"{base_url}/rest/login",
            # Field name for the identifier varies across n8n versions
            # (plain `email` vs. LDAP-aware `emailOrLdapLoginId`) — send
            # both so whichever schema this image expects gets a value.
            json={
                "email": state["email"],
                "emailOrLdapLoginId": state["email"],
                "password": state["password"],
            },
            timeout=15,
        )
        if not login_response.ok:
            raise RuntimeError(
                f"n8n login failed ({login_response.status_code}): "
                f"{login_response.text[:500]}"
            )

    if not session.cookies:
        raise RuntimeError(
            "n8n setup/login reported success but no session cookie was set "
            "— cannot make authenticated API calls"
        )

    # No on-disk value (first run on this instance, or it was invalidated
    # after a 401/manual deletion) — n8n won't hand back the value of a
    # pre-existing "private-ai-agent" key either, so clean up any such stale
    # entry first. Leaving it in place would make the create call below
    # 500 with "There is already an entry with this name."
    existing_keys = session.get(f"{base_url}/rest/api-keys", timeout=15)
    if existing_keys.ok:
        keys_data = (existing_keys.json() or {}).get("data")
        entries = keys_data if isinstance(keys_data, list) else (
            [keys_data] if isinstance(keys_data, dict) and keys_data else []
        )
        for entry in entries:
            if entry.get("label") == "private-ai-agent" and entry.get("id"):
                session.delete(f"{base_url}/rest/api-keys/{entry['id']}", timeout=15)
    elif existing_keys.status_code != 401:
        print(
            f"n8n api-keys list returned {existing_keys.status_code}: "
            f"{existing_keys.text[:500]}"
        )

    # This n8n version requires an explicit `scopes` array (a "scoped API
    # keys" feature) — its create-key modal offers an "All" preset that
    # resolves to every scope the instance supports. Fetch that same list
    # rather than hardcoding scope strings that could drift across versions.
    # NOTE: `/rest/api-keys/scopes` is a best guess at the endpoint the UI
    # itself must call to populate its scope checkboxes — if this 404s, the
    # error below will show the real response so the correct path/shape can
    # be swapped in.
    scopes_response = session.get(f"{base_url}/rest/api-keys/scopes", timeout=15)
    if not scopes_response.ok:
        raise RuntimeError(
            f"n8n api-key scopes lookup failed ({scopes_response.status_code}): "
            f"{scopes_response.text[:500]}"
        )
    scopes_body = scopes_response.json() or {}
    all_scopes = scopes_body.get("data") if isinstance(scopes_body, dict) else scopes_body
    if not isinstance(all_scopes, list) or not all_scopes:
        raise RuntimeError(
            f"Unexpected n8n api-key scopes response shape: {str(scopes_body)[:500]}"
        )

    # Unix seconds, matching the `exp` unit n8n's own issued JWT API keys
    # use. 1 year out — long enough to avoid re-provisioning on every
    # restart now that the value itself is cached to disk below; if this
    # instance enforces a shorter max, the error will say so.
    expires_at = int(time.time()) + 365 * 24 * 3600

    created = session.post(
        f"{base_url}/rest/api-keys",
        json={"label": "private-ai-agent", "scopes": all_scopes, "expiresAt": expires_at},
        timeout=15,
    )
    if not created.ok:
        cookie_names = ",".join(session.cookies.keys()) or "none"
        raise RuntimeError(
            f"n8n api-key creation failed ({created.status_code}): "
            f"{created.text[:500]} (session cookies present: {cookie_names})"
        )
    created_body = created.json() or {}
    created_data = created_body.get("data")
    if not isinstance(created_data, dict):
        created_data = created_body if isinstance(created_body, dict) else {}
    api_key = created_data.get("rawApiKey") or created_data.get("apiKey")

    if not api_key:
        raise RuntimeError("Could not obtain an n8n Public API key")

    # This is the only place the raw value is ever available — persist it
    # now, since n8n will never hand it back via a list/get call again.
    state["apiKey"] = api_key
    _n8n_write_state(state_path, state)

    _n8n_api_key_by_port[int(port)] = api_key
    return api_key


def _n8n_authed_request(port, container_name, method, path, **kwargs):
    """Call n8n's Public API with the cached API key, retrying once with a
    freshly provisioned key if the call comes back 401 — the cached key can
    go stale for reasons outside the agent's control (e.g. someone deletes
    it from n8n's own Settings > API page, or its expiry passes).
    """
    headers = kwargs.pop("headers", {})
    api_key = _n8n_ensure_api_key(port, container_name)
    response = requests.request(
        method,
        f"http://127.0.0.1:{port}{path}",
        headers={**headers, "X-N8N-API-KEY": api_key},
        **kwargs,
    )
    if response.status_code == 401:
        _n8n_forget_api_key(port, container_name)
        api_key = _n8n_ensure_api_key(port, container_name)
        response = requests.request(
            method,
            f"http://127.0.0.1:{port}{path}",
            headers={**headers, "X-N8N-API-KEY": api_key},
            **kwargs,
        )
    return response


def _raise_for_n8n_response(response, context):
    """Like response.raise_for_status(), but the error carries n8n's actual
    response body — plain raise_for_status() only gives a generic
    "400 Client Error" with no indication of what n8n actually rejected.
    """
    if not response.ok:
        raise RuntimeError(
            f"n8n {context} failed ({response.status_code}): {response.text[:1000]}"
        )


def handle_deploy_workflow(command_id, payload):
    workflow_id = payload.get("workflowId")
    container_name = payload.get("containerName")
    port = payload.get("port")
    workflow_json = payload.get("workflowJson")

    if not all([workflow_id, container_name, port, workflow_json]):
        raise ValueError("DEPLOY_WORKFLOW payload missing required fields")

    report_progress(command_id, "authenticate", "Authenticating with n8n", percent=20)

    report_progress(command_id, "import", "Importing workflow", percent=50)
    create_body = {
        "name": workflow_json.get("name") or f"workflow-{workflow_id[:8]}",
        "nodes": workflow_json.get("nodes", []),
        "connections": workflow_json.get("connections", {}),
        "settings": workflow_json.get("settings") or {},
    }
    create_response = _n8n_authed_request(
        port, container_name, "POST", "/api/v1/workflows",
        json=create_body,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    _raise_for_n8n_response(create_response, "workflow creation")
    n8n_workflow_id = create_response.json().get("id")
    if not n8n_workflow_id:
        raise RuntimeError("n8n did not return a workflow id")

    report_progress(command_id, "activate", "Activating workflow", percent=80)
    activate_response = _n8n_authed_request(
        port, container_name, "POST", f"/api/v1/workflows/{n8n_workflow_id}/activate",
        timeout=30,
    )
    _raise_for_n8n_response(activate_response, "workflow activation")

    complete_command(command_id, "COMPLETED", {"n8n_workflow_id": n8n_workflow_id})


def handle_remove_workflow(command_id, payload):
    container_name = payload.get("containerName")
    port = payload.get("port")
    n8n_workflow_id = payload.get("n8nWorkflowId")

    if not container_name or port is None:
        raise ValueError("REMOVE_WORKFLOW payload missing containerName or port")

    if n8n_workflow_id:
        report_progress(command_id, "delete", "Removing workflow from n8n", percent=50)
        delete_response = _n8n_authed_request(
            port, container_name, "DELETE", f"/api/v1/workflows/{n8n_workflow_id}",
            timeout=30,
        )
        if delete_response.status_code not in (200, 204, 404):
            _raise_for_n8n_response(delete_response, "workflow deletion")

    complete_command(command_id, "COMPLETED", {"action": "removed"})


def handle_power_command(command_id, command_type):
    action = "reboot" if command_type == "RESTART_MACHINE" else "poweroff"
    report_progress(
        command_id,
        "power",
        f"Issuing systemctl {action}",
        percent=50,
        log_line=f"sudo -n systemctl {action}"
    )
    # Best-effort complete before host dies (may not arrive)
    try:
        complete_command(
            command_id,
            "COMPLETED",
            {"action": action}
        )
    except Exception as error:
        print(f"Pre-power complete failed (ok if host reboots): {error}")

    result = subprocess.run(
        ["sudo", "-n", "systemctl", action],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sudo -n systemctl {action} failed: "
            f"{(result.stderr or result.stdout).strip()}. "
            "Ensure install.sh sudoers allows reboot/poweroff for the agent user."
        )


def handle_queued_command(command):
    global active_command_id

    command_id = command["commandId"]
    command_type = command["type"]
    payload = command.get("payload") or {}

    try:
        if command_type == "INSTALL_LLM":
            handle_install_llm(command_id, payload)
        elif command_type == "OFFLOAD_LLM":
            handle_offload_llm(command_id, payload)
        elif command_type == "ONLOAD_LLM":
            handle_onload_llm(command_id, payload)
        elif command_type == "DELETE_LLM":
            handle_delete_llm(command_id, payload)
        elif command_type == "INSTALL_APP":
            handle_install_app(command_id, payload)
        elif command_type == "OFFLOAD_APP":
            handle_offload_app(command_id, payload)
        elif command_type == "ONLOAD_APP":
            handle_onload_app(command_id, payload)
        elif command_type == "DELETE_APP":
            handle_delete_app(command_id, payload)
        elif command_type == "DEPLOY_WORKFLOW":
            handle_deploy_workflow(command_id, payload)
        elif command_type == "REMOVE_WORKFLOW":
            handle_remove_workflow(command_id, payload)
        elif command_type in ("RESTART_MACHINE", "SHUTDOWN_MACHINE"):
            handle_power_command(command_id, command_type)
        else:
            raise ValueError(f"Unsupported command: {command_type}")
    except InstallCancelled:
        # Best-effort cleanup of whatever got created before the cancel
        # signal was observed — a container name is the only thing every
        # install/onload payload has in common at this point.
        container_name = payload.get("containerName")
        if container_name:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
        print(f"{command_type} cancelled by user")
        complete_command(command_id, "CANCELLED", "Cancelled by user")
    except InstallError as error:
        print(f"{command_type} failed [{error.error_code}]: {error}")
        complete_command(
            command_id,
            "FAILED",
            str(error),
            error=str(error),
            error_code=error.error_code,
        )
    except Exception as error:
        # Plain str(error) is unreadable for exceptions whose message is
        # just their args (e.g. KeyError(0) stringifies to "0") — prefixing
        # the exception type keeps failures diagnosable from the stored
        # command/instance error field alone, without needing agent logs.
        description = f"{type(error).__name__}: {error}"
        print(f"{command_type} failed: {description}")
        complete_command(
            command_id,
            "FAILED",
            description,
            error=description
        )
    finally:
        with active_command_lock:
            if active_command_id == command_id:
                active_command_id = None


def complete_command(command_id, status, result, error=None, error_code=None):
    try:
        body = {
            "status": status,
            "result": result
        }
        if error is not None:
            body["error"] = error
        if error_code is not None:
            body["error_code"] = error_code

        response = requests.post(
            f"{CONTROL_PLANE_URL}/machines/{MACHINE_ID}/commands/{command_id}/complete",
            headers={
                "Authorization": f"Bearer {AGENT_TOKEN}",
                "Content-Type": "application/json"
            },
            json=body,
            timeout=10
        )

        response.raise_for_status()

        print(f"Command completion: {response.json()}")

    except Exception as err:
        print(f"Command completion error: {err}")


def websocket_url():
    parsed = urlparse(CONTROL_PLANE_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws/agent/{MACHINE_ID}"


def send_websocket_message(ws, message):
    payload = json.dumps(message)
    with websocket_send_lock:
        ws.send(payload)


def format_inference_http_error(error):
    if isinstance(error, requests.HTTPError) and error.response is not None:
        body = (error.response.text or "").strip()
        if len(body) > 2000:
            body = body[:2000] + "...[truncated]"
        return f"vLLM HTTP {error.response.status_code}: {body or str(error)}"
    return str(error)


def resolve_inference_target(payload):
    """Return (port, request_body) for a vLLM chat/completions call."""
    if isinstance(payload, dict) and "request" in payload and "port" in payload:
        port = payload.get("port")
        body = payload.get("request")
        if port is None or not isinstance(body, dict):
            raise ValueError("Inference payload must include port and request object")
        return int(port), body
    if isinstance(payload, dict):
        return 8000, payload
    raise ValueError("INFERENCE payload must be a JSON object")


def handle_websocket_inference(ws, request_id, payload):
    try:
        port, body = resolve_inference_target(payload)
        body = dict(body)
        body["model"] = resolve_vllm_model_id(port, body.get("model"))
        inference_url = f"http://127.0.0.1:{port}/v1/chat/completions"

        inference_response = requests.post(
            inference_url,
            json=body,
            timeout=300
        )
        inference_response.raise_for_status()
        result = inference_response.json()

        response = {
            "type": "INFERENCE_RESULT",
            "request_id": request_id,
            "success": True,
            "result": result
        }

        print(f"WebSocket inference completed: {request_id}")
    except Exception as error:
        response = {
            "type": "INFERENCE_RESULT",
            "request_id": request_id,
            "success": False,
            "error": format_inference_http_error(error)
        }

        print(f"WebSocket inference error: {request_id}: {response['error']}")

    try:
        send_websocket_message(ws, response)
    except Exception as error:
        print(f"WebSocket inference send error: {request_id}: {error}")


def handle_websocket_webhook_call(ws, request_id, payload):
    try:
        port = payload.get("port")
        method = payload.get("method")
        path = payload.get("path")
        headers = payload.get("headers") or {}
        body_b64 = payload.get("body")

        if port is None or not method or not path:
            raise ValueError("WEBHOOK_CALL payload missing port/method/path")

        body = base64.b64decode(body_b64) if body_b64 else None
        webhook_response = requests.request(
            method,
            f"http://127.0.0.1:{port}{path}",
            headers=headers,
            data=body,
            timeout=60
        )

        response_body = webhook_response.content
        response = {
            "type": "WEBHOOK_CALL_RESULT",
            "request_id": request_id,
            "success": True,
            "result": {
                "status": webhook_response.status_code,
                "headers": dict(webhook_response.headers),
                "body": base64.b64encode(response_body).decode("ascii") if response_body else None,
            }
        }

        print(f"WebSocket webhook call completed: {request_id}")
    except Exception as error:
        response = {
            "type": "WEBHOOK_CALL_RESULT",
            "request_id": request_id,
            "success": False,
            "error": str(error)
        }

        print(f"WebSocket webhook call error: {request_id}: {response['error']}")

    try:
        send_websocket_message(ws, response)
    except Exception as error:
        print(f"WebSocket webhook call send error: {request_id}: {error}")


def handle_websocket_credential_set(ws, request_id, payload):
    try:
        port = payload.get("port")
        name = payload.get("name")
        credential_type = payload.get("type")
        data = payload.get("data")

        if port is None or not name or not credential_type or not isinstance(data, dict):
            raise ValueError("CREDENTIAL_SET payload missing port/name/type/data")

        container_name = _container_name_by_port.get(int(port), f"n8n-port-{port}")

        create_response = _n8n_authed_request(
            port, container_name, "POST", "/api/v1/credentials",
            json={"name": name, "type": credential_type, "data": data},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        _raise_for_n8n_response(create_response, "credential creation")
        n8n_credential_id = create_response.json().get("id")

        response = {
            "type": "CREDENTIAL_SET_RESULT",
            "request_id": request_id,
            "success": True,
            "result": {"n8n_credential_id": n8n_credential_id}
        }

        print(f"WebSocket credential set completed: {request_id}")
    except Exception as error:
        # Deliberately never log `data` above — it contains the secret.
        response = {
            "type": "CREDENTIAL_SET_RESULT",
            "request_id": request_id,
            "success": False,
            "error": format_inference_http_error(error)
        }

        print(f"WebSocket credential set error: {request_id}: {response['error']}")

    try:
        send_websocket_message(ws, response)
    except Exception as error:
        print(f"WebSocket credential set send error: {request_id}: {error}")


def connect_websocket():
    url = websocket_url()
    reconnect_delay_seconds = 5

    def on_open(ws):
        print("WebSocket connected")

    def on_message(ws, message):
        try:
            data = json.loads(message)

            if data.get("type") == "PING":
                send_websocket_message(ws, {"type": "PONG"})
                return

            if data.get("type") == "INFERENCE":
                request_id = data.get("request_id")
                payload = data.get("payload")

                if not request_id:
                    print("WebSocket inference missing request_id")
                    return

                print(f"Received WebSocket inference: {request_id}")

                inference_thread = threading.Thread(
                    target=handle_websocket_inference,
                    args=(ws, request_id, payload),
                    daemon=True,
                    name=f"inference-{request_id}"
                )
                inference_thread.start()
                return

            if data.get("type") == "WEBHOOK_CALL":
                request_id = data.get("request_id")
                payload = data.get("payload")

                if not request_id:
                    print("WebSocket webhook call missing request_id")
                    return

                print(f"Received WebSocket webhook call: {request_id}")

                webhook_thread = threading.Thread(
                    target=handle_websocket_webhook_call,
                    args=(ws, request_id, payload),
                    daemon=True,
                    name=f"webhook-{request_id}"
                )
                webhook_thread.start()
                return

            if data.get("type") == "CREDENTIAL_SET":
                request_id = data.get("request_id")
                payload = data.get("payload")

                if not request_id:
                    print("WebSocket credential set missing request_id")
                    return

                print(f"Received WebSocket credential set: {request_id}")

                credential_thread = threading.Thread(
                    target=handle_websocket_credential_set,
                    args=(ws, request_id, payload),
                    daemon=True,
                    name=f"credential-{request_id}"
                )
                credential_thread.start()
        except json.JSONDecodeError:
            print(f"WebSocket invalid message: {message}")
        except Exception as error:
            print(f"WebSocket message handler error: {error}")

    def on_error(ws, error):
        print(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"WebSocket disconnected: {close_status_code} {close_msg}")

    while True:
        try:
            ws = websocket.WebSocketApp(
                url,
                header=[f"Authorization: Bearer {AGENT_TOKEN}"],
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
        except Exception as error:
            print(f"WebSocket connection failed: {error}")

        print(f"WebSocket reconnecting in {reconnect_delay_seconds} seconds")
        time.sleep(reconnect_delay_seconds)


websocket_thread = threading.Thread(
    target=connect_websocket,
    daemon=True
)
websocket_thread.start()

while True:
    send_heartbeat()
    poll_commands()
    time.sleep(5)
