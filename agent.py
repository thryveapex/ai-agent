import json
import os
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

import psutil
import requests
import websocket


DEFAULT_CONTROL_PLANE_URL = "http://192.168.1.3:3000"
VLLM_CHAT_COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
CREDENTIALS_PATH = os.environ.get(
    "AI_NODE_CREDENTIALS_PATH",
    "/opt/ai-node/credentials.json"
)


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
    except Exception as error:
        print(f"Progress report error: {error}")


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


def docker_pull_with_progress(command_id, container_image, timeout=3600):
    """Pull image while streaming layer progress to control plane."""
    command = ["docker", "pull", container_image]
    print("Running:", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    last_report = 0.0
    last_line = ""
    deadline = time.time() + timeout

    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line)
            last_line = line

            now = time.time()
            if now - last_report >= 5 or "Pull complete" in line or "Status:" in line:
                # Map pull sub-steps into 25–65% of overall install progress.
                percent = 25
                lower = line.lower()
                if "downloading" in lower:
                    percent = 35
                elif "extracting" in lower:
                    percent = 50
                elif "pull complete" in lower or "digest:" in lower:
                    percent = 60

                report_progress(
                    command_id,
                    "pull_image",
                    line[:500],
                    percent=percent,
                    log_line=line[:500]
                )
                last_report = now

            if now > deadline:
                process.kill()
                raise TimeoutError(
                    f"docker pull timed out after {timeout}s: {container_image}"
                )

        return_code = process.wait(timeout=30)
    except Exception:
        process.kill()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"docker pull failed ({return_code}): {container_image}\n{last_line}"
        )


def docker_available():
    run_subprocess(["docker", "info"], timeout=30)


def handle_install_llm(command_id, payload):
    container_name = payload.get("containerName")
    container_image = payload.get("containerImage")
    model_path = payload.get("modelPath")
    port = payload.get("port")

    if not all([container_name, container_image, model_path, port]):
        raise ValueError("INSTALL_LLM payload missing required fields")

    report_progress(command_id, "validate", "Checking Docker", percent=5)
    docker_available()

    report_progress(
        command_id,
        "pull_image",
        f"Pulling {container_image} (this can take 20–40 min on first run)",
        percent=25,
        log_line=f"docker pull {container_image}"
    )
    docker_pull_with_progress(command_id, container_image, timeout=3600)

    # Remove any leftover container with the same name.
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
    run_subprocess([
        "docker", "run", "-d",
        "--gpus", "all",
        "-p", f"{port}:8000",
        "--name", container_name,
        "--restart", "unless-stopped",
        container_image,
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", "0.30",
        "--max-model-len", "2048",
    ], timeout=120)

    wait_for_model_health(command_id, port)
    complete_command(
        command_id,
        "COMPLETED",
        {
            "port": port,
            "container_name": container_name,
            "model_path": model_path
        }
    )


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
    if not container_name:
        raise ValueError("DELETE_LLM payload missing containerName")

    report_progress(
        command_id,
        "stop_container",
        f"Stopping {container_name}",
        percent=30,
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
        percent=70,
        log_line=f"docker rm -f {container_name}"
    )
    result = subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "no such container" not in stderr:
            raise RuntimeError(
                f"docker rm failed: {(result.stderr or result.stdout).strip()}"
            )

    complete_command(
        command_id,
        "COMPLETED",
        {"container_name": container_name, "action": "deleted"}
    )


def wait_for_model_health(command_id, port, timeout_seconds=300):
    report_progress(
        command_id,
        "health_check",
        f"Waiting for http://127.0.0.1:{port}/v1/models",
        percent=90
    )
    deadline = time.time() + timeout_seconds
    last_error = None
    last_health_report = 0.0
    while time.time() < deadline:
        try:
            response = requests.get(
                f"http://127.0.0.1:{port}/v1/models",
                timeout=5
            )
            if response.status_code == 200:
                report_progress(
                    command_id,
                    "health_check",
                    "Model server is healthy",
                    percent=95,
                    log_line="health check ok"
                )
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as error:
            last_error = str(error)

        now = time.time()
        if now - last_health_report >= 10:
            report_progress(
                command_id,
                "health_check",
                f"Waiting for model server… ({last_error or 'starting'})",
                percent=90
            )
            last_health_report = now
        time.sleep(5)

    raise TimeoutError(
        f"Model server did not become healthy: {last_error}"
    )


def docker_run_vllm(command_id, payload):
    container_name = payload.get("containerName")
    container_image = payload.get("containerImage")
    model_path = payload.get("modelPath")
    port = payload.get("port")

    if not all([container_name, container_image, model_path, port]):
        raise ValueError("Missing fields for docker run (container/image/model/port)")

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
    run_subprocess([
        "docker", "run", "-d",
        "--gpus", "all",
        "-p", f"{port}:8000",
        "--name", container_name,
        "--restart", "unless-stopped",
        container_image,
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", "0.30",
        "--max-model-len", "2048",
    ], timeout=120)


def handle_onload_llm(command_id, payload):
    container_name = payload.get("containerName")
    port = payload.get("port")
    if not container_name or port is None:
        raise ValueError("ONLOAD_LLM payload missing containerName or port")

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
            report_progress(
                command_id,
                "start_container",
                "Container missing; recreating with docker run",
                percent=50,
                log_line="fallback docker run"
            )
            docker_run_vllm(command_id, payload)
        else:
            raise RuntimeError(
                f"docker start failed: {(result.stderr or result.stdout).strip()}"
            )

    wait_for_model_health(command_id, port)
    complete_command(
        command_id,
        "COMPLETED",
        {
            "port": port,
            "container_name": container_name,
            "action": "onloaded"
        }
    )


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
        elif command_type in ("RESTART_MACHINE", "SHUTDOWN_MACHINE"):
            handle_power_command(command_id, command_type)
        else:
            raise ValueError(f"Unsupported command: {command_type}")
    except Exception as error:
        print(f"{command_type} failed: {error}")
        complete_command(
            command_id,
            "FAILED",
            str(error),
            error=str(error)
        )
    finally:
        with active_command_lock:
            if active_command_id == command_id:
                active_command_id = None


def complete_command(command_id, status, result, error=None):
    try:
        body = {
            "status": status,
            "result": result
        }
        if error is not None:
            body["error"] = error

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


def handle_websocket_inference(ws, request_id, payload):
    try:
        if not isinstance(payload, dict):
            raise ValueError("INFERENCE payload must be a JSON object")

        inference_response = requests.post(
            VLLM_CHAT_COMPLETIONS_URL,
            json=payload,
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
