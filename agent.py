import json
import subprocess
import threading
import time
from urllib.parse import urlparse

import requests
import websocket


CONTROL_PLANE_URL = "http://192.168.1.3:3000"
VLLM_CHAT_COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"

MACHINE_ID = "4b660795-20c8-4daf-bdde-6e06d293594d"
AGENT_TOKEN = "65404611-68bb-4bb7-bf84-1a5d4b32f081"

websocket_send_lock = threading.Lock()


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
        memory_used = int(parts[3])
        memory_total = int(parts[4])
        power_draw = float(parts[5])

        gpus.append({
            "name": name,
            "temperature": temperature,
            "utilization": utilization,
            "memory_used": memory_used,
            "memory_total": memory_total,
            "power_draw": power_draw
        })

    return gpus


def send_heartbeat():
    gpus = get_gpu_stats()

    response = requests.post(
        f"{CONTROL_PLANE_URL}/machines/heartbeat",
        headers={
            "Authorization": f"Bearer {AGENT_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "machine_id": MACHINE_ID,
            "gpus": gpus
        },
        timeout=5
    )

    response.raise_for_status()

    print("Heartbeat:", response.json())
    print("GPU:", gpus)

def poll_commands():
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

        print(f"Received command: {command['type']}")

        if command["type"] == "PING":
            result = "PONG"

            print(f"Command result: {result}")

            complete_command(
                command["commandId"],
                "COMPLETED",
                result
            )

        if command["type"] == "INFERENCE":
            try:
                inference_response = requests.post(
                    VLLM_CHAT_COMPLETIONS_URL,
                    json=command["payload"],
                    timeout=300
                )
                inference_response.raise_for_status()
                result = inference_response.json()

                print(f"Command result: {result}")

                complete_command(
                    command["commandId"],
                    "COMPLETED",
                    result
                )
            except Exception as error:
                print(f"Inference error: {error}")
                complete_command(
                    command["commandId"],
                    "FAILED",
                    str(error)
                )
    except Exception as error:
        print(f"Command polling error: {error}")

def complete_command(command_id, status, result):
    try:
        response = requests.post(
            f"{CONTROL_PLANE_URL}/machines/{MACHINE_ID}/commands/{command_id}/complete",
            headers={
                "Authorization": f"Bearer {AGENT_TOKEN}"
            },
            json={
                "status": status,
                "result": result
            },
            timeout=10
        )

        response.raise_for_status()

        print(f"Command completion: {response.json()}")

    except Exception as error:
        print(f"Command completion error: {error}")


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
