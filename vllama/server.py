"""FastAPI server for request forwarding and model aggregation."""

import asyncio
import json
import logging
import os
import signal
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from vllama.config import VllamaConfig
from vllama.constants import VllamaPaths
from vllama.gpu import get_gpu_monitor
from vllama.instance import VLLMInstanceManager
from vllama.models import ModelInfo, find_model_by_name, get_model_list_for_api, scan_transformers_cache
from vllama.scheduler import UnloadScheduler
from vllama.state import InstanceStatus, StateManager
from vllama.yaml_manager import YAMLConfigManager

logger = logging.getLogger(__name__)


class VllamaServer:
    """Main vllama server handling request routing."""

    def __init__(self, config: VllamaConfig):
        """Initialize vllama server.

        Args:
            config: Global vllama configuration
        """
        self.config = config
        self.app = FastAPI(title="vllama", version="0.1.0")

        # Ensure directories exist
        VllamaPaths.ensure_directories()

        # Initialize managers
        self.state_manager = StateManager()
        self.yaml_manager = YAMLConfigManager(VllamaPaths.MODELS_CONFIG_FILE)
        self.instance_manager = VLLMInstanceManager(
            config, self.state_manager, self.yaml_manager
        )
        self.scheduler = UnloadScheduler(config, self.instance_manager)

        # Scan available models
        self.models: list[ModelInfo] = scan_transformers_cache(config.transformers_cache)
        logger.info(f"Found {len(self.models)} models in cache")

        # GPU monitor for memory info
        self.gpu_monitor = get_gpu_monitor()

        # Setup routes
        self._setup_routes()

        # HTTP client for forwarding
        self._http_client: Optional[httpx.AsyncClient] = None

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _get_pid_file_path(self) -> str:
        """Get the path to the PID file.

        Returns:
            Path to the PID file
        """
        return str(VllamaPaths.PID_FILE)

    def _write_pid_file(self):
        """Write server information to PID file."""
        pid_file = self._get_pid_file_path()
        pid_data = {
            "pid": os.getpid(),
            "port": self.config.port,
            "host": self.config.host,
            "start_time": datetime.now().timestamp(),
            "config_dir": str(VllamaPaths.CONFIG_DIR),
            "models_config": str(VllamaPaths.MODELS_CONFIG_FILE),
            "transformers_cache": str(self.config.transformers_cache),
            "vllm_port_range": [self.config.vllm_port_start, self.config.vllm_port_end],
            "unload_mode": self.config.unload_mode,
            "unload_timeout": self.config.unload_timeout,
        }

        with open(pid_file, "w") as f:
            json.dump(pid_data, f, indent=2)

        logger.info(f"Server PID file written to: {pid_file}")

    def _remove_pid_file(self):
        """Remove the PID file if it exists."""
        pid_file = self._get_pid_file_path()
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
                logger.info(f"Removed PID file: {pid_file}")
        except Exception as e:
            logger.warning(f"Failed to remove PID file: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            """Handle termination signals."""
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating graceful shutdown...")
            # The actual cleanup will be done in the shutdown event handler
            # Just log here for visibility
            raise KeyboardInterrupt

        # Register handlers for SIGTERM and SIGINT
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        logger.info("Signal handlers registered for graceful shutdown")

    async def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=300.0)
        return self._http_client

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.on_event("startup")
        async def startup():
            """Startup event handler."""
            # Write PID file
            self._write_pid_file()

            # Start scheduler
            await self.scheduler.start()
            logger.info("Vllama server started")

        @self.app.on_event("shutdown")
        async def shutdown():
            """Shutdown event handler."""
            logger.info("Shutting down vllama server...")

            # Stop scheduler
            await self.scheduler.stop()

            # Cleanup all vLLM instances
            await self.instance_manager.cleanup_all_instances()

            # Close HTTP client
            if self._http_client:
                await self._http_client.aclose()

            # Remove PID file
            self._remove_pid_file()

            logger.info("Vllama server shutdown complete")

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "ok"}

        @self.app.get("/instances")
        async def list_instances():
            """List all running model instances."""
            instances = self.instance_manager.get_all_instances()

            # Convert instances to API-friendly format
            instances_data = []
            for model_id, instance in instances.items():
                # Get GPU memory usage for this instance
                gpu_memory_used = None
                gpu_memory_total = None
                if instance.devices and self.gpu_monitor.get_device_count() > 0:
                    # Get memory info for the first device
                    device_id = instance.devices[0]
                    try:
                        mem_info = self.gpu_monitor.get_memory_info(device_id)
                        gpu_memory_used = mem_info.get("used", 0)
                        gpu_memory_total = mem_info.get("total", 0)
                    except Exception:
                        pass

                instance_dict = {
                    "model_id": model_id,
                    "status": instance.status.value,
                    "port": instance.port,
                    "pid": instance.pid,
                    "devices": instance.devices or [],
                    "start_time": instance.start_time,
                    "last_request_time": instance.last_request_time,
                    "sleep_level": instance.sleep_level,
                    "memory_delta": instance.memory_delta,
                    "gpu_memory_used": gpu_memory_used,
                    "gpu_memory_total": gpu_memory_total,
                }
                instances_data.append(instance_dict)

            return {"instances": instances_data}

        @self.app.post("/instances/{model_name:path}/start")
        async def start_model(model_name: str):
            """Start or wake up a model instance."""
            # Find model
            model_info = find_model_by_name(self.models, model_name)

            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            try:
                # Use ensure_instance_running to handle all cases (start, wake, recover)
                result_instance = await self.instance_manager.ensure_instance_running(model_info)

                return {
                    "status": "success",
                    "model_id": model_info.model_id,
                    "port": result_instance.port,
                    "pid": result_instance.pid,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/instances/{model_name:path}/sleep")
        async def sleep_model(model_name: str, level: int = 2):
            """Put a model instance to sleep."""
            # Find model
            model_info = find_model_by_name(self.models, model_name)

            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            try:
                success = await self.instance_manager.sleep_instance(model_info.model_id, level=level)
                if success:
                    return {"status": "success", "model_id": model_info.model_id, "level": level}
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to sleep {model_info.model_id}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/instances/{model_name:path}/stop")
        async def stop_model(model_name: str):
            """Stop a model instance."""
            # Find model
            model_info = find_model_by_name(self.models, model_name)

            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            try:
                success = await self.instance_manager.stop_instance(model_info.model_id)
                if success:
                    return {"status": "success", "model_id": model_info.model_id}
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to stop {model_info.model_id}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v1/models")
        async def list_models():
            """List all available models."""
            model_list = get_model_list_for_api(self.models)

            # Update model availability based on sleep status
            instances = self.instance_manager.get_all_instances()
            for model_dict in model_list:
                model_id = model_dict["id"]
                if model_id in instances:
                    instance = instances[model_id]
                    # Mark sleeping models in the response if unload_mode is sleep
                    if instance.status in (InstanceStatus.SLEEPING_L1, InstanceStatus.SLEEPING_L2):
                        if self.config.unload_mode in (1, 2):
                            model_dict["status"] = instance.status.value

            return {"object": "list", "data": model_list}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """Forward chat completion requests."""
            return await self._forward_request(request, "/v1/chat/completions")

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            """Forward completion requests."""
            return await self._forward_request(request, "/v1/completions")

        @self.app.post("/v1/embeddings")
        async def embeddings(request: Request):
            """Forward embedding requests."""
            return await self._forward_request(request, "/v1/embeddings")

        @self.app.post("/rerank")
        @self.app.post("/v1/rerank")
        async def rerank(request: Request):
            """Forward rerank requests."""
            path = "/rerank" if request.url.path == "/rerank" else "/v1/rerank"
            return await self._forward_request(request, path)

        @self.app.post("/score")
        @self.app.post("/v1/score")
        async def score(request: Request):
            """Forward score requests."""
            path = "/score" if request.url.path == "/score" else "/v1/score"
            return await self._forward_request(request, path)

        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def catch_all(request: Request, path: str):
            """Catch-all route for other requests."""
            return await self._forward_request(request, f"/{path}")

    async def _forward_request(self, request: Request, path: str):
        """Forward request to appropriate VLLM instance.

        Args:
            request: FastAPI request object
            path: Request path

        Returns:
            Response from VLLM instance
        """
        try:
            # Parse request body
            body = await request.json()
        except Exception:
            body = {}

        # Extract model name from request
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name required in request body")

        # Find model
        model_info = find_model_by_name(self.models, model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        try:
            # Ensure instance is running
            instance = await self.instance_manager.ensure_instance_running(model_info)

            # Forward request
            client = await self.get_http_client()
            target_url = f"http://localhost:{instance.port}{path}"

            # Check if streaming
            is_streaming = body.get("stream", False)

            # Filter headers to avoid conflicts
            # Exclude headers that should be auto-generated or may cause issues
            excluded_headers = {
                "host",
                "content-length",
                "content-type",  # Will be set by json parameter
                "transfer-encoding",
                "connection",
            }
            forwarded_headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in excluded_headers
            }

            logger.debug(f"Forwarding request to {target_url}")

            if is_streaming:
                # Handle streaming response
                async def stream_generator():
                    async with client.stream(
                        method=request.method,
                        url=target_url,
                        json=body,
                        headers=forwarded_headers,
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                )
            else:
                # Handle regular response
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    json=body,
                    headers=forwarded_headers,
                )

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

        except Exception as e:
            logger.error(f"Error forwarding request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    def get_app(self) -> FastAPI:
        """Get FastAPI app instance.

        Returns:
            FastAPI application
        """
        return self.app


def create_app(config: Optional[VllamaConfig] = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Optional configuration, will use default if not provided

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = VllamaConfig()

    server = VllamaServer(config)
    return server.get_app()
