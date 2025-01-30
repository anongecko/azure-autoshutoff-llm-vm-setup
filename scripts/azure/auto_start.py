import azure.functions as func
import logging
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
import asyncio
import aiohttp
import time
from typing import Optional
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VMManager:
    def __init__(
        self,
        resource_group: str,
        vm_name: str,
        subscription_id: str,
        api_port: int = 8000,
        startup_timeout: int = 300  # 5 minutes
    ):
        self.resource_group = resource_group
        self.vm_name = vm_name
        self.subscription_id = subscription_id
        self.api_port = api_port
        self.startup_timeout = startup_timeout
        self.credential = DefaultAzureCredential()
        self.compute_client = ComputeManagementClient(
            credential=self.credential,
            subscription_id=self.subscription_id
        )

    async def ensure_vm_running(self) -> bool:
        """Ensure VM is running, start if needed"""
        try:
            vm = self.compute_client.virtual_machines.get(
                self.resource_group,
                self.vm_name
            )
            
            if vm.instance_view.statuses[-1].code != "PowerState/running":
                logger.info(f"Starting VM {self.vm_name}")
                self.compute_client.virtual_machines.begin_start(
                    self.resource_group,
                    self.vm_name
                ).wait()
                
                return await self.wait_for_api_ready()
            return True
            
        except Exception as e:
            logger.error(f"Error managing VM: {e}")
            return False

    async def wait_for_api_ready(self) -> bool:
        """Wait for API to become ready"""
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{self.api_port}/health"
                    ) as response:
                        if response.status == 200:
                            return True
            except:
                pass
            await asyncio.sleep(5)
        return False

async def handle_request(req: func.HttpRequest) -> func.HttpResponse:
    """Handle incoming request and ensure VM is running"""
    try:
        # Validate required environment variables
        required_vars = [
            "AZURE_RESOURCE_GROUP",
            "AZURE_VM_NAME",
            "AZURE_SUBSCRIPTION_ID"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            return func.HttpResponse(
                json.dumps({
                    "error": f"Missing environment variables: {missing_vars}"
                }),
                status_code=500
            )

        vm_manager = VMManager(
            resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
            vm_name=os.getenv("AZURE_VM_NAME"),
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID")
        )

        # Ensure VM is running
        vm_ready = await vm_manager.ensure_vm_running()
        if not vm_ready:
            return func.HttpResponse(
                json.dumps({"error": "Failed to start VM or API not ready"}),
                status_code=500
            )

        # Forward the original request to the API
        original_url = req.url.replace(
            "azurewebsites.net",
            f"localhost:{vm_manager.api_port}"
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=req.method,
                url=original_url,
                headers=dict(req.headers),
                data=req.get_body()
            ) as response:
                return func.HttpResponse(
                    body=await response.read(),
                    status_code=response.status,
                    headers=dict(response.headers)
                )

    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500
        )

