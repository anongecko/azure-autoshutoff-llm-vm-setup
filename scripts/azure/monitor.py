import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import aiohttp
import json
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
import os
import sys
from pathlib import Path
import psutil
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureVMMonitor:
    def __init__(
        self,
        resource_group: str,
        vm_name: str,
        subscription_id: str,
        idle_timeout: int = 900,  # 15 minutes in seconds
        check_interval: int = 60,  # Check every minute
        api_port: int = 8000
    ):
        self.resource_group = resource_group
        self.vm_name = vm_name
        self.subscription_id = subscription_id
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.api_port = api_port
        self.last_activity = datetime.now()
        self.credential = DefaultAzureCredential()
        self.compute_client = ComputeManagementClient(
            credential=self.credential,
            subscription_id=self.subscription_id
        )
        
    async def check_api_activity(self) -> bool:
        """Check if API has recent activity"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{self.api_port}/health"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("active_requests", 0) > 0
        except Exception as e:
            logger.error(f"Error checking API activity: {e}")
        return False

    async def check_gpu_activity(self) -> bool:
        """Check if GPU has significant activity"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                return gpu_memory > 1 * (1024 ** 3)  # More than 1GB in use
        except Exception as e:
            logger.error(f"Error checking GPU activity: {e}")
        return False

    def deallocate_vm(self):
        """Deallocate the VM"""
        try:
            logger.info(f"Deallocating VM {self.vm_name}")
            self.compute_client.virtual_machines.begin_deallocate(
                self.resource_group,
                self.vm_name
            ).wait()
            logger.info("VM deallocated successfully")
        except Exception as e:
            logger.error(f"Error deallocating VM: {e}")

    async def monitor_activity(self):
        """Monitor VM activity and deallocate if idle"""
        while True:
            try:
                api_active = await self.check_api_activity()
                gpu_active = await self.check_gpu_activity()
                
                if api_active or gpu_active:
                    self.last_activity = datetime.now()
                    logger.debug("Activity detected, resetting timer")
                else:
                    idle_duration = datetime.now() - self.last_activity
                    if idle_duration.total_seconds() >= self.idle_timeout:
                        logger.info(f"VM idle for {idle_duration}, initiating shutdown")
                        self.deallocate_vm()
                        break
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(self.check_interval)

async def main():
    """Main execution function"""
    required_env_vars = [
        "AZURE_RESOURCE_GROUP",
        "AZURE_VM_NAME",
        "AZURE_SUBSCRIPTION_ID"
    ]
    
    # Verify environment variables
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    monitor = AzureVMMonitor(
        resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
        vm_name=os.getenv("AZURE_VM_NAME"),
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        idle_timeout=int(os.getenv("IDLE_TIMEOUT_SECONDS", "900")),
        check_interval=int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))
    )
    
    await monitor.monitor_activity()

if __name__ == "__main__":
    asyncio.run(main())

