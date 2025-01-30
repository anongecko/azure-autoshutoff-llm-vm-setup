import os
import torch
import json
from pathlib import Path
from safetensors.torch import save_file, load_file
import logging
from tqdm import tqdm
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectWeightMerger:
    def __init__(self, model_path: str, output_path: str):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        
        # Verify paths
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
            
        # Get weight files
        self.weight_files = sorted(
            self.model_path.glob("model-*.safetensors"),
            key=lambda x: int(x.stem.split("-")[1])
        )
        
        if not self.weight_files:
            raise ValueError(f"No weight files found in {model_path}")
        
        # Verify config exists
        self.config_path = self.model_path / "config.json"
        if not self.config_path.exists():
            raise ValueError(f"Config file not found: {self.config_path}")

    def _load_and_verify_shard(self, shard_path: Path):
        """Load a single shard and verify its integrity."""
        try:
            weights = load_file(shard_path)
            
            # Basic integrity checks
            for name, tensor in weights.items():
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN values found in tensor {name}")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf values found in tensor {name}")
                
            return weights
        except Exception as e:
            logger.error(f"Error loading shard {shard_path}: {e}")
            raise

    def merge(self):
        """Merge weights directly using safetensors."""
        try:
            logger.info("Starting direct weight merge process...")
            
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Copy config and other necessary files
            logger.info("Copying configuration files...")
            shutil.copy2(self.config_path, self.output_path / "config.json")
            if (self.model_path / "generation_config.json").exists():
                shutil.copy2(
                    self.model_path / "generation_config.json",
                    self.output_path / "generation_config.json"
                )
            if (self.model_path / "tokenizer_config.json").exists():
                shutil.copy2(
                    self.model_path / "tokenizer_config.json",
                    self.output_path / "tokenizer_config.json"
                )
            if (self.model_path / "tokenizer.model").exists():
                shutil.copy2(
                    self.model_path / "tokenizer.model",
                    self.output_path / "tokenizer.model"
                )

            # Initialize merged weights dictionary
            merged_weights = {}
            total_size = 0
            
            # Load and merge each shard
            for shard_file in tqdm(self.weight_files, desc="Merging weight files"):
                logger.info(f"Processing {shard_file.name}")
                shard_weights = self._load_and_verify_shard(shard_file)
                
                # Merge weights
                for key, tensor in shard_weights.items():
                    if key in merged_weights:
                        logger.warning(f"Duplicate key found: {key}")
                        # Verify tensors match if duplicated
                        if not torch.equal(merged_weights[key], tensor):
                            raise ValueError(f"Conflicting values for duplicate key: {key}")
                    else:
                        merged_weights[key] = tensor
                        total_size += tensor.numel() * tensor.element_size()

            logger.info(f"Total merged model size: {total_size / 1024**3:.2f} GB")
            
            # Save merged weights
            logger.info("Saving merged weights...")
            output_file = self.output_path / "pytorch_model.safetensors"
            save_file(
                merged_weights,
                output_file,
                metadata={
                    "format": "pt",
                    "framework": "pytorch"
                }
            )

            logger.info("Verifying saved weights...")
            # Verify saved file
            loaded_weights = load_file(output_file)
            if len(loaded_weights) != len(merged_weights):
                raise ValueError("Verification failed: number of tensors mismatch")
                
            for key in merged_weights:
                if key not in loaded_weights:
                    raise ValueError(f"Verification failed: missing key {key}")
                if not torch.equal(merged_weights[key], loaded_weights[key]):
                    raise ValueError(f"Verification failed: tensor mismatch for {key}")

            logger.info("Weight merge completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Merge process failed: {str(e)}")
            return False

def main():
    MODEL_PATH = "/data/models/weights"  # Update with your model path
    OUTPUT_PATH = "/data/models/merged"  # Update with your output path
    
    merger = DirectWeightMerger(MODEL_PATH, OUTPUT_PATH)
    success = merger.merge()
    
    if success:
        logger.info("Model weights merged successfully!")
    else:
        logger.error("Model weight merge failed!")

if __name__ == "__main__":
    main()
