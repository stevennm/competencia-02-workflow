"""
Bucket utilities for copying outputs to VM bucket storage
Simple version: just copy the experiment directory
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Optional
import os


def sync_experiment_to_bucket(experimento: str, config: Dict,
                              logger: Optional[logging.Logger] = None) -> bool:
    """
    Simple copy: cp -r output/{experimento} ~/bucket_path/
    
    Args:
        experimento: Experiment name
        config: Configuration dictionary
        logger: Optional logger
        
    Returns:
        True if copied successfully
    """
    if not config.get("bucket", {}).get("enabled", False):
        if logger:
            logger.info("Bucket sync disabled")
        return False
    
    # Source
    source_dir = Path(f"output/{experimento}")
    if not source_dir.exists():
        if logger:
            logger.warning(f"Experiment output not found: {source_dir}")
        return False
    
    # Destination: ~/bucket_path/experimento
    bucket_base = os.path.expanduser(config["bucket"]["base_path"])
    dest_dir = Path(bucket_base) / experimento
    
    try:
        # Remove old if exists
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        
        # Copy
        shutil.copytree(source_dir, dest_dir)
        
        if logger:
            logger.info(f"✓ Copied to bucket: {source_dir} → {dest_dir}")
        else:
            print(f"✓ Copied to bucket: {source_dir} → {dest_dir}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.warning(f"Could not copy to bucket: {e}")
        else:
            print(f"⚠ Could not copy to bucket: {e}")
        return False


def sync_output_to_bucket(config: Dict, logger: Optional[logging.Logger] = None):
    """
    Simple sync: cp -r output/{experimento} ~/bucket_path/
    
    Args:
        config: Configuration dictionary
        logger: Optional logger
    """
    if not config.get("bucket", {}).get("enabled", False):
        if logger:
            logger.info("Bucket sync disabled")
        return
    
    experimento = config["experimento"]
    
    if logger:
        logger.info("="*70)
        logger.info("SYNCING TO BUCKET")
        logger.info("="*70)
    else:
        print("\n" + "="*70)
        print("SYNCING TO BUCKET")
        print("="*70)
    
    # Just copy the experiment directory
    sync_experiment_to_bucket(experimento, config, logger)
    
    if logger:
        logger.info("="*70)
        logger.info("BUCKET SYNC COMPLETE")
        logger.info("="*70)
    else:
        print("="*70)
        print("BUCKET SYNC COMPLETE")
        print("="*70)


def get_bucket_info(config: Dict) -> Dict:
    """
    Get information about bucket configuration
    
    Returns:
        Dictionary with bucket info
    """
    bucket_config = config.get("bucket", {})
    
    if not bucket_config.get("enabled", False):
        return {"enabled": False}
    
    base_path = os.path.expanduser(bucket_config["base_path"])
    
    return {
        "enabled": True,
        "base_path": base_path,
        "exists": Path(base_path).exists(),
    }

