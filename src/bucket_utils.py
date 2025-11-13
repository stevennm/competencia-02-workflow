"""
Bucket utilities for copying outputs to VM bucket storage
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Optional
import os


def get_bucket_path(config: Dict, relative_path: str = "") -> Optional[Path]:
    """
    Get the full bucket path for a given relative path
    
    Args:
        config: Configuration dictionary
        relative_path: Relative path within bucket (e.g., "exp/experimento_name")
        
    Returns:
        Full bucket path or None if bucket is disabled
    """
    if not config.get("bucket", {}).get("enabled", False):
        return None
    
    base_path = config["bucket"]["base_path"]
    # Expand ~ to home directory
    base_path = os.path.expanduser(base_path)
    
    if relative_path:
        return Path(base_path) / relative_path
    return Path(base_path)


def copy_to_bucket(source_path: str, config: Dict, 
                   bucket_subdir: str = "", 
                   logger: Optional[logging.Logger] = None) -> bool:
    """
    Copy a file or directory to the bucket
    
    Args:
        source_path: Source file or directory path
        config: Configuration dictionary
        bucket_subdir: Subdirectory within bucket (e.g., "exp/experimento_name")
        logger: Optional logger for messages
        
    Returns:
        True if copied successfully, False otherwise
    """
    if not config.get("bucket", {}).get("enabled", False):
        return False
    
    source = Path(source_path)
    
    if not source.exists():
        if logger:
            logger.warning(f"Source does not exist, skipping bucket copy: {source}")
        return False
    
    # Get bucket destination
    bucket_base = get_bucket_path(config, bucket_subdir)
    if bucket_base is None:
        return False
    
    # Create destination directory
    bucket_base.mkdir(parents=True, exist_ok=True)
    
    # Determine destination path
    if source.is_file():
        dest = bucket_base / source.name
    else:
        dest = bucket_base / source.name
    
    try:
        if source.is_file():
            # Copy file
            shutil.copy2(source, dest)
            if logger:
                logger.info(f"✓ Copied to bucket: {source.name} → {dest}")
            else:
                print(f"✓ Copied to bucket: {source.name} → {dest}")
        else:
            # Copy directory
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
            if logger:
                logger.info(f"✓ Copied to bucket: {source.name}/ → {dest}/")
            else:
                print(f"✓ Copied to bucket: {source.name}/ → {dest}/")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to copy to bucket: {source} → {dest}: {e}")
        else:
            print(f"⚠ Failed to copy to bucket: {source} → {dest}: {e}")
        return False


def copy_experiment_to_bucket(experimento: str, config: Dict,
                              logger: Optional[logging.Logger] = None) -> bool:
    """
    Copy entire experiment output directory to bucket
    
    Args:
        experimento: Experiment name
        config: Configuration dictionary
        logger: Optional logger
        
    Returns:
        True if copied successfully
    """
    if not config.get("bucket", {}).get("enabled", False):
        return False
    
    source_dir = Path(f"output/{experimento}")
    
    if not source_dir.exists():
        if logger:
            logger.warning(f"Experiment output not found: {source_dir}")
        return False
    
    # Copy to bucket under exp/ subdirectory
    bucket_dest = get_bucket_path(config, "exp")
    if bucket_dest is None:
        return False
    
    bucket_dest.mkdir(parents=True, exist_ok=True)
    dest_dir = bucket_dest / experimento
    
    try:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        
        shutil.copytree(source_dir, dest_dir)
        
        if logger:
            logger.info(f"✓ Copied experiment to bucket: {source_dir} → {dest_dir}")
        else:
            print(f"✓ Copied experiment to bucket: {source_dir} → {dest_dir}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to copy experiment to bucket: {e}")
        else:
            print(f"⚠ Failed to copy experiment to bucket: {e}")
        return False


def sync_output_to_bucket(config: Dict, logger: Optional[logging.Logger] = None):
    """
    Sync all output files to bucket
    
    Copies:
    - output/{experimento}/ directory
    - output/kaggle/ directory
    - logs/ directory (optional)
    
    Args:
        config: Configuration dictionary
        logger: Optional logger
    """
    if not config.get("bucket", {}).get("enabled", False):
        if logger:
            logger.info("Bucket sync disabled in config")
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
    
    # Copy experiment directory
    copy_experiment_to_bucket(experimento, config, logger)
    
    # Copy kaggle directory if it exists
    kaggle_dir = Path("output/kaggle")
    if kaggle_dir.exists():
        copy_to_bucket(str(kaggle_dir), config, "exp", logger)
    
    # Copy logs directory (optional)
    logs_dir = Path("logs")
    if logs_dir.exists():
        bucket_logs = get_bucket_path(config, "logs")
        if bucket_logs:
            bucket_logs.mkdir(parents=True, exist_ok=True)
            # Copy only the latest log file
            log_files = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            if log_files:
                latest_log = log_files[0]
                copy_to_bucket(str(latest_log), config, "logs", logger)
    
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
        "exp_path": str(Path(base_path) / "exp"),
        "logs_path": str(Path(base_path) / "logs"),
    }

