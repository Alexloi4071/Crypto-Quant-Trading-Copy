"""
Version Manager Module
Handles model and configuration version control for the trading system
Supports versioning for models, features, labels, and strategies
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import shutil
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class VersionType(Enum):
    """Version types supported by the system"""
    MODEL = "model"
    FEATURE = "feature"
    LABEL = "label"
    STRATEGY = "strategy"
    OPTIMIZATION = "optimization"

@dataclass
class VersionInfo:
    """Version information container"""
    symbol: str
    timeframe: str
    version: str
    type: VersionType
    created_at: datetime
    description: str = ""
    parent_version: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModelVersion(VersionInfo):
    """Model-specific version information"""
    model_type: str = ""
    hyperparameters: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    feature_count: int = 0
    training_samples: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class FeatureVersion(VersionInfo):
    """Feature-specific version information"""
    feature_list: List[str] = None
    feature_importance: Dict[str, float] = None
    selection_method: str = ""
    correlation_threshold: float = 0.95
    
    def __post_init__(self):
        super().__post_init__()
        if self.feature_list is None:
            self.feature_list = []
        if self.feature_importance is None:
            self.feature_importance = {}

@dataclass
class LabelVersion(VersionInfo):
    """Label-specific version information"""
    pos_threshold: float = 0.005
    neg_threshold: float = -0.005
    balance_method: str = ""
    class_distribution: Dict[str, float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.class_distribution is None:
            self.class_distribution = {}

class VersionManager:
    """Main version management class"""
    
    def __init__(self):
        self.base_path = config.project_root
        self.versions_dir = self.base_path / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Version registry file
        self.registry_file = self.versions_dir / "version_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load version registry from file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to load version registry: {e}")
            return {}
    
    def _save_registry(self):
        """Save version registry to file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save version registry: {e}")
    
    def get_next_version(self, symbol: str, timeframe: str, version_type: VersionType = VersionType.MODEL) -> str:
        """Get next version number for a symbol/timeframe/type"""
        key = f"{symbol}_{timeframe}_{version_type.value}"
        
        if key not in self.registry:
            self.registry[key] = {'latest_version': 0, 'versions': []}
        
        next_version = self.registry[key]['latest_version'] + 1
        return f"V{next_version}"
    
    def register_version(self, version_info: VersionInfo) -> str:
        """Register a new version"""
        try:
            key = f"{version_info.symbol}_{version_info.timeframe}_{version_info.type.value}"
            
            # Initialize if not exists
            if key not in self.registry:
                self.registry[key] = {'latest_version': 0, 'versions': []}
            
            # Get version number
            if version_info.version.startswith('V'):
                version_num = int(version_info.version[1:])
            else:
                version_num = self.registry[key]['latest_version'] + 1
                version_info.version = f"V{version_num}"
            
            # Update registry
            self.registry[key]['latest_version'] = max(self.registry[key]['latest_version'], version_num)
            
            # Add version record
            version_record = {
                'version': version_info.version,
                'created_at': version_info.created_at.isoformat(),
                'description': version_info.description,
                'parent_version': version_info.parent_version,
                'tags': version_info.tags,
                'metadata': version_info.metadata
            }
            
            # Add type-specific fields
            if isinstance(version_info, ModelVersion):
                version_record.update({
                    'model_type': version_info.model_type,
                    'hyperparameters': version_info.hyperparameters,
                    'performance_metrics': version_info.performance_metrics,
                    'feature_count': version_info.feature_count,
                    'training_samples': version_info.training_samples
                })
            elif isinstance(version_info, FeatureVersion):
                version_record.update({
                    'feature_list': version_info.feature_list,
                    'feature_importance': version_info.feature_importance,
                    'selection_method': version_info.selection_method,
                    'correlation_threshold': version_info.correlation_threshold
                })
            elif isinstance(version_info, LabelVersion):
                version_record.update({
                    'pos_threshold': version_info.pos_threshold,
                    'neg_threshold': version_info.neg_threshold,
                    'balance_method': version_info.balance_method,
                    'class_distribution': version_info.class_distribution
                })
            
            self.registry[key]['versions'].append(version_record)
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Registered version {version_info.version} for {key}")
            return version_info.version
            
        except Exception as e:
            logger.error(f"Failed to register version: {e}")
            raise
    
    def get_version_info(self, symbol: str, timeframe: str, version: str, 
                        version_type: VersionType = VersionType.MODEL) -> Optional[Dict[str, Any]]:
        """Get information about a specific version"""
        try:
            key = f"{symbol}_{timeframe}_{version_type.value}"
            
            if key not in self.registry:
                return None
            
            for version_record in self.registry[key]['versions']:
                if version_record['version'] == version:
                    return version_record
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get version info: {e}")
            return None
    
    def list_versions(self, symbol: str = None, timeframe: str = None, 
                     version_type: VersionType = None) -> List[Dict[str, Any]]:
        """List all versions matching criteria"""
        try:
            results = []
            
            for key, data in self.registry.items():
                key_parts = key.split('_')
                if len(key_parts) >= 3:
                    key_symbol = key_parts[0]
                    key_timeframe = key_parts[1]
                    key_type = key_parts[2]
                    
                    # Apply filters
                    if symbol and key_symbol != symbol:
                        continue
                    if timeframe and key_timeframe != timeframe:
                        continue
                    if version_type and key_type != version_type.value:
                        continue
                    
                    # Add versions
                    for version_record in data['versions']:
                        version_record['symbol'] = key_symbol
                        version_record['timeframe'] = key_timeframe
                        version_record['type'] = key_type
                        results.append(version_record)
            
            # Sort by creation date
            results.sort(key=lambda x: x['created_at'], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def get_latest_version(self, symbol: str, timeframe: str, 
                          version_type: VersionType = VersionType.MODEL) -> Optional[str]:
        """Get latest version for symbol/timeframe/type"""
        try:
            key = f"{symbol}_{timeframe}_{version_type.value}"
            
            if key not in self.registry or not self.registry[key]['versions']:
                return None
            
            latest_num = self.registry[key]['latest_version']
            return f"V{latest_num}"
            
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None
    
    def compare_versions(self, symbol: str, timeframe: str, version1: str, version2: str,
                        version_type: VersionType = VersionType.MODEL) -> Dict[str, Any]:
        """Compare two versions and return differences"""
        try:
            v1_info = self.get_version_info(symbol, timeframe, version1, version_type)
            v2_info = self.get_version_info(symbol, timeframe, version2, version_type)
            
            if not v1_info or not v2_info:
                raise ValueError("One or both versions not found")
            
            comparison = {
                'version1': version1,
                'version2': version2,
                'differences': {},
                'summary': {}
            }
            
            # Compare common fields
            for key in v1_info.keys():
                if key in v2_info and v1_info[key] != v2_info[key]:
                    comparison['differences'][key] = {
                        'v1': v1_info[key],
                        'v2': v2_info[key]
                    }
            
            # Type-specific comparisons
            if version_type == VersionType.MODEL:
                # Performance comparison
                if 'performance_metrics' in v1_info and 'performance_metrics' in v2_info:
                    perf1 = v1_info['performance_metrics']
                    perf2 = v2_info['performance_metrics']
                    
                    comparison['summary']['performance_change'] = {}
                    for metric in set(perf1.keys()) | set(perf2.keys()):
                        val1 = perf1.get(metric, 0)
                        val2 = perf2.get(metric, 0)
                        comparison['summary']['performance_change'][metric] = val2 - val1
            
            elif version_type == VersionType.FEATURE:
                # Feature list comparison
                if 'feature_list' in v1_info and 'feature_list' in v2_info:
                    features1 = set(v1_info['feature_list'])
                    features2 = set(v2_info['feature_list'])
                    
                    comparison['summary']['features_added'] = list(features2 - features1)
                    comparison['summary']['features_removed'] = list(features1 - features2)
                    comparison['summary']['features_common'] = list(features1 & features2)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            return {}
    
    def create_version_branch(self, symbol: str, timeframe: str, base_version: str,
                             new_branch: str, version_type: VersionType = VersionType.MODEL,
                             description: str = "") -> str:
        """Create a new version branch from existing version"""
        try:
            # Get base version info
            base_info = self.get_version_info(symbol, timeframe, base_version, version_type)
            if not base_info:
                raise ValueError(f"Base version {base_version} not found")
            
            # Create new version
            new_version = self.get_next_version(symbol, timeframe, version_type)
            
            # Copy base version directory if exists
            base_path = config.get_model_path(symbol, timeframe, base_version) if version_type == VersionType.MODEL else \
                       config.get_processed_path(symbol, timeframe, base_version)
            new_path = config.get_model_path(symbol, timeframe, new_version) if version_type == VersionType.MODEL else \
                      config.get_processed_path(symbol, timeframe, new_version)
            
            if base_path.exists():
                shutil.copytree(base_path, new_path)
                logger.info(f"Copied {base_path} to {new_path}")
            
            # Create new version info
            if version_type == VersionType.MODEL:
                version_info = ModelVersion(
                    symbol=symbol,
                    timeframe=timeframe,
                    version=new_version,
                    type=version_type,
                    created_at=datetime.now(),
                    description=description or f"Branch from {base_version}",
                    parent_version=base_version,
                    tags=[new_branch],
                    model_type=base_info.get('model_type', ''),
                    hyperparameters=base_info.get('hyperparameters', {}),
                    performance_metrics=base_info.get('performance_metrics', {}),
                    feature_count=base_info.get('feature_count', 0),
                    training_samples=base_info.get('training_samples', 0)
                )
            elif version_type == VersionType.FEATURE:
                version_info = FeatureVersion(
                    symbol=symbol,
                    timeframe=timeframe,
                    version=new_version,
                    type=version_type,
                    created_at=datetime.now(),
                    description=description or f"Branch from {base_version}",
                    parent_version=base_version,
                    tags=[new_branch],
                    feature_list=base_info.get('feature_list', []),
                    feature_importance=base_info.get('feature_importance', {}),
                    selection_method=base_info.get('selection_method', ''),
                    correlation_threshold=base_info.get('correlation_threshold', 0.95)
                )
            else:
                version_info = VersionInfo(
                    symbol=symbol,
                    timeframe=timeframe,
                    version=new_version,
                    type=version_type,
                    created_at=datetime.now(),
                    description=description or f"Branch from {base_version}",
                    parent_version=base_version,
                    tags=[new_branch]
                )
            
            # Register new version
            self.register_version(version_info)
            
            logger.info(f"Created branch {new_branch} as version {new_version}")
            return new_version
            
        except Exception as e:
            logger.error(f"Failed to create version branch: {e}")
            raise
    
    def tag_version(self, symbol: str, timeframe: str, version: str, tag: str,
                   version_type: VersionType = VersionType.MODEL):
        """Add tag to existing version"""
        try:
            key = f"{symbol}_{timeframe}_{version_type.value}"
            
            if key in self.registry:
                for version_record in self.registry[key]['versions']:
                    if version_record['version'] == version:
                        if tag not in version_record['tags']:
                            version_record['tags'].append(tag)
                            self._save_registry()
                            logger.info(f"Added tag '{tag}' to version {version}")
                        return
            
            raise ValueError(f"Version {version} not found")
            
        except Exception as e:
            logger.error(f"Failed to tag version: {e}")
            raise
    
    def delete_version(self, symbol: str, timeframe: str, version: str,
                      version_type: VersionType = VersionType.MODEL, 
                      force: bool = False):
        """Delete a version (with safety checks)"""
        try:
            # Check if version exists
            version_info = self.get_version_info(symbol, timeframe, version, version_type)
            if not version_info:
                raise ValueError(f"Version {version} not found")
            
            # Safety check - don't delete latest version without force
            latest_version = self.get_latest_version(symbol, timeframe, version_type)
            if version == latest_version and not force:
                raise ValueError("Cannot delete latest version without force=True")
            
            # Remove from registry
            key = f"{symbol}_{timeframe}_{version_type.value}"
            self.registry[key]['versions'] = [
                v for v in self.registry[key]['versions'] 
                if v['version'] != version
            ]
            
            # Delete files
            if version_type == VersionType.MODEL:
                version_path = config.get_model_path(symbol, timeframe, version)
            else:
                version_path = config.get_processed_path(symbol, timeframe, version)
            
            if version_path.exists():
                shutil.rmtree(version_path)
                logger.info(f"Deleted version directory: {version_path}")
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Deleted version {version} for {symbol}_{timeframe}")
            
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            raise
    
    def export_version_history(self, output_file: str, 
                             symbol: str = None, timeframe: str = None):
        """Export version history to file"""
        try:
            versions = self.list_versions(symbol, timeframe)
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_versions': len(versions),
                'versions': versions
            }
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif output_path.suffix.lower() in ['.yaml', '.yml']:
                with open(output_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            else:
                raise ValueError("Output file must be .json or .yaml")
            
            logger.info(f"Exported {len(versions)} versions to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export version history: {e}")
            raise
    
    def get_version_lineage(self, symbol: str, timeframe: str, version: str,
                           version_type: VersionType = VersionType.MODEL) -> List[Dict[str, Any]]:
        """Get version lineage (parent chain)"""
        try:
            lineage = []
            current_version = version
            
            while current_version:
                version_info = self.get_version_info(symbol, timeframe, current_version, version_type)
                if not version_info:
                    break
                
                lineage.append(version_info)
                current_version = version_info.get('parent_version')
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get version lineage: {e}")
            return []
    
    def calculate_version_hash(self, symbol: str, timeframe: str, version: str,
                             version_type: VersionType = VersionType.MODEL) -> str:
        """Calculate hash for version content"""
        try:
            if version_type == VersionType.MODEL:
                version_path = config.get_model_path(symbol, timeframe, version)
            else:
                version_path = config.get_processed_path(symbol, timeframe, version)
            
            if not version_path.exists():
                return ""
            
            hash_md5 = hashlib.md5()
            
            # Hash all files in version directory
            for file_path in sorted(version_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate version hash: {e}")
            return ""

# Usage examples and utility functions
def create_model_version(symbol: str, timeframe: str, model_type: str, 
                        hyperparameters: Dict[str, Any], performance_metrics: Dict[str, float],
                        description: str = "") -> str:
    """Convenient function to create a model version"""
    vm = VersionManager()
    
    version_info = ModelVersion(
        symbol=symbol,
        timeframe=timeframe,
        version="",  # Will be auto-generated
        type=VersionType.MODEL,
        created_at=datetime.now(),
        description=description,
        model_type=model_type,
        hyperparameters=hyperparameters,
        performance_metrics=performance_metrics
    )
    
    return vm.register_version(version_info)

def create_feature_version(symbol: str, timeframe: str, feature_list: List[str],
                          feature_importance: Dict[str, float], description: str = "") -> str:
    """Convenient function to create a feature version"""
    vm = VersionManager()
    
    version_info = FeatureVersion(
        symbol=symbol,
        timeframe=timeframe,
        version="",  # Will be auto-generated
        type=VersionType.FEATURE,
        created_at=datetime.now(),
        description=description,
        feature_list=feature_list,
        feature_importance=feature_importance
    )
    
    return vm.register_version(version_info)

if __name__ == "__main__":
    # Example usage
    vm = VersionManager()
    
    # Create a model version
    version = create_model_version(
        "BTCUSDT", "1h", "LightGBM",
        {"num_leaves": 31, "learning_rate": 0.05},
        {"accuracy": 0.75, "auc": 0.82},
        "Initial LightGBM model"
    )
    print(f"Created model version: {version}")
    
    # List all versions
    versions = vm.list_versions()
    print(f"Total versions: {len(versions)}")
    
    # Get latest version
    latest = vm.get_latest_version("BTCUSDT", "1h", VersionType.MODEL)
    print(f"Latest version: {latest}")