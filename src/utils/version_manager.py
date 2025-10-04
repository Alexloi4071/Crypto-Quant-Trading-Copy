"""
Version Manager Module
Simple version management for models and features
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

class VersionManager:
    """Simple version manager for features and models"""

    def __init__(self, base_path: str = "data/processed/versions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.version_file = self.base_path / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict:
        """Load existing versions"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_versions(self):
        """Save versions to file"""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save versions: {e}")

    def get_next_version(self, symbol: str, timeframe: str, component: str = "model") -> str:
        """Get next version number for a component"""
        key = f"{symbol}_{timeframe}_{component}"

        if key not in self.versions:
            self.versions[key] = {
                'current_version': 1,
                'history': []
            }
        else:
            self.versions[key]['current_version'] += 1

        version = f"v{self.versions[key]['current_version']}"

        # Add to history
        self.versions[key]['history'].append({
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'component': component
        })

        self._save_versions()
        return version

    def get_current_version(self, symbol: str, timeframe: str, component: str = "model") -> str:
        """Get current version for a component"""
        key = f"{symbol}_{timeframe}_{component}"

        if key in self.versions:
            return f"v{self.versions[key]['current_version']}"
        else:
            return "v1"

    def list_versions(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict]:
        """List all versions, optionally filtered"""
        result = []

        for key, data in self.versions.items():
            parts = key.split('_')
            if len(parts) >= 3:
                sym, tf, comp = parts[0], parts[1], '_'.join(parts[2:])

                if symbol and sym != symbol:
                    continue
                if timeframe and tf != timeframe:
                    continue

                result.append({
                    'key': key,
                    'symbol': sym,
                    'timeframe': tf,
                    'component': comp,
                    'current_version': data['current_version'],
                    'history': data['history']
                })

        return result

    def create_version_info(self, symbol: str, timeframe: str, component: str,
                          metadata: Dict = None) -> Dict:
        """Create version information"""
        version = self.get_next_version(symbol, timeframe, component)

        info = {
            'version': version,
            'symbol': symbol,
            'timeframe': timeframe,
            'component': component,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        return info
