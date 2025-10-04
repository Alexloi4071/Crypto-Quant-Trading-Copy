#!/usr/bin/env python3
"""
System Maintenance Script
Automated maintenance tasks for the trading system
Includes data cleanup, model versioning, performance monitoring, and health checks
"""

import asyncio
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import psutil
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.database_manager import DatabaseManager
from src.data.storage import DataStorage
from src.models.model_manager import ModelManager
from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.health_checker import HealthChecker
from src.monitoring.notifications import NotificationService

logger = setup_logger(__name__)

class SystemMaintainer:
    """System maintenance and housekeeping"""
    
    def __init__(self):
        self.db_manager = None
        self.storage = None
        self.model_manager = None
        self.system_monitor = None
        self.health_checker = None
        self.notification_service = None
        
        # Maintenance statistics
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'data_cleaned_mb': 0,
            'models_archived': 0,
            'files_removed': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info("System maintainer initialized")
    
    async def initialize(self) -> bool:
        """Initialize maintenance components"""
        try:
            logger.info("Initializing maintenance components")
            
            # Initialize components
            self.db_manager = DatabaseManager()
            self.storage = DataStorage()
            self.model_manager = ModelManager()
            self.system_monitor = SystemMonitor()
            self.health_checker = HealthChecker()
            self.notification_service = NotificationService()
            
            # Initialize connections
            if not await self.db_manager.connect():
                logger.warning("Database connection failed - some functions may be limited")
            
            if not await self.storage.initialize():
                logger.warning("Storage initialization failed - some functions may be limited")
            
            logger.info("Maintenance components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Maintenance initialization failed: {e}")
            return False
    
    async def run_full_maintenance(self, clean_data_days: int = 30,
                                 archive_models_days: int = 90,
                                 cleanup_logs_days: int = 7,
                                 send_report: bool = True) -> Dict[str, Any]:
        """Run comprehensive system maintenance"""
        try:
            logger.info("Starting full system maintenance")
            self.stats['start_time'] = datetime.now()
            
            results = {
                'success': True,
                'maintenance_tasks': {},
                'failed_tasks': [],
                'summary': {}
            }
            
            # Task 1: Clean old data
            try:
                data_cleanup_result = await self._cleanup_old_data(clean_data_days)
                results['maintenance_tasks']['data_cleanup'] = data_cleanup_result
                if data_cleanup_result['success']:
                    self.stats['tasks_completed'] += 1
                    self.stats['data_cleaned_mb'] = data_cleanup_result.get('cleaned_mb', 0)
                else:
                    self.stats['tasks_failed'] += 1
                    results['failed_tasks'].append('data_cleanup')
            except Exception as e:
                logger.error(f"Data cleanup failed: {e}")
                results['failed_tasks'].append('data_cleanup')
                self.stats['tasks_failed'] += 1
            
            # Task 2: Archive old models
            try:
                model_archive_result = await self._archive_old_models(archive_models_days)
                results['maintenance_tasks']['model_archive'] = model_archive_result
                if model_archive_result['success']:
                    self.stats['tasks_completed'] += 1
                    self.stats['models_archived'] = model_archive_result.get('archived_count', 0)
                else:
                    self.stats['tasks_failed'] += 1
                    results['failed_tasks'].append('model_archive')
            except Exception as e:
                logger.error(f"Model archiving failed: {e}")
                results['failed_tasks'].append('model_archive')
                self.stats['tasks_failed'] += 1
            
            # Task 3: Clean logs
            try:
                log_cleanup_result = await self._cleanup_logs(cleanup_logs_days)
                results['maintenance_tasks']['log_cleanup'] = log_cleanup_result
                if log_cleanup_result['success']:
                    self.stats['tasks_completed'] += 1
                    self.stats['files_removed'] += log_cleanup_result.get('files_removed', 0)
                else:
                    self.stats['tasks_failed'] += 1
                    results['failed_tasks'].append('log_cleanup')
            except Exception as e:
                logger.error(f"Log cleanup failed: {e}")
                results['failed_tasks'].append('log_cleanup')
                self.stats['tasks_failed'] += 1
            
            # Task 4: Database maintenance
            try:
                db_maintenance_result = await self._database_maintenance()
                results['maintenance_tasks']['database_maintenance'] = db_maintenance_result
                if db_maintenance_result['success']:
                    self.stats['tasks_completed'] += 1
                else:
                    self.stats['tasks_failed'] += 1
                    results['failed_tasks'].append('database_maintenance')
            except Exception as e:
                logger.error(f"Database maintenance failed: {e}")
                results['failed_tasks'].append('database_maintenance')
                self.stats['tasks_failed'] += 1
            
            # Task 5: System health check
            try:
                health_check_result = await self._comprehensive_health_check()
                results['maintenance_tasks']['health_check'] = health_check_result
                if health_check_result['success']:
                    self.stats['tasks_completed'] += 1
                else:
                    self.stats['tasks_failed'] += 1
                    results['failed_tasks'].append('health_check')
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                results['failed_tasks'].append('health_check')
                self.stats['tasks_failed'] += 1
            
            # Task 6: Cache cleanup
            try:
                cache_cleanup_result = await self._cleanup_caches()
                results['maintenance_tasks']['cache_cleanup'] = cache_cleanup_result
                if cache_cleanup_result['success']:
                    self.stats['tasks_completed'] += 1
                else:
                    self.stats['tasks_failed'] += 1
                    results['failed_tasks'].append('cache_cleanup')
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
                results['failed_tasks'].append('cache_cleanup')
                self.stats['tasks_failed'] += 1
            
            # Generate summary
            self.stats['end_time'] = datetime.now()
            results['summary'] = self._generate_maintenance_summary()
            
            # Send maintenance report
            if send_report:
                await self._send_maintenance_report(results)
            
            logger.info(f"Full maintenance completed: {self.stats['tasks_completed']} successful, {self.stats['tasks_failed']} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Full maintenance failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _cleanup_old_data(self, days_to_keep: int) -> Dict[str, Any]:
        """Clean up old data files and database records"""
        try:
            logger.info(f"Cleaning data older than {days_to_keep} days")
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cleaned_size = 0
            files_removed = 0
            
            # Clean old database records
            if self.db_manager and self.db_manager.is_connected():
                try:
                    await self.db_manager.cleanup_old_records(cutoff_date)
                    logger.debug("Old database records cleaned")
                except Exception as e:
                    logger.error(f"Database record cleanup failed: {e}")
            
            # Clean old data files
            data_dir = Path(config.get('DATA_DIR', 'data'))
            if data_dir.exists():
                for file_path in data_dir.rglob('*'):
                    try:
                        if file_path.is_file():
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                cleaned_size += file_size
                                files_removed += 1
                    except Exception as e:
                        logger.debug(f"Failed to remove file {file_path}: {e}")
                        continue
            
            # Clean storage cache
            if self.storage:
                try:
                    await self.storage.cleanup_old_data(days_to_keep)
                    logger.debug("Storage cache cleaned")
                except Exception as e:
                    logger.error(f"Storage cache cleanup failed: {e}")
            
            cleaned_mb = cleaned_size / (1024 * 1024)
            
            logger.info(f"Data cleanup completed: {files_removed} files removed, {cleaned_mb:.1f}MB freed")
            
            return {
                'success': True,
                'files_removed': files_removed,
                'cleaned_mb': cleaned_mb,
                'cutoff_date': cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _archive_old_models(self, days_to_keep: int) -> Dict[str, Any]:
        """Archive old model files"""
        try:
            logger.info(f"Archiving models older than {days_to_keep} days")
            
            if not self.model_manager:
                return {'success': False, 'error': 'Model manager not available'}
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Create archive directory
            archive_dir = Path(config.get('MODELS_DIR', 'models')) / 'archive' / datetime.now().strftime('%Y%m%d')
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            archived_count = 0
            archived_size = 0
            
            # Get models to archive
            models_dir = Path(config.get('MODELS_DIR', 'models'))
            if models_dir.exists():
                for model_path in models_dir.rglob('*.pkl'):
                    try:
                        model_time = datetime.fromtimestamp(model_path.stat().st_mtime)
                        if model_time < cutoff_date:
                            # Move to archive
                            relative_path = model_path.relative_to(models_dir)
                            archive_path = archive_dir / relative_path
                            archive_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            model_size = model_path.stat().st_size
                            shutil.move(str(model_path), str(archive_path))
                            
                            archived_count += 1
                            archived_size += model_size
                            
                            logger.debug(f"Archived model: {model_path}")
                            
                    except Exception as e:
                        logger.debug(f"Failed to archive model {model_path}: {e}")
                        continue
            
            archived_mb = archived_size / (1024 * 1024)
            
            logger.info(f"Model archiving completed: {archived_count} models archived, {archived_mb:.1f}MB moved")
            
            return {
                'success': True,
                'archived_count': archived_count,
                'archived_mb': archived_mb,
                'archive_dir': str(archive_dir)
            }
            
        except Exception as e:
            logger.error(f"Model archiving failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _cleanup_logs(self, days_to_keep: int) -> Dict[str, Any]:
        """Clean up old log files"""
        try:
            logger.info(f"Cleaning logs older than {days_to_keep} days")
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            files_removed = 0
            size_freed = 0
            
            # Clean log directories
            log_dirs = [
                Path(config.get('LOG_DIR', 'logs')),
                Path('logs'),
                Path('/tmp')  # Clean temporary logs
            ]
            
            for log_dir in log_dirs:
                if not log_dir.exists():
                    continue
                
                # Remove old log files
                for log_file in log_dir.glob('*.log*'):
                    try:
                        if log_file.is_file():
                            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                            if file_time < cutoff_date:
                                file_size = log_file.stat().st_size
                                log_file.unlink()
                                files_removed += 1
                                size_freed += file_size
                                logger.debug(f"Removed log file: {log_file}")
                    except Exception as e:
                        logger.debug(f"Failed to remove log file {log_file}: {e}")
                        continue
            
            freed_mb = size_freed / (1024 * 1024)
            
            logger.info(f"Log cleanup completed: {files_removed} files removed, {freed_mb:.1f}MB freed")
            
            return {
                'success': True,
                'files_removed': files_removed,
                'freed_mb': freed_mb
            }
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _database_maintenance(self) -> Dict[str, Any]:
        """Perform database maintenance tasks"""
        try:
            logger.info("Performing database maintenance")
            
            if not self.db_manager or not self.db_manager.is_connected():
                return {'success': False, 'error': 'Database not available'}
            
            maintenance_tasks = []
            
            # Vacuum/optimize tables
            try:
                await self.db_manager.vacuum_database()
                maintenance_tasks.append("Database vacuumed")
            except Exception as e:
                logger.error(f"Database vacuum failed: {e}")
                maintenance_tasks.append(f"Database vacuum failed: {e}")
            
            # Update statistics
            try:
                await self.db_manager.update_statistics()
                maintenance_tasks.append("Statistics updated")
            except Exception as e:
                logger.error(f"Statistics update failed: {e}")
                maintenance_tasks.append(f"Statistics update failed: {e}")
            
            # Check integrity
            try:
                integrity_ok = await self.db_manager.check_integrity()
                if integrity_ok:
                    maintenance_tasks.append("Integrity check passed")
                else:
                    maintenance_tasks.append("Integrity check failed")
            except Exception as e:
                logger.error(f"Integrity check failed: {e}")
                maintenance_tasks.append(f"Integrity check failed: {e}")
            
            # Get database size
            try:
                db_stats = await self.db_manager.get_database_stats()
                size_mb = db_stats.get('size_mb', 0)
                maintenance_tasks.append(f"Database size: {size_mb:.1f}MB")
            except Exception as e:
                logger.debug(f"Database stats retrieval failed: {e}")
            
            logger.info(f"Database maintenance completed: {len(maintenance_tasks)} tasks")
            
            return {
                'success': True,
                'tasks_completed': maintenance_tasks,
                'task_count': len(maintenance_tasks)
            }
            
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            logger.info("Performing comprehensive health check")
            
            health_results = {}
            issues_found = []
            
            # System resources check
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                health_results['system_resources'] = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': (disk.used / disk.total) * 100,
                    'available_memory_gb': memory.available / (1024**3),
                    'free_disk_gb': disk.free / (1024**3)
                }
                
                # Check for resource issues
                if cpu_percent > 90:
                    issues_found.append(f"High CPU usage: {cpu_percent:.1f}%")
                if memory.percent > 90:
                    issues_found.append(f"High memory usage: {memory.percent:.1f}%")
                if (disk.used / disk.total) * 100 > 90:
                    issues_found.append(f"Low disk space: {((disk.used / disk.total) * 100):.1f}% used")
                
            except Exception as e:
                health_results['system_resources'] = {'error': str(e)}
                issues_found.append(f"System resources check failed: {e}")
            
            # Health checker results
            if self.health_checker:
                try:
                    health_summary = self.health_checker.get_system_health_summary()
                    health_results['health_checks'] = health_summary
                    
                    if health_summary.get('overall_status') != 'healthy':
                        issues_found.append(f"System health status: {health_summary.get('overall_status')}")
                    
                except Exception as e:
                    health_results['health_checks'] = {'error': str(e)}
                    issues_found.append(f"Health checks failed: {e}")
            
            # Database connectivity
            if self.db_manager:
                try:
                    db_connected = self.db_manager.is_connected()
                    health_results['database'] = {'connected': db_connected}
                    
                    if not db_connected:
                        issues_found.append("Database not connected")
                    
                except Exception as e:
                    health_results['database'] = {'error': str(e)}
                    issues_found.append(f"Database check failed: {e}")
            
            # File system permissions
            try:
                important_dirs = [
                    Path(config.get('DATA_DIR', 'data')),
                    Path(config.get('MODELS_DIR', 'models')),
                    Path(config.get('LOG_DIR', 'logs'))
                ]
                
                permission_issues = []
                for dir_path in important_dirs:
                    if dir_path.exists():
                        if not dir_path.is_dir():
                            permission_issues.append(f"{dir_path} is not a directory")
                        elif not os.access(dir_path, os.R_OK | os.W_OK):
                            permission_issues.append(f"{dir_path} has insufficient permissions")
                    else:
                        permission_issues.append(f"{dir_path} does not exist")
                
                health_results['file_permissions'] = {
                    'issues': permission_issues,
                    'directories_checked': len(important_dirs)
                }
                
                issues_found.extend(permission_issues)
                
            except Exception as e:
                health_results['file_permissions'] = {'error': str(e)}
                issues_found.append(f"Permission check failed: {e}")
            
            # Determine overall health
            overall_healthy = len(issues_found) == 0
            
            logger.info(f"Health check completed: {'healthy' if overall_healthy else 'issues found'}")
            
            return {
                'success': True,
                'overall_healthy': overall_healthy,
                'issues_count': len(issues_found),
                'issues_found': issues_found,
                'detailed_results': health_results
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _cleanup_caches(self) -> Dict[str, Any]:
        """Clean up various system caches"""
        try:
            logger.info("Cleaning system caches")
            
            cleanup_results = {}
            total_cleaned = 0
            
            # Storage cache cleanup
            if self.storage:
                try:
                    self.storage.clear_cache()
                    cleanup_results['storage_cache'] = 'cleared'
                    total_cleaned += 1
                except Exception as e:
                    cleanup_results['storage_cache'] = f'failed: {e}'
            
            # System monitor cache
            if self.system_monitor:
                try:
                    self.system_monitor.clear_cache()
                    cleanup_results['monitor_cache'] = 'cleared'
                    total_cleaned += 1
                except Exception as e:
                    cleanup_results['monitor_cache'] = f'failed: {e}'
            
            # Temporary files cleanup
            try:
                temp_dir = Path('/tmp')
                temp_files_removed = 0
                
                if temp_dir.exists():
                    for temp_file in temp_dir.glob('trading_*'):
                        try:
                            if temp_file.is_file():
                                temp_file.unlink()
                                temp_files_removed += 1
                        except Exception:
                            continue
                
                cleanup_results['temp_files'] = f'{temp_files_removed} files removed'
                total_cleaned += 1
                
            except Exception as e:
                cleanup_results['temp_files'] = f'failed: {e}'
            
            logger.info(f"Cache cleanup completed: {total_cleaned} cache types cleaned")
            
            return {
                'success': True,
                'caches_cleaned': total_cleaned,
                'cleanup_details': cleanup_results
            }
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_maintenance_summary(self) -> Dict[str, Any]:
        """Generate maintenance summary"""
        try:
            duration = None
            if self.stats['start_time'] and self.stats['end_time']:
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            success_rate = 0
            total_tasks = self.stats['tasks_completed'] + self.stats['tasks_failed']
            if total_tasks > 0:
                success_rate = (self.stats['tasks_completed'] / total_tasks) * 100
            
            return {
                'total_tasks': total_tasks,
                'completed_tasks': self.stats['tasks_completed'],
                'failed_tasks': self.stats['tasks_failed'],
                'success_rate_percent': round(success_rate, 1),
                'duration_seconds': round(duration, 1) if duration else None,
                'data_cleaned_mb': self.stats['data_cleaned_mb'],
                'models_archived': self.stats['models_archived'],
                'files_removed': self.stats['files_removed'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {}
    
    async def _send_maintenance_report(self, results: Dict[str, Any]):
        """Send maintenance report via notifications"""
        try:
            if not self.notification_service:
                return
            
            summary = results.get('summary', {})
            failed_tasks = results.get('failed_tasks', [])
            
            # Create report message
            title = "🔧 System Maintenance Report"
            
            content_parts = [
                f"**Maintenance Summary:**",
                f"✅ Completed: {summary.get('completed_tasks', 0)} tasks",
                f"❌ Failed: {summary.get('failed_tasks', 0)} tasks",
                f"📊 Success Rate: {summary.get('success_rate_percent', 0):.1f}%",
                f"⏱️ Duration: {summary.get('duration_seconds', 0):.1f} seconds",
                "",
                f"**Resources Cleaned:**",
                f"💾 Data: {summary.get('data_cleaned_mb', 0):.1f} MB",
                f"🗃️ Models Archived: {summary.get('models_archived', 0)}",
                f"🗑️ Files Removed: {summary.get('files_removed', 0)}"
            ]
            
            if failed_tasks:
                content_parts.extend([
                    "",
                    "**Failed Tasks:**"
                ])
                for task in failed_tasks[:5]:  # Show first 5
                    content_parts.append(f"• {task}")
            
            content = "\n".join(content_parts)
            
            # Send notification
            await self.notification_service.send_notification(
                title=title,
                content=content,
                channel=self.notification_service.NotificationChannel.EMAIL,
                priority=self.notification_service.NotificationPriority.NORMAL
            )
            
            logger.info("Maintenance report sent")
            
        except Exception as e:
            logger.error(f"Failed to send maintenance report: {e}")
    
    async def backup_system(self, backup_dir: str = None) -> Dict[str, Any]:
        """Create system backup"""
        try:
            logger.info("Creating system backup")
            
            if backup_dir is None:
                backup_dir = f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            backup_results = {
                'success': True,
                'backup_path': str(backup_path),
                'backup_size_mb': 0,
                'components_backed_up': []
            }
            
            # Backup data
            if self.storage:
                try:
                    await self.storage.backup_data(backup_path / 'data')
                    backup_results['components_backed_up'].append('data')
                except Exception as e:
                    logger.error(f"Data backup failed: {e}")
            
            # Backup models
            models_dir = Path(config.get('MODELS_DIR', 'models'))
            if models_dir.exists():
                try:
                    shutil.copytree(models_dir, backup_path / 'models', dirs_exist_ok=True)
                    backup_results['components_backed_up'].append('models')
                except Exception as e:
                    logger.error(f"Models backup failed: {e}")
            
            # Backup configuration
            config_files = ['config.json', '.env', 'settings.py']
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    try:
                        shutil.copy2(config_path, backup_path / config_file)
                        backup_results['components_backed_up'].append(config_file)
                    except Exception as e:
                        logger.debug(f"Config backup failed for {config_file}: {e}")
            
            # Calculate backup size
            try:
                total_size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
                backup_results['backup_size_mb'] = total_size / (1024 * 1024)
            except Exception as e:
                logger.debug(f"Backup size calculation failed: {e}")
            
            logger.info(f"System backup completed: {len(backup_results['components_backed_up'])} components backed up")
            
            return backup_results
            
        except Exception as e:
            logger.error(f"System backup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            import os
            
            # System information
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                    'python_version': sys.version,
                    'platform': sys.platform
                },
                'directories': {},
                'database_connected': self.db_manager.is_connected() if self.db_manager else False,
                'maintenance_stats': self.stats
            }
            
            # Check important directories
            important_dirs = {
                'data': Path(config.get('DATA_DIR', 'data')),
                'models': Path(config.get('MODELS_DIR', 'models')),
                'logs': Path(config.get('LOG_DIR', 'logs'))
            }
            
            for name, dir_path in important_dirs.items():
                if dir_path.exists():
                    try:
                        # Count files and calculate size
                        files = list(dir_path.rglob('*'))
                        file_count = len([f for f in files if f.is_file()])
                        total_size = sum(f.stat().st_size for f in files if f.is_file())
                        
                        status['directories'][name] = {
                            'exists': True,
                            'file_count': file_count,
                            'size_mb': total_size / (1024 * 1024)
                        }
                    except Exception as e:
                        status['directories'][name] = {'exists': True, 'error': str(e)}
                else:
                    status['directories'][name] = {'exists': False}
            
            return status
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {'error': str(e)}

async def main():
    """Main function for system maintenance script"""
    parser = argparse.ArgumentParser(description='System Maintenance Script')
    
    # Command selection
    parser.add_argument('command', choices=[
        'full', 'data', 'models', 'logs', 'db', 'health', 'backup', 'status'
    ], help='Maintenance command')
    
    # Time-based options
    parser.add_argument('--data-days', type=int, default=30,
                       help='Days of data to keep')
    
    parser.add_argument('--model-days', type=int, default=90,
                       help='Days of models to keep')
    
    parser.add_argument('--log-days', type=int, default=7,
                       help='Days of logs to keep')
    
    # Backup options
    parser.add_argument('--backup-dir',
                       help='Backup directory path')
    
    # Notification options
    parser.add_argument('--no-report', action='store_true',
                       help='Skip sending maintenance report')
    
    # Output options
    parser.add_argument('--output',
                       help='Output file for results (JSON format)')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize maintainer
    maintainer = SystemMaintainer()
    
    if not await maintainer.initialize():
        logger.error("Failed to initialize system maintainer")
        return False
    
    try:
        results = None
        
        if args.command == 'full':
            # Full maintenance
            results = await maintainer.run_full_maintenance(
                clean_data_days=args.data_days,
                archive_models_days=args.model_days,
                cleanup_logs_days=args.log_days,
                send_report=not args.no_report
            )
            
        elif args.command == 'data':
            # Data cleanup only
            results = await maintainer._cleanup_old_data(args.data_days)
            
        elif args.command == 'models':
            # Model archiving only
            results = await maintainer._archive_old_models(args.model_days)
            
        elif args.command == 'logs':
            # Log cleanup only
            results = await maintainer._cleanup_logs(args.log_days)
            
        elif args.command == 'db':
            # Database maintenance only
            results = await maintainer._database_maintenance()
            
        elif args.command == 'health':
            # Health check only
            results = await maintainer._comprehensive_health_check()
            
        elif args.command == 'backup':
            # System backup
            results = await maintainer.backup_system(args.backup_dir)
            
        elif args.command == 'status':
            # System status
            results = maintainer.get_system_status()
            results['success'] = True
        
        # Output results
        if results:
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {args.output}")
            else:
                # Print summary
                if results.get('success'):
                    print("✅ Maintenance completed successfully")
                    
                    if results.get('summary'):
                        summary = results['summary']
                        print(f"Tasks completed: {summary.get('completed_tasks', 0)}")
                        print(f"Tasks failed: {summary.get('failed_tasks', 0)}")
                        print(f"Data cleaned: {summary.get('data_cleaned_mb', 0):.1f} MB")
                    
                    if results.get('failed_tasks'):
                        print(f"Failed tasks: {', '.join(results['failed_tasks'])}")
                else:
                    print("❌ Maintenance failed")
                    if results.get('error'):
                        print(f"Error: {results['error']}")
            
            return results.get('success', True)
        else:
            logger.error("No results generated")
            return False
            
    except KeyboardInterrupt:
        logger.info("Maintenance cancelled by user")
        return False
    except Exception as e:
        logger.error(f"Maintenance script failed: {e}")
        return False

if __name__ == "__main__":
    import os
    success = asyncio.run(main())
    sys.exit(0 if success else 1)