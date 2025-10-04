"""
部署腳本
自動化部署量化交易系統到不同環境
支持本地開發、測試環境、生產環境部署
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml
import json
import warnings

warnings.filterwarnings('ignore')

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DeploymentManager:
    """部署管理器"""

    def __init__(self):
        self.project_root = project_root
        self.environments = {
            'development': {
                'name': '開發環境',
                'python_version': '3.10',
                'requirements_file': 'requirements.txt',
                'env_file': '.env.dev',
                'database_url': 'sqlite:///data/dev.db',
                'log_level': 'DEBUG',
                'testnet': True
            },
            'testing': {
                'name': '測試環境',
                'python_version': '3.10',
                'requirements_file': 'requirements.txt',
                'env_file': '.env.test',
                'database_url': 'sqlite:///data/test.db',
                'log_level': 'INFO',
                'testnet': True
            },
            'staging': {
                'name': '預發布環境',
                'python_version': '3.10',
                'requirements_file': 'requirements.txt',
                'env_file': '.env.staging',
                'database_url': 'postgresql://user:pass@localhost/crypto_staging',
                'log_level': 'INFO',
                'testnet': True
            },
            'production': {
                'name': '生產環境',
                'python_version': '3.10',
                'requirements_file': 'requirements.txt',
                'env_file': '.env.prod',
                'database_url': 'postgresql://user:pass@localhost/crypto_trading',
                'log_level': 'WARNING',
                'testnet': False
            }
        }

        logger.info("部署管理器初始化完成")

    def deploy_to_environment(self, environment: str, force: bool = False) -> bool:
        """部署到指定環境"""
        try:
            if environment not in self.environments:
                logger.error(f"未知環境: {environment}")
                return False

            env_config = self.environments[environment]
            logger.info(f"開始部署到{env_config['name']}")

            # 1. 環境檢查
            if not self._check_environment_prerequisites(environment):
                if not force:
                    logger.error("環境檢查失敗")
                    return False
                else:
                    logger.warning("環境檢查失敗，但強制繼續")

            # 2. 備份現有部署（生產環境）
            if environment == 'production':
                self._backup_current_deployment()

            # 3. 準備部署目錄
            deploy_dir = self._prepare_deployment_directory(environment)
            if not deploy_dir:
                return False

            # 4. 複製代碼文件
            if not self._copy_source_code(deploy_dir):
                return False

            # 5. 設置虛擬環境
            venv_path = self._setup_virtual_environment(deploy_dir, env_config)
            if not venv_path:
                return False

            # 6. 安裝依賴
            if not self._install_dependencies(venv_path, deploy_dir, env_config):
                return False

            # 7. 配置環境變量
            if not self._setup_environment_config(deploy_dir, env_config):
                return False

            # 8. 初始化數據庫
            if not self._initialize_database(venv_path, deploy_dir, environment):
                return False

            # 9. 運行測試（非生產環境）
            if environment != 'production':
                if not self._run_deployment_tests(venv_path, deploy_dir):
                    logger.warning("部署測試失敗")

            # 10. 設置系統服務
            if environment in ['staging', 'production']:
                self._setup_system_services(deploy_dir, environment)

            # 11. 創建部署記錄
            self._create_deployment_record(environment, deploy_dir)

            logger.info(f"部署到{env_config['name']}完成")
            return True

        except Exception as e:
            logger.error(f"部署失敗: {e}")
            return False

    def _check_environment_prerequisites(self, environment: str) -> bool:
        """檢查環境前置條件"""
        try:
            env_config = self.environments[environment]

            # 檢查Python版本
            python_version = self._get_python_version()
            required_version = env_config['python_version']

            if not python_version.startswith(required_version):
                logger.error(f"Python版本不匹配: 需要{required_version}, 當前{python_version}")
                return False

            # 檢查必要的工具
            required_tools = ['pip', 'git']
            if environment in ['staging', 'production']:
                required_tools.extend(['systemctl', 'nginx'])

            for tool in required_tools:
                if not self._check_command_exists(tool):
                    logger.error(f"缺少必要工具: {tool}")
                    return False

            # 檢查磁盤空間
            free_space_gb = self._get_free_disk_space()
            required_space_gb = 2.0  # 至少2GB空間

            if free_space_gb < required_space_gb:
                logger.error(f"磁盤空間不足: 需要{required_space_gb}GB, 可用{free_space_gb}GB")
                return False

            logger.info("環境檢查通過")
            return True

        except Exception as e:
            logger.error(f"環境檢查失敗: {e}")
            return False

    def _backup_current_deployment(self) -> bool:
        """備份當前部署"""
        try:
            backup_dir = Path(f"/opt/crypto-trading-backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            current_deployment = Path("/opt/crypto-trading")

            if current_deployment.exists():
                shutil.copytree(current_deployment, backup_dir)
                logger.info(f"當前部署已備份到: {backup_dir}")

            return True

        except Exception as e:
            logger.error(f"備份部署失敗: {e}")
            return False

    def _prepare_deployment_directory(self, environment: str) -> Optional[Path]:
        """準備部署目錄"""
        try:
            if environment == 'development':
                deploy_dir = self.project_root / "deploy" / "dev"
            elif environment == 'testing':
                deploy_dir = self.project_root / "deploy" / "test"
            elif environment == 'staging':
                deploy_dir = Path("/opt/crypto-trading-staging")
            else:  # production
                deploy_dir = Path("/opt/crypto-trading")

            # 創建目錄
            deploy_dir.mkdir(parents=True, exist_ok=True)

            # 設置權限
            if environment in ['staging', 'production']:
                os.chmod(deploy_dir, 0o755)

            logger.info(f"部署目錄準備完成: {deploy_dir}")
            return deploy_dir

        except Exception as e:
            logger.error(f"準備部署目錄失敗: {e}")
            return None

    def _copy_source_code(self, deploy_dir: Path) -> bool:
        """複製源代碼"""
        try:
            # 需要複製的目錄和文件
            source_items = [
                'src', 'config', 'scripts', 'database', 'api', 'frontend',
                'realtime', 'extensions', 'advanced', 'ai_enhanced',
                'requirements.txt', 'setup.py'
            ]

            # 忽略的模式
            ignore_patterns = [
                '*.pyc', '__pycache__', '.pytest_cache', '.git',
                '*.log', 'logs/*', 'data/raw/*', 'results/*',
                '.env*', 'venv', 'env', '.vscode', '.idea'
            ]

            def ignore_function(dir_path, names):
                ignored = []
                for name in names:
                    for pattern in ignore_patterns:
                        import fnmatch
                        if fnmatch.fnmatch(name, pattern):
                            ignored.append(name)
                            break
                return ignored

            # 複製文件
            for item in source_items:
                source_path = self.project_root / item
                target_path = deploy_dir / item

                if source_path.exists():
                    if source_path.is_file():
                        shutil.copy2(source_path, target_path)
                    else:
                        if target_path.exists():
                            shutil.rmtree(target_path)
                        shutil.copytree(source_path, target_path, ignore=ignore_function)

                    logger.debug(f"複製: {source_path} -> {target_path}")

            logger.info("源代碼複製完成")
            return True

        except Exception as e:
            logger.error(f"複製源代碼失敗: {e}")
            return False

    def _setup_virtual_environment(self, deploy_dir: Path, env_config: Dict) -> Optional[Path]:
        """設置虛擬環境"""
        try:
            venv_path = deploy_dir / "venv"

            # 刪除現有虛擬環境
            if venv_path.exists():
                shutil.rmtree(venv_path)

            # 創建新虛擬環境
            result = subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"創建虛擬環境失敗: {result.stderr}")
                return None

            # 升級pip
            pip_path = venv_path / ("Scripts" if os.name == 'nt' else "bin") / "pip"
            result = subprocess.run([
                str(pip_path), "install", "--upgrade", "pip"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"升級pip失敗: {result.stderr}")

            logger.info(f"虛擬環境創建完成: {venv_path}")
            return venv_path

        except Exception as e:
            logger.error(f"設置虛擬環境失敗: {e}")
            return None

    def _install_dependencies(self, venv_path: Path, deploy_dir: Path, env_config: Dict) -> bool:
        """安裝依賴包"""
        try:
            pip_path = venv_path / ("Scripts" if os.name == 'nt' else "bin") / "pip"
            requirements_file = deploy_dir / env_config['requirements_file']

            if not requirements_file.exists():
                logger.error(f"requirements文件不存在: {requirements_file}")
                return False

            # 安裝依賴
            logger.info("安裝Python依賴包...")
            result = subprocess.run([
                str(pip_path), "install", "-r", str(requirements_file)
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"安裝依賴失敗: {result.stderr}")
                return False

            # 安裝項目包（開發模式）
            if (deploy_dir / "setup.py").exists():
                result = subprocess.run([
                    str(pip_path), "install", "-e", "."
                ], cwd=deploy_dir, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.warning(f"安裝項目包失敗: {result.stderr}")

            logger.info("依賴包安裝完成")
            return True

        except Exception as e:
            logger.error(f"安裝依賴包失敗: {e}")
            return False

    def _setup_environment_config(self, deploy_dir: Path, env_config: Dict) -> bool:
        """設置環境配置"""
        try:
            env_file = deploy_dir / ".env"

            # 生成環境配置
            env_content = f"""# 自動生成的環境配置文件
# 生成時間: {datetime.now().isoformat()}

# 環境設置
ENVIRONMENT={env_config.get('name', 'development')}
LOG_LEVEL={env_config.get('log_level', 'INFO')}
TESTNET={str(env_config.get('testnet', True)).lower()}

# 數據庫配置
DATABASE_URL={env_config.get('database_url', 'sqlite:///data/trading.db')}

# API配置（需要手動設置）
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Telegram配置（可選）
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# 系統配置
API_HOST=0.0.0.0
API_PORT=8080
"""

            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)

            # 設置權限
            os.chmod(env_file, 0o600)  # 只有所有者可讀寫

            logger.info("環境配置設置完成")
            logger.warning("請手動設置API密鑰和其他敏感配置")
            return True

        except Exception as e:
            logger.error(f"設置環境配置失敗: {e}")
            return False

    def _initialize_database(self, venv_path: Path, deploy_dir: Path, environment: str) -> bool:
        """初始化數據庫"""
        try:
            python_path = venv_path / ("Scripts" if os.name == 'nt' else "bin") / "python"
            init_script = deploy_dir / "database" / "init_db.py"

            if not init_script.exists():
                logger.warning("數據庫初始化腳本不存在，跳過")
                return True

            # 運行數據庫初始化
            logger.info("初始化數據庫...")
            result = subprocess.run([
                str(python_path), str(init_script)
            ], cwd=deploy_dir, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"數據庫初始化失敗: {result.stderr}")
                return False

            logger.info("數據庫初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化數據庫失敗: {e}")
            return False

    def _run_deployment_tests(self, venv_path: Path, deploy_dir: Path) -> bool:
        """運行部署測試"""
        try:
            python_path = venv_path / ("Scripts" if os.name == 'nt' else "bin") / "python"

            # 運行環境檢查
            check_script = deploy_dir / "check_environment.py"
            if check_script.exists():
                logger.info("運行環境檢查...")
                result = subprocess.run([
                    str(python_path), str(check_script)
                ], cwd=deploy_dir, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"環境檢查失敗: {result.stderr}")
                    return False

            # 運行基礎測試
            test_dir = deploy_dir / "tests"
            if test_dir.exists():
                logger.info("運行基礎測試...")
                pytest_path = venv_path / ("Scripts" if os.name == 'nt' else "bin") / "pytest"

                if pytest_path.exists():
                    result = subprocess.run([
                        str(pytest_path), str(test_dir), "-v"
                    ], cwd=deploy_dir, capture_output=True, text=True)

                    if result.returncode != 0:
                        logger.warning(f"部分測試失敗: {result.stderr}")

            logger.info("部署測試完成")
            return True

        except Exception as e:
            logger.error(f"運行部署測試失敗: {e}")
            return False

    def _setup_system_services(self, deploy_dir: Path, environment: str):
        """設置系統服務"""
        try:
            if os.name == 'nt':
                logger.warning("Windows系統不支持systemd服務")
                return

            service_name = f"crypto-trading-{environment}"
            service_file = f"/etc/systemd/system/{service_name}.service"

            # 服務配置內容
            service_content = f"""[Unit]
Description=Crypto Trading System ({environment})
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory={deploy_dir}
Environment=PATH={deploy_dir}/venv/bin
ExecStart={deploy_dir}/venv/bin/python {deploy_dir}/api/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

            # 寫入服務文件（需要root權限）
            try:
                with open(service_file, 'w') as f:
                    f.write(service_content)

                # 重載systemd並啟用服務
                subprocess.run(["systemctl", "daemon-reload"], check=True)
                subprocess.run(["systemctl", "enable", service_name], check=True)

                logger.info(f"系統服務已設置: {service_name}")

            except PermissionError:
                logger.warning("設置系統服務需要root權限，請手動創建服務文件")
                logger.info(f"服務文件內容:\n{service_content}")

        except Exception as e:
            logger.error(f"設置系統服務失敗: {e}")

    def _create_deployment_record(self, environment: str, deploy_dir: Path):
        """創建部署記錄"""
        try:
            deployment_info = {
                'environment': environment,
                'deploy_dir': str(deploy_dir),
                'timestamp': datetime.now().isoformat(),
                'python_version': self._get_python_version(),
                'git_commit': self._get_git_commit(),
                'deployment_user': os.getenv('USER', 'unknown')
            }

            record_file = deploy_dir / "deployment_info.json"
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(deployment_info, f, indent=2, ensure_ascii=False)

            logger.info(f"部署記錄已創建: {record_file}")

        except Exception as e:
            logger.error(f"創建部署記錄失敗: {e}")

    def rollback_deployment(self, environment: str) -> bool:
        """回滾部署"""
        try:
            if environment != 'production':
                logger.error("只支持生產環境回滾")
                return False

            # 查找最新備份
            backup_pattern = "/opt/crypto-trading-backup-*"
            import glob

            backups = glob.glob(backup_pattern)
            if not backups:
                logger.error("沒有找到備份")
                return False

            latest_backup = max(backups, key=os.path.getctime)
            current_deployment = Path("/opt/crypto-trading")

            # 停止服務
            try:
                subprocess.run(["systemctl", "stop", "crypto-trading-production"], check=True)
            except:
                logger.warning("無法停止服務")

            # 執行回滾
            if current_deployment.exists():
                shutil.rmtree(current_deployment)

            shutil.copytree(latest_backup, current_deployment)

            # 重啟服務
            try:
                subprocess.run(["systemctl", "start", "crypto-trading-production"], check=True)
            except:
                logger.warning("無法重啟服務")

            logger.info(f"回滾完成，使用備份: {latest_backup}")
            return True

        except Exception as e:
            logger.error(f"回滾部署失敗: {e}")
            return False

    def _get_python_version(self) -> str:
        """獲取Python版本"""
        try:
            result = subprocess.run([sys.executable, "--version"],
                                  capture_output=True, text=True)
            return result.stdout.strip().split()[1]
        except:
            return "unknown"

    def _check_command_exists(self, command: str) -> bool:
        """檢查命令是否存在"""
        try:
            result = subprocess.run(["which", command],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _get_free_disk_space(self) -> float:
        """獲取可用磁盤空間（GB）"""
        try:
            statvfs = os.statvfs('.')
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            return free_bytes / (1024 ** 3)
        except:
            return 0.0

    def _get_git_commit(self) -> str:
        """獲取Git提交ID"""
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"],
                                  capture_output=True, text=True, cwd=self.project_root)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="量化交易系統部署工具")
    parser.add_argument("command", choices=['deploy', 'rollback', 'status'],
                       help="部署命令")
    parser.add_argument("--env", choices=['development', 'testing', 'staging', 'production'],
                       required=True, help="目標環境")
    parser.add_argument("--force", action="store_true", help="強制部署（忽略檢查失敗）")

    args = parser.parse_args()

    manager = DeploymentManager()

    try:
        if args.command == 'deploy':
            success = manager.deploy_to_environment(args.env, args.force)
            if success:
                print(f"✅ 部署到{args.env}環境成功")

                # 顯示後續步驟
                if args.env in ['staging', 'production']:
                    print(f"\n📋 後續步驟:")
                    print(f"1. 手動設置 .env 文件中的API密鑰")
                    print(f"2. 檢查數據庫連接")
                    print(f"3. 啟動服務: systemctl start crypto-trading-{args.env}")
                    print(f"4. 檢查服務狀態: systemctl status crypto-trading-{args.env}")
            else:
                print(f"❌ 部署到{args.env}環境失敗")
                sys.exit(1)

        elif args.command == 'rollback':
            success = manager.rollback_deployment(args.env)
            if success:
                print(f"✅ {args.env}環境回滾成功")
            else:
                print(f"❌ {args.env}環境回滾失敗")
                sys.exit(1)

        elif args.command == 'status':
            # 顯示部署狀態
            print(f"📊 {args.env}環境狀態:")
            # 這裡可以添加狀態檢查邏輯
            print("功能開發中...")

    except KeyboardInterrupt:
        print("\n部署被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 部署過程中發生錯誤: {e}")
        logger.error(f"部署失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
