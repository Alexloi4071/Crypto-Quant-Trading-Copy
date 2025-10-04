"""
API Configuration Module
FastAPI 應用配置管理
"""

import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

class APIConfig:
    """API 配置類"""
    
    # 服務器配置
    HOST: str = os.getenv('API_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('API_PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'True').lower() == 'true'
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')
    
    # CORS 配置
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ]
    
    # 如果是開發環境，允許所有來源
    if ENVIRONMENT == 'development':
        CORS_ORIGINS.append("*")
    
    # 安全配置
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
    
    # 認證配置
    ENABLE_AUTH: bool = os.getenv('ENABLE_AUTH', 'False').lower() == 'true'
    
    # 交易配置
    AUTO_START_TRADING: bool = os.getenv('AUTO_START_TRADING', 'False').lower() == 'true'
    PAPER_TRADING: bool = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
    
    # API 限流配置
    RATE_LIMIT_ENABLED: bool = os.getenv('RATE_LIMIT_ENABLED', 'False').lower() == 'true'
    RATE_LIMIT_CALLS: int = int(os.getenv('RATE_LIMIT_CALLS', '100'))
    RATE_LIMIT_PERIOD: int = int(os.getenv('RATE_LIMIT_PERIOD', '60'))
    
    # WebSocket 配置
    WS_HEARTBEAT_INTERVAL: int = 30  # 秒
    WS_MAX_CONNECTIONS: int = 100
    
    # 文件上傳配置
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ['.json', '.yaml', '.yml', '.csv', '.parquet']
    
    # 路徑配置
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    OPTUNA_RESULTS_DIR: Path = PROJECT_ROOT / "optuna_system" / "results"
    
    # 日誌配置
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 性能配置
    ENABLE_PROFILING: bool = os.getenv('ENABLE_PROFILING', 'False').lower() == 'true'
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 300  # 5分鐘
    
    @classmethod
    def is_development(cls) -> bool:
        """檢查是否為開發環境"""
        return cls.ENVIRONMENT == 'development'
    
    @classmethod
    def is_production(cls) -> bool:
        """檢查是否為生產環境"""
        return cls.ENVIRONMENT == 'production'
    
    @classmethod
    def get_cors_origins(cls) -> List[str]:
        """獲取 CORS 允許的來源"""
        if cls.is_development():
            return ["*"]
        return cls.CORS_ORIGINS
    
    @classmethod
    def validate(cls):
        """驗證配置"""
        errors = []
        
        # 檢查必要的配置
        if cls.is_production() and cls.SECRET_KEY == 'your-secret-key-change-in-production':
            errors.append("生產環境必須設置 SECRET_KEY")
        
        if cls.ENABLE_AUTH and not cls.SECRET_KEY:
            errors.append("啟用認證時必須設置 SECRET_KEY")
        
        # 檢查目錄是否存在
        for dir_path in [cls.DATA_DIR, cls.LOGS_DIR, cls.RESULTS_DIR]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
        
        if errors:
            raise ValueError(f"配置驗證失敗: {', '.join(errors)}")
        
        return True

# 創建全局配置實例
api_config = APIConfig()

# 驗證配置（開發環境跳過某些檢查）
if not api_config.is_development():
    api_config.validate()

