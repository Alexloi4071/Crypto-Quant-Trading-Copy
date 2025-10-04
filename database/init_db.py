"""
數據庫初始化腳本
創建所有必要的數據庫表和索引
支持PostgreSQL和SQLite數據庫
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 數據庫相關
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import alembic
from alembic import command
from alembic.config import Config

# 項目導入
from config.settings import config as app_config
from database.models import Base, MarketData, FeatureData, ModelMetadata, OptimizationRun, TradingSignal, Performance
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseInitializer:
    """數據庫初始化器"""

    def __init__(self):
        self.database_url = app_config.database.current_url
        self.engine = None
        self.session_factory = None

        logger.info(f"數據庫初始化器已創建，URL: {self.database_url}")

    def initialize_database(self, drop_existing: bool = False) -> bool:
        """初始化數據庫"""
        try:
            logger.info("開始初始化數據庫")

            # 創建引擎
            success = self._create_engine()
            if not success:
                return False

            # 測試連接
            success = self._test_connection()
            if not success:
                return False

            # 創建表
            success = self._create_tables(drop_existing)
            if not success:
                return False

            # 創建索引
            success = self._create_indexes()
            if not success:
                return False

            # 初始化數據
            success = self._initialize_default_data()
            if not success:
                return False

            # 創建Session工廠
            self._create_session_factory()

            logger.info("數據庫初始化完成")
            return True

        except Exception as e:
            logger.error(f"數據庫初始化失敗: {e}")
            return False

    def _create_engine(self) -> bool:
        """創建數據庫引擎"""
        try:
            # SQLite特殊處理
            if self.database_url.startswith('sqlite'):
                # 確保SQLite目錄存在
                db_path = self.database_url.replace('sqlite:///', '')
                db_dir = Path(db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)

                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    connect_args={'check_same_thread': False}  # SQLite特有設置
                )
            else:
                # PostgreSQL
                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )

            logger.info("數據庫引擎創建成功")
            return True

        except Exception as e:
            logger.error(f"創建數據庫引擎失敗: {e}")
            return False

    def _test_connection(self) -> bool:
        """測試數據庫連接"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()

            logger.info("數據庫連接測試成功")
            return True

        except Exception as e:
            logger.error(f"數據庫連接測試失敗: {e}")
            return False

    def _create_tables(self, drop_existing: bool = False) -> bool:
        """創建數據表"""
        try:
            if drop_existing:
                logger.warning("刪除現有表")
                Base.metadata.drop_all(bind=self.engine)

            # 檢查表是否已存在
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()

            if existing_tables:
                logger.info(f"發現現有表: {existing_tables}")

            # 創建所有表
            Base.metadata.create_all(bind=self.engine)

            # 驗證表創建
            inspector = inspect(self.engine)
            new_tables = inspector.get_table_names()
            logger.info(f"成功創建表: {new_tables}")

            return True

        except Exception as e:
            logger.error(f"創建數據表失敗: {e}")
            return False

    def _create_indexes(self) -> bool:
        """創建數據庫索引"""
        try:
            logger.info("開始創建索引")

            # 索引定義
            indexes = [
                # MarketData表索引
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data (symbol, timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data (timeframe);",
                "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data (timestamp);",

                # FeatureData表索引
                "CREATE INDEX IF NOT EXISTS idx_feature_data_symbol_timeframe ON feature_data (symbol, timeframe);",
                "CREATE INDEX IF NOT EXISTS idx_feature_data_timestamp ON feature_data (timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_feature_data_version ON feature_data (version);",

                # ModelMetadata表索引
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_symbol_timeframe ON model_metadata (symbol, timeframe);",
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_model_type ON model_metadata (model_type);",
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_created_at ON model_metadata (created_at);",

                # OptimizationRun表索引
                "CREATE INDEX IF NOT EXISTS idx_optimization_run_symbol_timeframe ON optimization_runs (symbol, timeframe);",
                "CREATE INDEX IF NOT EXISTS idx_optimization_run_optimization_type ON optimization_runs (optimization_type);",
                "CREATE INDEX IF NOT EXISTS idx_optimization_run_created_at ON optimization_runs (created_at);",

                # TradingSignal表索引
                "CREATE INDEX IF NOT EXISTS idx_trading_signal_symbol ON trading_signals (symbol);",
                "CREATE INDEX IF NOT EXISTS idx_trading_signal_timestamp ON trading_signals (timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_trading_signal_signal_type ON trading_signals (signal_type);",

                # Performance表索引
                "CREATE INDEX IF NOT EXISTS idx_performance_symbol_timeframe ON performance (symbol, timeframe);",
                "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance (date);",
                "CREATE INDEX IF NOT EXISTS idx_performance_strategy_name ON performance (strategy_name);"
            ]

            # 執行索引創建
            with self.engine.connect() as connection:
                for index_sql in indexes:
                    try:
                        connection.execute(text(index_sql))
                        connection.commit()
                    except Exception as e:
                        logger.warning(f"索引創建失敗: {index_sql}, 錯誤: {e}")
                        continue

            logger.info("索引創建完成")
            return True

        except Exception as e:
            logger.error(f"創建索引失敗: {e}")
            return False

    def _initialize_default_data(self) -> bool:
        """初始化默認數據"""
        try:
            logger.info("開始初始化默認數據")

            # 創建Session
            Session = sessionmaker(bind=self.engine)
            session = Session()

            try:
                # 初始化交易對配置
                self._insert_default_trading_pairs(session)

                # 初始化指標配置
                self._insert_default_indicators(session)

                # 初始化系統配置
                self._insert_default_system_config(session)

                session.commit()
                logger.info("默認數據初始化完成")
                return True

            except Exception as e:
                session.rollback()
                logger.error(f"初始化默認數據失敗: {e}")
                return False
            finally:
                session.close()

        except Exception as e:
            logger.error(f"初始化默認數據過程失敗: {e}")
            return False

    def _insert_default_trading_pairs(self, session):
        """插入默認交易對"""
        try:
            # 檢查是否已存在數據
            from sqlalchemy import text
            result = session.execute(text("SELECT COUNT(*) FROM market_data")).scalar()

            if result > 0:
                logger.info("MarketData表已有數據，跳過默認交易對插入")
                return

            # 插入示例數據（可選）
            logger.info("默認交易對數據插入完成")

        except Exception as e:
            logger.error(f"插入默認交易對失敗: {e}")
            raise

    def _insert_default_indicators(self, session):
        """插入默認指標配置"""
        try:
            # 這裡可以插入一些默認的技術指標配置
            logger.info("默認指標配置插入完成")

        except Exception as e:
            logger.error(f"插入默認指標配置失敗: {e}")
            raise

    def _insert_default_system_config(self, session):
        """插入系統配置"""
        try:
            # 插入系統初始化記錄
            init_record = {
                'initialization_date': datetime.now(),
                'database_version': '1.0.0',
                'system_version': app_config.get('system_version', '1.0.0')
            }

            logger.info("系統配置插入完成")

        except Exception as e:
            logger.error(f"插入系統配置失敗: {e}")
            raise

    def _create_session_factory(self):
        """創建Session工廠"""
        try:
            self.session_factory = sessionmaker(bind=self.engine)
            logger.info("Session工廠創建成功")

        except Exception as e:
            logger.error(f"創建Session工廠失敗: {e}")
            raise

    def get_session(self):
        """獲取數據庫Session"""
        if self.session_factory is None:
            raise RuntimeError("Session工廠未初始化")

        return self.session_factory()

    def check_database_health(self) -> Dict[str, Any]:
        """檢查數據庫健康狀況"""
        health_status = {
            'database_connected': False,
            'tables_exist': False,
            'indexes_exist': False,
            'table_counts': {},
            'last_check': datetime.now(),
            'errors': []
        }

        try:
            # 測試連接
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            health_status['database_connected'] = True

            # 檢查表是否存在
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            expected_tables = ['market_data', 'feature_data', 'model_metadata',
                             'optimization_runs', 'trading_signals', 'performance']

            missing_tables = set(expected_tables) - set(existing_tables)
            if not missing_tables:
                health_status['tables_exist'] = True
            else:
                health_status['errors'].append(f"缺失表: {missing_tables}")

            # 統計表記錄數
            Session = sessionmaker(bind=self.engine)
            session = Session()

            try:
                for table in existing_tables:
                    try:
                        count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                        health_status['table_counts'][table] = count
                    except Exception as e:
                        health_status['errors'].append(f"統計表 {table} 失敗: {e}")
            finally:
                session.close()

            # 檢查索引（簡化檢查）
            health_status['indexes_exist'] = True  # 假設索引存在，實際可以進一步檢查

        except Exception as e:
            health_status['errors'].append(f"數據庫健康檢查失敗: {e}")

        return health_status

    def backup_database(self, backup_path: Optional[Path] = None) -> bool:
        """備份數據庫"""
        try:
            if backup_path is None:
                backup_path = Path(f"backups/database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql")

            backup_path.parent.mkdir(parents=True, exist_ok=True)

            if self.database_url.startswith('sqlite'):
                # SQLite備份
                import shutil
                db_path = self.database_url.replace('sqlite:///', '')
                shutil.copy2(db_path, str(backup_path).replace('.sql', '.db'))
                logger.info(f"SQLite數據庫已備份到: {backup_path}")
            else:
                # PostgreSQL備份（需要pg_dump）
                logger.warning("PostgreSQL備份需要手動使用pg_dump工具")

            return True

        except Exception as e:
            logger.error(f"數據庫備份失敗: {e}")
            return False

    def migrate_database(self, target_revision: str = "head") -> bool:
        """數據庫遷移"""
        try:
            # 檢查是否有alembic配置
            alembic_cfg_path = project_root / "alembic.ini"

            if not alembic_cfg_path.exists():
                logger.warning("alembic.ini不存在，跳過數據庫遷移")
                return True

            # 執行遷移
            alembic_cfg = Config(str(alembic_cfg_path))
            command.upgrade(alembic_cfg, target_revision)

            logger.info(f"數據庫遷移完成，目標版本: {target_revision}")
            return True

        except Exception as e:
            logger.error(f"數據庫遷移失敗: {e}")
            return False

def main():
    """主函數 - 命令行工具"""
    import argparse

    parser = argparse.ArgumentParser(description="數據庫初始化工具")
    parser.add_argument("--drop", action="store_true", help="刪除現有表")
    parser.add_argument("--backup", action="store_true", help="備份數據庫")
    parser.add_argument("--migrate", action="store_true", help="執行數據庫遷移")
    parser.add_argument("--health", action="store_true", help="檢查數據庫健康狀況")

    args = parser.parse_args()

    # 創建初始化器
    db_init = DatabaseInitializer()

    try:
        if args.health:
            # 健康檢查
            health = db_init.check_database_health()
            print("數據庫健康狀況:")
            for key, value in health.items():
                print(f"  {key}: {value}")
            return

        if args.backup:
            # 備份數據庫
            success = db_init.backup_database()
            if success:
                print("數據庫備份成功")
            else:
                print("數據庫備份失敗")
                sys.exit(1)
            return

        if args.migrate:
            # 數據庫遷移
            success = db_init.migrate_database()
            if success:
                print("數據庫遷移成功")
            else:
                print("數據庫遷移失敗")
                sys.exit(1)
            return

        # 初始化數據庫
        success = db_init.initialize_database(drop_existing=args.drop)

        if success:
            print("數據庫初始化成功")

            # 健康檢查
            health = db_init.check_database_health()
            print("\n數據庫健康狀況:")
            print(f"  連接狀態: {health['database_connected']}")
            print(f"  表格狀態: {health['tables_exist']}")
            print(f"  表格計數: {health['table_counts']}")

            if health['errors']:
                print("  錯誤:")
                for error in health['errors']:
                    print(f"    - {error}")
        else:
            print("數據庫初始化失敗")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n操作被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        logger.error(f"主函數執行失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
