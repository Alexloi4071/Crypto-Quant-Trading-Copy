#!/usr/bin/env python3
"""
Trading System Web Server Launcher
交易系统Web服务器启动脚本
集成所有组件，提供完整的Web服务
"""

import sys
import os
import asyncio
import signal
from pathlib import Path
import argparse
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # 导入API主模块
    from api.main import main, trading_api
    from api.config import api_config, validate_config, get_config_summary
    from src.utils.logger import setup_logger
    from config.settings import setup_config
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有依赖包已安装：pip install -r requirements.txt")
    sys.exit(1)

# 设置日志
logger = setup_logger(__name__)

class TradingSystemWebServer:
    """交易系统Web服务器"""
    
    def __init__(self):
        self.server_process = None
        self.is_running = False
        
    async def start_server(self, host="0.0.0.0", port=8080, debug=False):
        """启动Web服务器"""
        try:
            logger.info("🚀 启动 Trading System Web Server")
            
            # 验证配置
            try:
                validate_config()
                logger.info("✅ 配置验证通过")
            except Exception as e:
                logger.warning(f"⚠️ 配置验证警告: {e}")
            
            # 显示配置摘要
            config_summary = get_config_summary()
            logger.info("📊 服务器配置摘要:")
            logger.info(f"   - 服务地址: {host}:{port}")
            logger.info(f"   - 调试模式: {debug}")
            logger.info(f"   - 认证启用: {config_summary['features']['authentication']}")
            logger.info(f"   - WebSocket: {config_summary['features']['websocket']}")
            logger.info(f"   - 自动交易: {config_summary['features']['auto_trading']}")
            logger.info(f"   - 数据库: {config_summary['integration']['database']}")
            logger.info(f"   - 交易对: {', '.join(config_summary['integration']['symbols'][:3])}{'...' if len(config_summary['integration']['symbols']) > 3 else ''}")
            
            # 启动服务器
            import uvicorn
            
            uvicorn_config = {
                'host': host,
                'port': port,
                'debug': debug,
                'reload': debug,
                'log_level': 'debug' if debug else 'info',
                'access_log': True,
                'workers': 1  # API应用已经是异步的，单worker即可
            }
            
            if debug:
                uvicorn_config['reload_dirs'] = [str(project_root)]
                uvicorn_config['reload_includes'] = ['*.py', '*.html', '*.js', '*.css']
            
            logger.info("🌐 访问地址:")
            logger.info(f"   - 主页面: http://{host}:{port}")
            logger.info(f"   - API文档: http://{host}:{port}/docs")
            logger.info(f"   - 健康检查: http://{host}:{port}/api/v1/health")
            logger.info(f"   - WebSocket: ws://{host}:{port}/ws")
            
            # 显示认证信息（如果启用）
            if config_summary['features']['authentication']:
                logger.info("🔐 认证信息 (开发环境):")
                logger.info("   - 管理员: admin / admin123")
                logger.info("   - 交易员: trader / trader123") 
                logger.info("   - 观察员: viewer / viewer123")
            
            self.is_running = True
            
            # 启动uvicorn服务器
            await uvicorn.run("api.main:app", **uvicorn_config)
            
        except KeyboardInterrupt:
            logger.info("⏸️ 收到停止信号")
            await self.shutdown()
        except Exception as e:
            logger.error(f"❌ 服务器启动失败: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """关闭服务器"""
        if not self.is_running:
            return
        
        logger.info("🛑 正在关闭服务器...")
        self.is_running = False
        
        try:
            # 关闭交易系统组件
            if trading_api:
                await trading_api.shutdown_systems()
            
            logger.info("✅ 服务器已安全关闭")
        except Exception as e:
            logger.error(f"❌ 关闭服务器时出现错误: {e}")
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(sig, frame):
            logger.info(f"收到信号 {sig}，准备关闭...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Trading System Web Server')
    
    parser.add_argument(
        '--host', 
        default=api_config.HOST,
        help=f'服务器主机地址 (默认: {api_config.HOST})'
    )
    
    parser.add_argument(
        '--port', 
        type=int,
        default=api_config.PORT,
        help=f'服务器端口 (默认: {api_config.PORT})'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        default=api_config.DEBUG,
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--no-auth',
        action='store_true',
        help='禁用认证系统'
    )
    
    parser.add_argument(
        '--auto-trading',
        action='store_true',
        help='启用自动交易'
    )
    
    parser.add_argument(
        '--paper-trading',
        action='store_true',
        default=True,
        help='启用纸交易模式'
    )
    
    parser.add_argument(
        '--config-check',
        action='store_true',
        help='仅检查配置并退出'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Trading System Web Server 1.0.0'
    )
    
    return parser.parse_args()

def check_dependencies():
    """检查依赖"""
    required_packages = [
        'fastapi', 'uvicorn', 'websockets', 'pandas', 'numpy',
        'scikit-learn', 'lightgbm', 'xgboost', 'redis', 'sqlalchemy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_configuration():
    """检查配置"""
    logger.info("🔍 检查系统配置...")
    
    # 检查必要的目录
    required_dirs = [
        project_root / "data",
        project_root / "models", 
        project_root / "logs",
        project_root / "config"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.info(f"📁 创建目录: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # 检查配置文件
    config_file = project_root / "config" / "trading_config.yaml"
    if not config_file.exists():
        logger.warning(f"⚠️ 配置文件不存在: {config_file}")
        logger.info("系统将使用默认配置运行")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Trading System Web Server")
    print("=" * 60)
    
    # 解析参数
    args = parse_arguments()
    
    # 仅检查配置
    if args.config_check:
        try:
            setup_config()
            validate_config()
            config_summary = get_config_summary()
            
            print("✅ 配置检查通过")
            print("\n📊 配置摘要:")
            for category, items in config_summary.items():
                print(f"  {category}:")
                for key, value in items.items():
                    print(f"    {key}: {value}")
            
        except Exception as e:
            print(f"❌ 配置检查失败: {e}")
            return 1
        
        return 0
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 检查配置
    if not check_configuration():
        return 1
    
    # 设置环境变量
    if args.no_auth:
        os.environ['ENABLE_AUTH'] = 'false'
    
    if args.auto_trading:
        os.environ['AUTO_START_TRADING'] = 'true'
    
    if args.paper_trading:
        os.environ['PAPER_TRADING'] = 'true'
    
    # 设置调试模式
    if args.debug:
        os.environ['API_DEBUG'] = 'true'
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 初始化配置
        setup_config()
        
        # 创建Web服务器实例
        web_server = TradingSystemWebServer()
        web_server.setup_signal_handlers()
        
        # 启动服务器
        asyncio.run(web_server.start_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        ))
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️ 用户中断，服务器停止")
        return 0
    except Exception as e:
        print(f"\n❌ 服务器运行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)