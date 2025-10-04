#!/usr/bin/env python3
"""
Setup script for the Crypto Quant Trading System
Handles installation, dependencies, and configuration
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read version from file
def get_version():
    version_file = Path(__file__).parent / "src" / "__version__.py"
    if version_file.exists():
        with open(version_file) as f:
            exec(f.read())
            return locals()['__version__']
    return "1.0.0"

# Read README for long description
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def get_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Development requirements
dev_requirements = [
    'pytest>=7.4.0',
    'pytest-asyncio>=0.21.0',
    'pytest-cov>=4.1.0',
    'pytest-mock>=3.11.0',
    'black>=23.7.0',
    'flake8>=6.0.0',
    'mypy>=1.5.0',
    'pre-commit>=3.4.0',
    'jupyter>=1.0.0',
    'ipython>=8.15.0',
]

# Documentation requirements
docs_requirements = [
    'sphinx>=7.0.0',
    'sphinx-rtd-theme>=1.3.0',
    'myst-parser>=2.0.0',
]

# Production deployment requirements
deploy_requirements = [
    'docker>=6.1.0',
    'gunicorn>=21.2.0',
    'supervisor>=4.2.5',
]

setup(
    name="crypto-quant-trading",
    version=get_version(),
    author="Crypto Quant Trading Team",
    author_email="team@cryptoquant.trading",
    description="Advanced multi-currency, multi-timeframe quantitative trading system for cryptocurrencies",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/crypto-quant/trading-system",
    project_urls={
        "Bug Tracker": "https://github.com/crypto-quant/trading-system/issues",
        "Documentation": "https://crypto-quant-trading.readthedocs.io/",
        "Source Code": "https://github.com/crypto-quant/trading-system",
    },
    
    # Package configuration
    packages=find_packages(include=['src', 'src.*', 'config', 'config.*']),
    package_dir={'': '.'},
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.txt', '*.md'],
        'config': ['*.yaml', '*.yml'],
        'src': ['*.yaml', '*.yml'],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=get_requirements(),
    extras_require={
        'dev': dev_requirements,
        'docs': docs_requirements,
        'deploy': deploy_requirements,
        'full': dev_requirements + docs_requirements + deploy_requirements,
    },
    
    # Entry points for command line tools
    entry_points={
        'console_scripts': [
            'crypto-trading=src.main:main',
            'crypto-setup=scripts.setup_database:main',
            'crypto-download=scripts.data_downloader:main',
            'crypto-train=scripts.model_trainer:main',
            'crypto-backtest=scripts.backtesting_runner:main',
            'crypto-optimize=scripts.optimizer:main',
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    
    # Keywords for package discovery
    keywords=[
        "cryptocurrency", "trading", "quantitative", "machine learning",
        "algorithmic trading", "bitcoin", "ethereum", "binance",
        "technical analysis", "backtesting", "portfolio management",
        "risk management", "time series", "financial modeling"
    ],
    
    # Zip safety
    zip_safe=False,
    
    # Custom commands
    cmdclass={},
    
    # Platform specific
    platforms=["any"],
)

# Post-installation setup
def post_install():
    """Run post-installation setup tasks"""
    print("🚀 Crypto Quant Trading System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9 or higher is required")
        sys.exit(1)
    
    print("✅ Python version check passed")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/backups",
        "logs/trading",
        "logs/optimization",
        "logs/system",
        "results/optimization",
        "results/backtesting",
        "results/models",
        "results/reports",
        "versions",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")
    
    # Check for required system dependencies
    try:
        import talib
        print("✅ TA-Lib found")
    except ImportError:
        print("⚠️  TA-Lib not found. Please install it:")
        print("   Ubuntu/Debian: sudo apt-get install ta-lib")
        print("   macOS: brew install ta-lib")
        print("   Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
    
    # Check optional dependencies
    optional_deps = [
        ('tensorflow', 'TensorFlow (for LSTM models)'),
        ('torch', 'PyTorch (for advanced models)'),
        ('imblearn', 'Imbalanced-learn (for SMOTE)'),
    ]
    
    for dep, description in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {description} found")
        except ImportError:
            print(f"⚠️  {description} not found (optional)")
    
    print("\n📋 Next Steps:")
    print("1. Copy .env.example to .env and configure your API keys")
    print("2. Run: crypto-setup --mode dev (for development setup)")
    print("3. Run: crypto-download --symbols BTCUSDT ETHUSDT (to download data)")
    print("4. Run: crypto-trading pipeline --symbols BTCUSDT (to start full pipeline)")
    print("\n📖 Documentation: https://crypto-quant-trading.readthedocs.io/")
    print("🐛 Issues: https://github.com/crypto-quant/trading-system/issues")

if __name__ == "__main__":
    # Run post-installation setup if called directly
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'develop']:
        post_install()

# Custom installation notes
INSTALL_NOTES = """
Crypto Quant Trading System Installation
========================================

Prerequisites:
--------------
1. Python 3.9 or higher
2. TA-Lib library (for technical indicators)
3. PostgreSQL (for production) or SQLite (for development)

Installation Options:
--------------------

# Basic installation
pip install crypto-quant-trading

# Development installation
pip install crypto-quant-trading[dev]

# Full installation with all optional dependencies
pip install crypto-quant-trading[full]

# From source
git clone https://github.com/crypto-quant/trading-system.git
cd trading-system
pip install -e .

Configuration:
--------------
1. Copy .env.example to .env
2. Edit .env with your API keys and configuration
3. Run initial setup: crypto-setup --mode dev

Quick Start:
-----------
1. Download data: crypto-download --symbols BTCUSDT ETHUSDT
2. Run pipeline: crypto-trading pipeline --symbols BTCUSDT
3. Start trading: crypto-trading trade --symbols BTCUSDT --mode simulation

Environment Variables:
---------------------
Required:
- BINANCE_API_KEY: Your Binance API key
- BINANCE_API_SECRET: Your Binance API secret
- TELEGRAM_TOKEN: Telegram bot token for notifications
- TELEGRAM_CHAT_ID: Your Telegram chat ID

Optional:
- DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD: Database configuration
- CRYPTOCOMPARE_API_KEY: CryptoCompare API key for on-chain data
- TWITTER_BEARER_TOKEN: Twitter API for sentiment data

Database Setup:
--------------
Development (SQLite):
- Automatic setup with crypto-setup --mode dev

Production (PostgreSQL + TimescaleDB):
- Install PostgreSQL and TimescaleDB
- Create database: createdb crypto_trading
- Enable TimescaleDB: psql crypto_trading -c "CREATE EXTENSION timescaledb;"
- Run setup: crypto-setup --mode prod

Docker Deployment:
-----------------
docker-compose up -d

This will start:
- PostgreSQL + TimescaleDB
- Redis
- Main trading application
- Celery workers
- Monitoring stack (Prometheus, Grafana)

For detailed documentation, visit:
https://crypto-quant-trading.readthedocs.io/
"""

# Print installation notes if setup.py is run directly
if __name__ == "__main__" and len(sys.argv) == 1:
    print(INSTALL_NOTES)