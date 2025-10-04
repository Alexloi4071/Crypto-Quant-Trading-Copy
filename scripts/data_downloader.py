#!/usr/bin/env python3
"""
Data Download Script
Automated data downloading for cryptocurrency trading pairs
Supports multiple exchanges and timeframes with error handling
"""

import asyncio
import argparse
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.data.collector import BinanceDataCollector, DataCollectionOrchestrator
from src.data.external_apis import ExternalDataManager
from src.data.preprocessor import DataPreprocessor
from src.data.storage import DataStorage

logger = setup_logger(__name__)


class DataDownloader:
    """Automated data downloading system"""


    def __init__(self):
        self.data_manager = None
        self.data_orchestrator = None  # 🔧 新增：支持自動重採樣
        self.external_apis = None
        self.preprocessor = None
        self.storage = None

        # Download statistics
        self.stats = {
            'symbols_processed': 0,
            'timeframes_processed': 0,
            'total_records_downloaded': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'start_time': None,
            'end_time': None
        }

        logger.info("Data downloader initialized")


    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing data downloader components")

            # Initialize components
            self.data_manager = BinanceDataCollector()
            self.data_orchestrator = DataCollectionOrchestrator()  # 🔧 新增：支持自動重採樣
            self.external_apis = ExternalDataManager()
            self.preprocessor = DataPreprocessor()
            self.storage = DataStorage()

            # Initialize connections
            # BinanceDataCollector initializes in constructor
            # success = await self.data_manager.initialize()
            # if not success:
            #     logger.error("Failed to initialize data manager")
            #     return False
            logger.info("Data manager already initialized")

            api_success = await self.external_apis.initialize()
            if not api_success:
                logger.warning("External APIs initialization failed - some features may be limited")

            storage_success = await self.storage.initialize()
            if not storage_success:
                logger.error("Failed to initialize storage")
                return False

            logger.info("Data downloader initialization completed")
            return True

        except Exception as e:
            logger.error(f"Data downloader initialization failed: {e}")
            return False


    async def download_historical_data(self, symbols: List[str],
                                     timeframes: List[str],
                                     days_back: int = 90,
                                     exchange: str = 'binance',
                                     preprocess: bool = True,
                                     include_external: bool = True) -> Dict[str, Any]:
        """Download historical OHLCV data"""
        try:
            logger.info(f"Starting historical data download for {len(symbols)} symbols")
            self.stats['start_time'] = datetime.now()

            results = {
                'success': True,
                'downloaded_data': {},
                'failed_downloads': [],
                'summary': {}
            }

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Calculate total combinations for progress tracking
            total_combinations = len(symbols) * len(timeframes)
            current_combination = 0

            print("🚀 開始批量數據下載任務")
            print(f"   📊 交易對數量: {len(symbols)}")
            print(f"   ⏰ 時間框架: {len(timeframes)} 個")
            print(f"   📅 時間範圍: {days_back} 天 ({start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')})")
            print(f"   🎯 總任務數: {total_combinations} 個下載任務")
            print("-" * 60)

            # Download data for each symbol/timeframe combination
            for symbol in symbols:
                self.stats['symbols_processed'] += 1
                symbol_results = {}

                for timeframe in timeframes:
                    self.stats['timeframes_processed'] += 1
                    current_combination += 1

                    # Display overall progress
                    overall_progress = (current_combination / total_combinations) * 100
                    print(f"\n🎯 總進度: {overall_progress:.1f}% | 任務 {current_combination}/{total_combinations}")
                    print(f"📊 當前處理: {symbol} {timeframe}")

                    try:
                        # Download OHLCV data
                        df = await self._download_ohlcv_data(
                            symbol, timeframe, start_date, end_date, exchange
                        )

                        if df.empty:
                            print(f"   ⚠️ 沒有獲取到 {symbol}_{timeframe} 的數據")
                            logger.warning(f"No data retrieved for {symbol}_{timeframe}")
                            self.stats['failed_downloads'] += 1
                            results['failed_downloads'].append(f"{symbol}_{timeframe}")
                            continue

                        # Preprocess if requested
                        if preprocess:
                            print("   🔄 數據預處理中...")
                            df = self._preprocess_data(df, symbol, timeframe)

                        # Store data
                        print("   💾 數據存儲中...")
                        success = await self.storage.store_ohlcv_data(symbol, timeframe, df, exchange)

                        if success:
                            self.stats['successful_downloads'] += 1
                            self.stats['total_records_downloaded'] += len(df)
                            symbol_results[timeframe] = {
                                'records': len(df),
                                'date_range': {
                                    'start': df.index.min().isoformat(),
                                    'end': df.index.max().isoformat()
                                }
                            }
                            print(f"   ✅ 成功: {len(df):,} 條記錄已保存")
                            logger.info(f"Downloaded {len(df)} records for {symbol}_{timeframe}")
                        else:
                            print(f"   ❌ 存儲失敗: {symbol}_{timeframe}")
                            logger.error(f"Failed to store data for {symbol}_{timeframe}")
                            self.stats['failed_downloads'] += 1
                            results['failed_downloads'].append(f"{symbol}_{timeframe}")

                    except Exception as e:
                        logger.error(f"Download failed for {symbol}_{timeframe}: {e}")
                        self.stats['failed_downloads'] += 1
                        results['failed_downloads'].append(f"{symbol}_{timeframe}")
                        continue

                    # Rate limiting
                    await asyncio.sleep(0.5)

                # Store symbol results
                if symbol_results:
                    results['downloaded_data'][symbol] = symbol_results

                # Download external data if requested
                if include_external:
                    await self._download_external_data(symbol)

            # Finalize statistics
            self.stats['end_time'] = datetime.now()

            # Generate summary
            results['summary'] = self._generate_download_summary()

            logger.info(f"Historical data download completed: {self.stats['successful_downloads']} successful, {self.stats['failed_downloads']} failed")

            return results

        except Exception as e:
            logger.error(f"Historical data download failed: {e}")
            return {'success': False, 'error': str(e)}


    async def _download_ohlcv_data(self, symbol: str, timeframe: str,
                                 start_date: datetime, end_date: datetime,
                                 exchange: str) -> Any:
        """Download OHLCV data for a specific symbol/timeframe"""
        try:
            # Use data manager to fetch data
            df = await self.data_manager.download_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            return df

        except Exception as e:
            logger.error(f"OHLCV download failed for {symbol}_{timeframe}: {e}")
            raise


    def _preprocess_data(self, df: Any, symbol: str, timeframe: str) -> Any:
        """Preprocess downloaded data"""
        try:
            # Basic preprocessing
            cleaned_df = self.preprocessor.preprocess_ohlcv_data(df)

            logger.debug(f"Preprocessed {symbol}_{timeframe}: {len(df)} -> {len(cleaned_df)} records")

            return cleaned_df

        except Exception as e:
            logger.error(f"Data preprocessing failed for {symbol}_{timeframe}: {e}")
            return df  # Return original data if preprocessing fails


    async def _download_external_data(self, symbol: str):
        """Download external data for symbol"""
        try:
            if not self.external_apis:
                return

            # Market data
            market_data = await self.external_apis.get_market_data(symbol)
            if market_data:
                logger.debug(f"Downloaded market data for {symbol}")

            # News sentiment
            sentiment_data = await self.external_apis.get_news_sentiment(symbol)
            if sentiment_data:
                logger.debug(f"Downloaded sentiment data for {symbol}")

            # Small delay for rate limiting
            await asyncio.sleep(1)

        except Exception as e:
            logger.debug(f"External data download failed for {symbol}: {e}")


    def _generate_download_summary(self) -> Dict[str, Any]:
        """Generate download summary"""
        try:
            duration = None
            if self.stats['start_time'] and self.stats['end_time']:
                duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            success_rate = 0
            total_attempts = self.stats['successful_downloads'] + self.stats['failed_downloads']
            if total_attempts > 0:
                success_rate = (self.stats['successful_downloads'] / total_attempts) * 100

            return {
                'total_symbols': self.stats['symbols_processed'],
                'total_timeframes': self.stats['timeframes_processed'],
                'total_records': self.stats['total_records_downloaded'],
                'successful_downloads': self.stats['successful_downloads'],
                'failed_downloads': self.stats['failed_downloads'],
                'success_rate_percent': round(success_rate, 1),
                'duration_seconds': round(duration, 1) if duration else None,
                'records_per_second': round(self.stats['total_records_downloaded'] / duration, 1) if duration and duration > 0 else None
            }

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {}


    async def download_specific_range(self, symbol: str, timeframe: str,
                                    start_date: str, end_date: str,
                                    exchange: str = 'binance') -> Dict[str, Any]:
        """Download data for specific date range"""
        try:
            logger.info(f"Downloading specific range for {symbol}_{timeframe}: {start_date} to {end_date}")

            # Parse dates (validation only)
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')

            # Download data
            df = await self.data_manager.download_historical_data(symbol, timeframe, start_date, end_date)

            if df.empty:
                return {'success': False, 'message': 'No data retrieved'}

            # Preprocess
            cleaned_df = self._preprocess_data(df, symbol, timeframe)

            # Store
            success = await self.storage.store_ohlcv_data(symbol, timeframe, cleaned_df, exchange)

            if success:
                return {
                    'success': True,
                    'records': len(cleaned_df),
                    'date_range': {
                        'start': cleaned_df.index.min().isoformat(),
                        'end': cleaned_df.index.max().isoformat()
                    }
                }
            else:
                return {'success': False, 'message': 'Failed to store data'}

        except Exception as e:
            logger.error(f"Specific range download failed: {e}")
            return {'success': False, 'error': str(e)}

    async def download_with_auto_resample(self, symbol: str, start_date: str, 
                                        end_date: str, save_to_db: bool = True) -> Dict[str, Any]:
        """🔧 新功能：下載1m數據並自動重採樣到多時間框架"""
        try:
            logger.info(f"🔧 開始自動重採樣下載 {symbol}: {start_date} 到 {end_date}")
            
            # 使用DataCollectionOrchestrator進行完整的歷史數據收集（包含自動重採樣）
            datasets = await self.data_orchestrator.collect_historical_data(
                symbol, start_date, end_date, save_to_db
            )
            
            if not datasets:
                return {'success': False, 'message': 'No datasets retrieved'}
            
            # 統計結果
            total_records = sum(len(df) for df in datasets.values())
            timeframes_processed = list(datasets.keys())
            
            logger.info(f"✅ {symbol} 自動重採樣完成: {len(timeframes_processed)} 個時間框架, {total_records} 條總記錄")
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframes_processed': timeframes_processed,
                'total_records': total_records,
                'datasets_info': {tf: len(df) for tf, df in datasets.items()},
                'date_range': {
                    'start': start_date,
                    'end': end_date
                }
            }
            
        except Exception as e:
            logger.error(f"Auto-resample download failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}


    async def update_recent_data(self, symbols: List[str],
                               timeframes: List[str],
                               hours_back: int = 24) -> Dict[str, Any]:
        """Update with recent data"""
        try:
            logger.info(f"Updating recent data ({hours_back} hours back)")

            # Calculate time range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)

            results = {
                'success': True,
                'updated_data': {},
                'failed_updates': []
            }

            for symbol in symbols:
                symbol_results = {}

                for timeframe in timeframes:
                    try:
                        # Download recent data
                        df = await self._download_ohlcv_data(symbol, timeframe, start_date, end_date, 'binance')

                        if not df.empty:
                            # Preprocess
                            cleaned_df = self._preprocess_data(df, symbol, timeframe)

                            # Store (will merge with existing data)
                            success = await self.storage.store_ohlcv_data(symbol, timeframe, cleaned_df, 'binance')

                            if success:
                                symbol_results[timeframe] = len(cleaned_df)
                                logger.debug(f"Updated {len(cleaned_df)} records for {symbol}_{timeframe}")
                            else:
                                results['failed_updates'].append(f"{symbol}_{timeframe}")
                        else:
                            results['failed_updates'].append(f"{symbol}_{timeframe}")

                    except Exception as e:
                        logger.error(f"Recent data update failed for {symbol}_{timeframe}: {e}")
                        results['failed_updates'].append(f"{symbol}_{timeframe}")
                        continue

                    await asyncio.sleep(0.2)  # Rate limiting

                if symbol_results:
                    results['updated_data'][symbol] = symbol_results

            logger.info(f"Recent data update completed: {len(results['updated_data'])} symbols updated")

            return results

        except Exception as e:
            logger.error(f"Recent data update failed: {e}")
            return {'success': False, 'error': str(e)}


    async def verify_data_integrity(self, symbols: List[str],
                                  timeframes: List[str]) -> Dict[str, Any]:
        """Verify integrity of downloaded data"""
        try:
            logger.info("Verifying data integrity")

            verification_results = {
                'success': True,
                'verified_data': {},
                'issues_found': []
            }

            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Load data
                        df = await self.storage.load_ohlcv_data(symbol, timeframe)

                        if df.empty:
                            verification_results['issues_found'].append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'issue': 'No data found'
                            })
                            continue

                        # Run data quality checks
                        quality_report = self.preprocessor.validate_data_quality(df)

                        verification_results['verified_data'][f"{symbol}_{timeframe}"] = {
                            'records': len(df),
                            'date_range': {
                                'start': df.index.min().isoformat(),
                                'end': df.index.max().isoformat()
                            },
                            'quality_score': quality_report.get('quality_score', 0),
                            'issues': quality_report.get('issues', [])
                        }

                        # Check for critical issues
                        if quality_report.get('quality_score', 0) < 50:
                            verification_results['issues_found'].append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'issue': f"Low quality score: {quality_report.get('quality_score', 0):.1f}",
                                'details': quality_report.get('issues', [])
                            })

                    except Exception as e:
                        verification_results['issues_found'].append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'issue': f"Verification failed: {e}"
                        })

            logger.info(f"Data integrity verification completed: {len(verification_results['issues_found'])} issues found")

            return verification_results

        except Exception as e:
            logger.error(f"Data integrity verification failed: {e}")
            return {'success': False, 'error': str(e)}


    def get_download_statistics(self) -> Dict[str, Any]:
        """Get download statistics"""
        return self.stats.copy()


    async def resample_data(self, symbols: List[str], target_timeframes: List[str]) -> Dict[str, Any]:
        """Resample existing 1m data to specified timeframes"""
        try:
            logger.info(f"Starting data resampling for {len(symbols)} symbols to {target_timeframes}")
            self.stats['start_time'] = datetime.now()

            from src.utils.helpers import resample_ohlcv
            from src.data.preprocessor import DataPreprocessor
            from pathlib import Path
            import pandas as pd

            results = {
                'success': True,
                'resampled_data': {},
                'failed_resamples': [],
                'summary': {}
            }

            print("🔄 開始數據重採樣任務")
            print(f"   💰 交易對: {', '.join(symbols)}")
            print(f"   ⏰ 目標時框: {', '.join(target_timeframes)}")
            print("-" * 60)

            total_operations = len(symbols) * len(target_timeframes)
            current_operation = 0
            successful_operations = 0

            for symbol in symbols:
                # Load 1m data from database
                try:
                    print(f"\n📊 處理 {symbol}...")

                    # Load 1m data from parquet file
                    data_path = f"data/raw/{symbol}_1m_ohlcv.parquet"
                    alt_path = f"data/raw/{symbol}/{symbol}_1m_ohlcv.parquet"
                    
                    df_1m = pd.DataFrame()
                    try:
                        df_1m = pd.read_parquet(data_path)
                        print(f"   ✅ 從主目錄載入: {data_path}")
                    except FileNotFoundError:
                        try:
                            df_1m = pd.read_parquet(alt_path)
                            print(f"   ✅ 從符號目錄載入: {alt_path}")
                        except FileNotFoundError:
                            print(f"   ❌ 沒有找到 {symbol} 的1m數據文件")
                            print(f"   查找路径: {data_path} 或 {alt_path}")
                            for tf in target_timeframes:
                                results['failed_resamples'].append(f"{symbol}_{tf}")
                            continue

                    if df_1m.empty:
                        print(f"   ❌ {symbol} 的1m數據為空")
                        for tf in target_timeframes:
                            results['failed_resamples'].append(f"{symbol}_{tf}")
                        continue

                    print(f"   ✅ 載入1m數據: {len(df_1m):,} 條記錄")
                    print(f"   📅 時間範圍: {df_1m.index.min()} 到 {df_1m.index.max()}")

                    symbol_results = {}

                    for timeframe in target_timeframes:
                        current_operation += 1

                        try:
                            print(f"\n   🔄 重採樣到 {timeframe}...")

                            # Resample data using updated helper function
                            resampled_df = resample_ohlcv(df_1m, timeframe)

                            if resampled_df.empty:
                                print("      ❌ 重採樣失敗: 結果為空")
                                results['failed_resamples'].append(f"{symbol}_{timeframe}")
                                continue

                            # Apply data cleaning using updated preprocessor
                            preprocessor = DataPreprocessor()
                            cleaned_df = preprocessor.preprocess_ohlcv_data(resampled_df, {
                                'remove_duplicates': True,
                                'handle_missing_values': True,
                                'detect_outliers': True,
                                'validate_ohlc': True,
                                'remove_zero_volume': False,  # 保留零成交量
                                'max_short_gap': 3,
                                'imputation_method': 'forward_fill'
                            })
                            
                            print(f"      🧹 清洗後: {len(cleaned_df):,} 條記錄")

                            # Create organized directory structure
                            timeframe_dir = Path(f"data/raw/{symbol}")
                            timeframe_dir.mkdir(parents=True, exist_ok=True)

                            # Save cleaned data to parquet file
                            parquet_path = timeframe_dir / f"{symbol}_{timeframe}_ohlcv.parquet"
                            cleaned_df.to_parquet(parquet_path)

                            # Also save to main raw directory for easy access
                            main_parquet = Path(f"data/raw/{symbol}_{timeframe}_ohlcv.parquet")
                            cleaned_df.to_parquet(main_parquet)

                            symbol_results[timeframe] = {
                                'records': len(cleaned_df),
                                'date_range': {
                                    'start': cleaned_df.index.min().isoformat(),
                                    'end': cleaned_df.index.max().isoformat()
                                },
                                'file_path': str(parquet_path)
                            }

                            successful_operations += 1
                            print(f"      ✅ 成功: {len(cleaned_df):,} 條記錄")
                            print(f"      💾 已保存: {parquet_path}")
                            print(f"      📈 進度: {current_operation}/{total_operations} ({current_operation/total_operations*100:.1f}%)")

                        except Exception as e:
                            print(f"      ❌ 重採樣失敗: {e}")
                            results['failed_resamples'].append(f"{symbol}_{timeframe}")

                    if symbol_results:
                        results['resampled_data'][symbol] = symbol_results

                except Exception as e:
                    print(f"   ❌ 處理 {symbol} 失敗: {e}")
                    for tf in target_timeframes:
                        results['failed_resamples'].append(f"{symbol}_{tf}")

            # Generate summary
            self.stats['end_time'] = datetime.now()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            results['summary'] = {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'failed_operations': len(results['failed_resamples']),
                'success_rate_percent': (successful_operations / total_operations * 100) if total_operations > 0 else 0,
                'duration_seconds': duration,
                'symbols_processed': len(symbols),
                'target_timeframes': target_timeframes
            }

            print("\n" + "=" * 60)
            print("🎉 重採樣完成！")
            print(f"   成功: {successful_operations}/{total_operations} ({successful_operations/total_operations*100:.1f}%)")
            print(f"   耗時: {duration:.1f} 秒")
            print("   文件位置: data/raw/[SYMBOL]/")

            if results['failed_resamples']:
                print(f"   ⚠️ 失敗的操作: {len(results['failed_resamples'])}")
                for failed in results['failed_resamples'][:5]:
                    print(f"     - {failed}")

            logger.info(f"Data resampling completed: {successful_operations} successful, {len(results['failed_resamples'])} failed")

            return results

        except Exception as e:
            logger.error(f"Data resampling failed: {e}")
            return {'success': False, 'error': str(e)}


async def main():
    """Main function for data download script"""
    parser = argparse.ArgumentParser(description='Data Download Script')

    # Command selection
    parser.add_argument('command', choices=[
        'download', 'update', 'range', 'verify', 'resample'
    ], help='Download command')

    # Symbol and timeframe selection
    parser.add_argument('--symbols', nargs='+',
                       default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'],
                       help='Trading symbols to download')

    parser.add_argument('--timeframes', nargs='+',
                       default=['1h', '4h', '1d'],
                       help='Timeframes to download')

    # Time range options
    parser.add_argument('--days-back', type=int, default=90,
                       help='Days of historical data (for download command)')

    parser.add_argument('--hours-back', type=int, default=24,
                       help='Hours of recent data (for update command)')

    parser.add_argument('--start-date',
                       help='Start date for range download (YYYY-MM-DD)')

    parser.add_argument('--end-date',
                       help='End date for range download (YYYY-MM-DD)')

    # Options
    parser.add_argument('--exchange', default='binance',
                       help='Exchange to download from')

    parser.add_argument('--no-preprocess', action='store_true',
                       help='Skip data preprocessing')

    parser.add_argument('--no-external', action='store_true',
                       help='Skip external data download')

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

    # Initialize downloader
    downloader = DataDownloader()

    if not await downloader.initialize():
        logger.error("Failed to initialize data downloader")
        return False

    try:
        results = None

        if args.command == 'download':
            # Download historical data
            results = await downloader.download_historical_data(
                symbols=args.symbols,
                timeframes=args.timeframes,
                days_back=args.days_back,
                exchange=args.exchange,
                preprocess=not args.no_preprocess,
                include_external=not args.no_external
            )

        elif args.command == 'update':
            # Update recent data
            results = await downloader.update_recent_data(
                symbols=args.symbols,
                timeframes=args.timeframes,
                hours_back=args.hours_back
            )

        elif args.command == 'range':
            # Download specific range with auto-resample
            if not args.start_date or not args.end_date:
                logger.error("Range download requires --start-date and --end-date")
                return False

            if len(args.symbols) != 1:
                logger.error("Range download works with single symbol only")
                return False

            # 🔧 使用新的自動重採樣功能，如果只指定1m則自動重採樣到多時間框架
            if len(args.timeframes) == 1 and args.timeframes[0] == '1m':
                logger.info(f"🔧 使用自動重採樣模式，下載 {args.symbols[0]} 1m 數據並重採樣到多時間框架")
                results = await downloader.download_with_auto_resample(
                    symbol=args.symbols[0],
                    start_date=args.start_date,
                    end_date=args.end_date,
                    save_to_db=True
                )
            else:
                # 原有的單時間框架下載邏輯
                if len(args.timeframes) != 1:
                    logger.error("Range download requires single timeframe (use '1m' for auto-resample)")
                    return False
                    
                results = await downloader.download_specific_range(
                    symbol=args.symbols[0],
                    timeframe=args.timeframes[0],
                    start_date=args.start_date,
                    end_date=args.end_date,
                    exchange=args.exchange
                )

        elif args.command == 'verify':
            # Verify data integrity
            results = await downloader.verify_data_integrity(
                symbols=args.symbols,
                timeframes=args.timeframes
            )

        elif args.command == 'resample':
            # Resample existing 1m data to specified timeframes
            results = await downloader.resample_data(
                symbols=args.symbols,
                target_timeframes=args.timeframes
            )

        # Output results
        if results:
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {args.output}")
            else:
                # Print summary
                if results.get('summary'):
                    summary = results['summary']
                    print("Download Summary:")
                    print(f"  Success Rate: {summary.get('success_rate_percent', 0):.1f}%")
                    print(f"  Total Records: {summary.get('total_records', 0):,}")
                    print(f"  Duration: {summary.get('duration_seconds', 0):.1f}s")

                    if summary.get('records_per_second'):
                        print(f"  Speed: {summary.get('records_per_second', 0):.1f} records/sec")

                if results.get('failed_downloads'):
                    print(f"Failed Downloads: {len(results['failed_downloads'])}")
                    for failure in results['failed_downloads'][:5]:  # Show first 5
                        print(f"  - {failure}")

            return results.get('success', True)
        else:
            logger.error("No results generated")
            return False

    except KeyboardInterrupt:
        logger.info("Download cancelled by user")
        return False
    except Exception as e:
        logger.error(f"Download script failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
