#!/usr/bin/env python3
"""
验证重采样结果质量
"""
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import DataPreprocessor

def verify_data_quality():
    print("🔍 验证重采样数据质量")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['15m', '1h', '4h', '1d']
    
    for symbol in symbols:
        print(f"\n📊 {symbol} 数据质量检查:")
        print("-" * 30)
        
        for tf in timeframes:
            file_path = f"data/raw/{symbol}/{symbol}_{tf}_ohlcv.parquet"
            try:
                df = pd.read_parquet(file_path)
                quality = preprocessor.validate_data_quality(df)
                
                file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
                
                print(f"  {tf:>3}: {len(df):>7,} 条 | 质量: {quality['quality_score']:.1f}/100 | {file_size:.1f}MB")
                
                # 显示时间范围
                if len(df) > 0:
                    start_time = df.index.min()
                    end_time = df.index.max()
                    print(f"       时间: {start_time} ~ {end_time}")
                
                # 显示价格范围
                if 'close' in df.columns:
                    min_price = df['close'].min()
                    max_price = df['close'].max()
                    print(f"       价格: ${min_price:.2f} ~ ${max_price:.2f}")
                
                # 检查质量问题
                if quality.get('issues'):
                    print(f"       问题: {quality['issues']}")
                    
                # 显示索引缺口统计
                if 'index_gap_stats' in quality:
                    gap_info = quality['index_gap_stats']
                    if gap_info.get('num_large_gaps', 0) > 0:
                        print(f"       缺口: {gap_info['num_large_gaps']} 个大缺口")
                
                # 显示零成交量统计
                if 'volume' in df.columns:
                    zero_vol = (df['volume'] == 0).sum()
                    if zero_vol > 0:
                        print(f"       零量: {zero_vol} 条 ({zero_vol/len(df)*100:.1f}%)")
                
            except Exception as e:
                print(f"  {tf:>3}: ❌ 错误 - {e}")
    
    print(f"\n{'='*50}")
    print("✅ 数据质量验证完成!")

if __name__ == "__main__":
    verify_data_quality()
