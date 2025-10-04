# -*- coding: utf-8 -*-
"""
多币种验证脚本
验证策略在不同币种上的通用性
"""
import sys
from pathlib import Path
import pandas as pd
import json

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from optuna_system.coordinator import OptunaCoordinator
from config.timeframe_scaler import MultiTimeframeCoordinator


def validate_multi_symbols(
    symbols: List[str] = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT'],
    timeframe: str = '15m',
    n_trials: int = 50  # 快速验证，试验次数减少
):
    """
    多币种验证
    
    Args:
        symbols: 验证的币种列表
        timeframe: 时间框架
        n_trials: 每个币种的优化次数（快速验证用50）
    """
    
    print("="*70)
    print("🌐 多币种验证工具")
    print("="*70)
    print(f"验证币种: {symbols}")
    print(f"时间框架: {timeframe}")
    print(f"试验次数: {n_trials}")
    print("="*70)
    
    results_summary = {}
    
    for idx, symbol in enumerate(symbols):
        print(f"\n[{idx+1}/{len(symbols)}] 验证币种: {symbol}")
        print("-"*70)
        
        try:
            # 创建协调器
            multi_scaler = MultiTimeframeCoordinator(symbol=symbol, data_path='data')
            scaled_config = multi_scaler.get_scaled_config_for_timeframe(timeframe)
            
            coordinator = OptunaCoordinator(
                symbol=symbol,
                timeframe=timeframe,
                data_path='data',
                scaled_config=scaled_config
            )
            
            # 运行Layer0-2优化
            print(f"  [Step 1/3] Layer0 数据清洗...")
            layer0 = coordinator.run_layer0_data_cleaning(n_trials=max(10, n_trials//5))
            
            print(f"  [Step 2/3] Layer1 标签优化...")
            layer1 = coordinator.run_layer1_label_optimization(n_trials=n_trials)
            
            print(f"  [Step 3/3] Layer2 特征优化...")
            layer2 = coordinator.run_layer2_feature_optimization(n_trials=n_trials)
            
            # 提取结果
            layer0_score = layer0.get('best_score', 0)
            layer1_score = layer1.get('best_score', 0)
            layer2_score = layer2.get('best_score', 0)
            
            # 标签分布
            label_dist = {}
            if 'metadata' in layer1:
                label_dist = layer1['metadata'].get('label_distribution', {})
            
            # 特征数量
            n_features = 0
            if 'best_params' in layer2 and 'selected_features' in layer2['best_params']:
                n_features = len(layer2['best_params']['selected_features'])
                
                # 策略特征统计
                selected = layer2['best_params']['selected_features']
                strategy_features = [f for f in selected if any(p in f for p in ['wyk_', 'td_', 'micro_'])]
                n_strategy = len(strategy_features)
            else:
                n_strategy = 0
            
            # 记录结果
            results_summary[symbol] = {
                'success': True,
                'layer0_score': layer0_score,
                'layer1_score': layer1_score,
                'layer2_score': layer2_score,
                'label_distribution': label_dist,
                'n_features': n_features,
                'n_strategy_features': n_strategy
            }
            
            print(f"\n  ✅ {symbol} 验证完成:")
            print(f"     Layer0: {layer0_score:.4f}")
            print(f"     Layer1: {layer1_score:.4f}")
            print(f"     Layer2: {layer2_score:.4f}")
            print(f"     特征数: {n_features}")
            print(f"     策略特征: {n_strategy}个")
            
        except Exception as e:
            print(f"  ❌ {symbol} 验证失败: {e}")
            results_summary[symbol] = {
                'success': False,
                'error': str(e)
            }
    
    # 生成汇总报告
    print("\n" + "="*70)
    print("📊 验证汇总")
    print("="*70)
    
    generate_validation_report(results_summary, timeframe, symbols)
    
    return results_summary


def generate_validation_report(results: Dict, timeframe: str, symbols: List[str]):
    """生成验证报告"""
    
    # 统计成功率
    successful = [s for s, r in results.items() if r.get('success')]
    success_rate = len(successful) / len(symbols)
    
    print(f"\n成功率: {len(successful)}/{len(symbols)} ({success_rate:.1%})")
    
    # 表格输出
    print("\n| 币种 | Layer1 F1 | Layer2 F1 | 特征数 | 策略特征 | 状态 |")
    print("|------|-----------|-----------|--------|----------|------|")
    
    for symbol in symbols:
        data = results[symbol]
        
        if data.get('success'):
            l1 = data.get('layer1_score', 0)
            l2 = data.get('layer2_score', 0)
            nf = data.get('n_features', 0)
            ns = data.get('n_strategy_features', 0)
            status = "✅"
            
            print(f"| {symbol} | {l1:.4f} | {l2:.4f} | {nf} | {ns} | {status} |")
        else:
            print(f"| {symbol} | - | - | - | - | ❌ 失败 |")
    
    # 计算平均值
    valid_results = [r for r in results.values() if r.get('success')]
    
    if valid_results:
        avg_l1 = np.mean([r['layer1_score'] for r in valid_results])
        avg_l2 = np.mean([r['layer2_score'] for r in valid_results])
        avg_features = np.mean([r['n_features'] for r in valid_results])
        
        print(f"\n平均指标:")
        print(f"  Layer1 F1: {avg_l1:.4f}")
        print(f"  Layer2 F1: {avg_l2:.4f}")
        print(f"  特征数: {avg_features:.0f}")
    
    # 保存详细结果
    output_file = project_root / 'optuna_system' / 'results' / 'multi_symbol_validation.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timeframe': timeframe,
            'symbols': symbols,
            'results': results,
            'summary': {
                'success_rate': success_rate,
                'avg_layer1_f1': avg_l1 if valid_results else 0,
                'avg_layer2_f1': avg_l2 if valid_results else 0
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细结果已保存: {output_file}")
    
    # 生成Markdown报告
    md_report = output_file.parent / 'multi_symbol_validation.md'
    
    md_lines = [
        "# 多币种验证报告\n",
        f"\n**时间框架**: {timeframe}",
        f"\n**验证币种**: {len(symbols)}个\n",
        "\n## 结果汇总\n",
        "| 币种 | Layer1 F1 | Layer2 F1 | 特征数 | 策略特征 |",
        "|------|-----------|-----------|--------|----------|"
    ]
    
    for symbol in symbols:
        data = results[symbol]
        if data.get('success'):
            md_lines.append(
                f"| {symbol} | {data['layer1_score']:.4f} | "
                f"{data['layer2_score']:.4f} | {data['n_features']} | "
                f"{data['n_strategy_features']} |"
            )
        else:
            md_lines.append(f"| {symbol} | ❌ | ❌ | - | - |")
    
    if valid_results:
        md_lines.extend([
            "",
            f"**平均Layer1 F1**: {avg_l1:.4f}",
            f"**平均Layer2 F1**: {avg_l2:.4f}",
            f"**成功率**: {success_rate:.1%}"
        ])
    
    with open(md_report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"📄 Markdown报告已保存: {md_report}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='多币种验证工具')
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                       help='验证的币种列表')
    parser.add_argument('--timeframe', default='15m',
                       help='时间框架')
    parser.add_argument('--trials', type=int, default=50,
                       help='每个币种的优化次数')
    
    args = parser.parse_args()
    
    validate_multi_symbols(
        symbols=args.symbols,
        timeframe=args.timeframe,
        n_trials=args.trials
    )
