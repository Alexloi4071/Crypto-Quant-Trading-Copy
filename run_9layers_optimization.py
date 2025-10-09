#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
執行BTCUSDT 15m完整Layer0+9層優化（10層系統）
Layer0: 數據清洗基礎層 + Layer1-8: 核心與專項優化
"""
import json
import sys
import traceback

# 添加路徑到導入列表頂部
sys.path.append('.')
from optuna_system.coordinator import OptunaCoordinator  # noqa: E402
from config.timeframe_scaler import MultiTimeframeCoordinator  # noqa: E402


def main():
    """執行BTCUSDT完整Layer0+9層優化 - 支持單一時框或多時框模式"""
    import argparse
    
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='BTCUSDT加密货币量化交易优化系统')
    parser.add_argument('--mode', choices=['single', 'multi'], default='single',
                       help='优化模式: single(单时框) 或 multi(多时框)')
    parser.add_argument('--timeframe', default='15m',
                       help='单时框模式下的时框 (默认: 15m)')
    parser.add_argument('--multi-timeframes', nargs='+', default=['15m', '1h', '4h'],
                       help='多时框模式下的时框列表 (默认: 15m 1h 4h)')
    parser.add_argument('--trials', type=int, default=50,
                       help='每个时框的优化试验次数 (默认: 50)')
    parser.add_argument('--stage3-trials', type=int, default=None,
                       help='Layer3 每个模型的 trials 数（默认沿用 run_layer3_optimization.py 默认值）')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        print(f">> 開始 BTCUSDT {args.timeframe} 單一時框Layer0+9層優化...")
        print(f"時框: {args.timeframe}")
    else:
        print(f">> 開始 BTCUSDT 多時框特徵優化...")
        print(f"時框列表: {args.multi_timeframes}")
        print(f"每時框試驗次數: {args.trials}")
    
    print("策略: 快速完整優化")
    print("層級: Layer0數據清洗 + Layer1-4核心層 + Layer5-8專項層")
    print("-" * 80)

    try:
        # 首先測試能否正常導入模塊（修復版改進）
        print("Step 1: 測試模塊導入...")
        
        try:
            coordinator = OptunaCoordinator(
                symbol='BTCUSDT',
                timeframe=args.timeframe,
                data_path='data'
            )
            print("成功: 協調器初始化成功")
        except ImportError as e:
            print(f"錯誤: 模塊導入失敗: {e}")
            print("請檢查所有優化器文件是否存在")
            return None
        
        # 測試單個層級（修復版診斷）
        print("Step 2: 測試單層級運行...")
        
        # 先測試Layer0（數據清洗）
        print("[清洗] 測試 Layer0: 數據清洗...")
        try:
            layer0_result = coordinator.run_layer0_data_cleaning(n_trials=5)
            if 'error' in layer0_result:
                print(f"警告: Layer0運行有問題: {layer0_result['error']}")
            else:
                print(f"成功: Layer0運行成功，分數: {layer0_result.get('best_score', 'N/A')}")
        except Exception as e:
            print(f"錯誤: Layer0運行失敗: {e}")

        # 根據模式執行不同的優化策略
        if args.mode == 'multi':
            print("Step 3: 準備多時框縮放配置...")
            multi_scaler = MultiTimeframeCoordinator(symbol='BTCUSDT', data_path='data')
            timeframe_configs = {}
            for tf in args.multi_timeframes:
                cfg = multi_scaler.get_scaled_config_for_timeframe(tf)
                timeframe_configs[tf] = cfg

            print("Step 4: 逐時框執行Layer0-2優化...")
            multi_results = {}
            for tf in args.multi_timeframes:
                print("-" * 40)
                print(f"⏱️ 時框: {tf}")
                scaled_cfg = timeframe_configs[tf]
                coordinator_tf = OptunaCoordinator(
                    symbol='BTCUSDT',
                    timeframe=tf,
                    data_path='data',
                    scaled_config=scaled_cfg
                )

                print("  ➤ Layer0 清洗...")
                coordinator_tf.run_layer0_data_cleaning(max(10, args.trials // 5))

                print("  ➤ Layer1 標籤優化...")
                layer1_result = coordinator_tf.run_layer1_label_optimization(n_trials=args.trials)

                print("  ➤ Layer2 特徵優化...")
                layer2_result = coordinator_tf.run_layer2_feature_optimization(n_trials=args.trials)

                multi_results[tf] = {
                    'layer1': layer1_result,
                    'layer2': layer2_result
                }

            result = {
                'version': f"multi_timeframe_{len(args.multi_timeframes)}tf",
                'layer_results': multi_results,
                'meta_vol': multi_scaler.meta_vol,
                'global_vol': multi_scaler.global_vol
            }
            
            # 包裝結果以保持一致性
            if 'error' not in result:
                result = {
                    'version': f"multi_timeframe_{len(args.multi_timeframes)}tf",
                    'layer_results': {'layer2_features': result},
                    'optimization_summary': {
                        'total_modules': len(args.multi_timeframes),
                        'successful_modules': result['summary']['successful_optimizations'],
                        'failed_modules': result['summary']['failed_optimizations'],
                        'success_rate': result['summary']['successful_optimizations'] / len(args.multi_timeframes),
                        'best_scores': {tf: cfg.get('best_score', 0.0) for tf, cfg in result['best_configs'].items()}
                    }
                }
        else:
            # 🔄 單一時框完整優化模式
            print("Step 3: 執行單一時框完整優化...")
            result = coordinator.quick_complete_optimization()
            stage3_trials = args.stage3_trials

        # 根據優化模式顯示不同的結果
        if args.mode == 'multi':
            print("\n>> 多時框Layer0-2優化完成！")
            print(f"版本: {result.get('version', 'N/A')}")

            for tf, layers in result['layer_results'].items():
                print(f"\n⏱️ 時框 {tf}:")
                layer1 = layers.get('layer1', {})
                layer2 = layers.get('layer2', {})
                if 'best_score' in layer1:
                    print(f"  Layer1 標籤: F1={layer1['best_score']:.4f}, lag={layer1.get('best_params', {}).get('lag')}")
                else:
                    print("  Layer1 標籤: 失敗或無結果")
                if 'best_score' in layer2:
                    print(f"  Layer2 特徵: F1={layer2['best_score']:.4f}, coarse_k={layer2.get('best_params', {}).get('coarse_k')}")
                else:
                    print("  Layer2 特徵: 失敗或無結果")

            print("\n>> 部署建議:")
            print("   1. 使用 configs/label_params_{tf}.json 與 feature_params_{tf}.json 作為對應時框設定")
            print("   2. 後續 Layer3-8 可依相同縮放配置逐層擴充")
            print(f"   3. 參考 meta_vol={result.get('meta_vol', 0.0):.4f}, global_vol={result.get('global_vol', {})}")

        else:
            print("\n>> 單一時框Layer0+9層優化完成！")
            print(f"版本: {result.get('version', 'N/A')}")

            summary = result.get('optimization_summary', {})
            print(f"總模塊數: {summary.get('total_modules', 0)} (包含Layer0)")
            print(f"成功模塊: {summary.get('successful_modules', 0)}")
            print(f"失敗模塊: {summary.get('failed_modules', 0)}")
            print(f"成功率: {summary.get('success_rate', 0):.1%}")

            print("\n>> 各層最佳分數:")
            best_scores = summary.get('best_scores', {})
            for layer, score in best_scores.items():
                if 'layer0' in layer.lower():
                    print(f"  [清洗] {layer}: {score:.4f}")
                elif 'layer1' in layer.lower() or 'layer2' in layer.lower() or 'layer3' in layer.lower() or 'layer4' in layer.lower():
                    print(f"  [核心] {layer}: {score:.4f}")
                else:
                    print(f"  [專項] {layer}: {score:.4f}")

        print(f"\n>> 結果已保存，版本: {result.get('version')}")
        
        if args.mode == 'multi':
            print(">> 🎉 恭喜！多時框特徵優化完成！")
            print(">> 部署建議:")
            print("   1. 根據交易時框選擇對應的最佳特徵配置")
            print("   2. 使用記錄的coarse_k、fine_k參數進行特徵選擇") 
            print("   3. 定期重新運行優化以適應市場變化")
        else:
            print(">> 恭喜！現在數據從Layer0開始得到完整9層優化！")
            print(">> 所有優化器現在統一由 coordinator.py 調用管理")

        # 保存詳細結果到文件
        result_file = f"optimization_result_{result.get('version')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 詳細結果已保存至: {result_file}")

        return result

    except Exception as e:
        print(f"錯誤: 優化過程出現錯誤: {e}")
        print(f"錯誤詳情：")
        traceback.print_exc()
        
        # 修復版診斷信息
        print("\n>> 診斷建議:")
        
        # 檢查數據目錄
        from pathlib import Path
        data_dir = Path('data/raw/BTCUSDT')
        if not data_dir.exists():
            print(f"錯誤: 數據目錄不存在: {data_dir}")
            print("系統將使用模擬數據，這是正常的")
        else:
            print(f"成功: 數據目錄存在: {data_dir}")
        
        return None


def print_usage_examples():
    """打印使用示例"""
    print("="*80)
    print("📖 使用示例:")
    print("="*80)
    print("1. 單一時框完整優化 (默認):")
    print("   python run_9layers_optimization.py")
    print("   python run_9layers_optimization.py --timeframe 1h")
    print()
    print("2. 多時框特徵優化:")
    print("   python run_9layers_optimization.py --mode multi")
    print("   python run_9layers_optimization.py --mode multi --multi-timeframes 15m 1h 4h 1d")
    print("   python run_9layers_optimization.py --mode multi --trials 100")
    print()
    print("3. 自定義參數:")
    print("   python run_9layers_optimization.py --mode single --timeframe 4h")
    print("   python run_9layers_optimization.py --mode multi --trials 30 --multi-timeframes 15m 1h")
    print()
    print("🎯 多時框特徵優化說明:")
    print("   • 為每個時框找到最佳的140-203粗選→10-25精選特徵組合（全量203特徵池）")
    print("   • 15m時框適合短期交易，4h時框適合中長期交易")
    print("   • 使用70%-100%特徵池，確保所有203個特徵都有被選中機會")
    print("   • 結果保存在 multi_timeframe_feature_optimization.json")
    print("   • 部署時根據交易時框動態選擇對應特徵配置")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # 如果用戶請求幫助，顯示使用示例
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_usage_examples()
    else:
        result = main()
