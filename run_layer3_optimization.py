#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Layer3 五模型超参数优化和训练脚本
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import os
import json
import argparse

try:
    # 确保控制台使用 UTF-8 编码，否则中文日志会报错
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
except Exception:
    pass

# 设置日志
log_filename = f'layer3_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

try:
    from optuna_system.utils.io_utils import read_dataframe
except ImportError:
    from optuna_system.utils.io_utils import read_dataframe  # type: ignore


def resolve_layer_json(json_name: str) -> dict:
    json_path = Path("optuna_system/configs") / json_name
    if not json_path.exists():
        return {}
    try:
        with json_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_latest_version() -> str | None:
    try:
        latest_file = Path("optuna_system/results/latest.txt")
        if latest_file.exists():
            v = latest_file.read_text(encoding="utf-8").strip()
            return v if v else None
    except Exception:
        pass
    return None


def load_dataframe_with_fallback(path: Path) -> pd.DataFrame:
    df = read_dataframe(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

def find_latest_file(directory: Path, pattern: str, prefer_suffix: tuple[str, ...] = (".parquet", ".pkl")) -> Path | None:
    candidates = sorted(directory.rglob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    for suffix in prefer_suffix:
        for path in candidates:
            if path.suffix.lower() == suffix.lower():
                try:
                    _ = read_dataframe(path)
                    return path
                except Exception:
                    continue
    for path in candidates:
        try:
            _ = read_dataframe(path)
            return path
        except Exception:
            continue
    return None

def check_prerequisites():
    """检查前置条件"""
    logger.info("="*80)
    logger.info("检查Layer3前置条件...")
    logger.info("="*80)
    
    version_sub = get_latest_version()

    # 優先在 latest 版本子目錄中尋找，否則回退到整個目錄掃描
    base_labels_dir = Path("data/processed/labels/BTCUSDT_15m")
    labels_dir = base_labels_dir / version_sub if version_sub else base_labels_dir
    labels_file = find_latest_file(labels_dir, "labels*", prefer_suffix=(".pkl", ".parquet"))
    if labels_file is None:
        # 回退：在更高層級掃描含 15m 的標籤檔（遞迴）
        labels_file = find_latest_file(Path("data/processed/labels"), "labels*15m*", prefer_suffix=(".pkl", ".parquet"))
    
    if labels_file:
        size_mb = os.path.getsize(labels_file) / (1024*1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(labels_file))
        df = read_dataframe(labels_file)
        logger.info(f"✅ Layer1 标签文件:")
        logger.info(f"   路径: {labels_file}")
        logger.info(f"   大小: {size_mb:.2f}MB")
        logger.info(f"   修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   形状: {df.shape}")
        if 'label' in df.columns:
            label_dist = df['label'].value_counts().to_dict()
            logger.info(f"   标签分布: {label_dist}")
    else:
        logger.error("❌ 未找到Layer1标签文件")
        return False, None, None
    
    base_features_dir = Path("data/processed/features/BTCUSDT_15m")
    features_dir = base_features_dir / version_sub if version_sub else base_features_dir
    features_file = find_latest_file(features_dir, "features*", prefer_suffix=(".parquet", ".pkl"))
    if features_file is None:
        # 回退：在更高層級掃描含 15m 的特徵檔（遞迴）
        features_file = find_latest_file(Path("data/processed/features"), "features*15m*", prefer_suffix=(".parquet", ".pkl"))
    
    if features_file:
        size_mb = os.path.getsize(features_file) / (1024*1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(features_file))
        df = read_dataframe(features_file)
        logger.info(f"✅ Layer2 特征文件:")
        logger.info(f"   路径: {features_file}")
        logger.info(f"   大小: {size_mb:.2f}MB")
        logger.info(f"   修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   形状: {df.shape} (样本数 × 特征数)")
    else:
        logger.error("❌ 未找到Layer2特征文件")
        return False, None, None
    
    cleaned_candidates = [
        Path("data/processed/cleaned/BTCUSDT_15m/cleaned_ohlcv.pkl"),
        Path("data/processed/cleaned/BTCUSDT_15m/cleaned_ohlcv.parquet")
    ]
    cleaned_file = next((p for p in cleaned_candidates if p.exists()), None)
    if cleaned_file:
        logger.info(f"✅ Layer0 清洗文件: {cleaned_file}")
    else:
        logger.warning("⚠️ 未找到cleaned_ohlcv (收益率计算可能受影响)")
    
    # 版本一致性提示
    try:
        def extract_vn(p: Path) -> str | None:
            parts = [part for part in p.parts if part.startswith('v') and len(part) > 1 and part[1:].isdigit()]
            return parts[-1] if parts else None
        vn_labels = extract_vn(labels_file) if labels_file else None
        vn_features = extract_vn(features_file) if features_file else None
        if vn_labels and vn_features and vn_labels != vn_features:
            logger.warning(f"⚠️ 標籤與特徵版本不一致: labels={vn_labels}, features={vn_features} — 建議重新物化或指定一致版本")
        if version_sub:
            logger.info(f"版本優先策略: 使用 latest 版本 {version_sub}")
    except Exception:
        pass

    logger.info("\n✅ 所有前置条件满足！准备开始Layer3优化...\n")
    return True, labels_file, features_file

def prepare_config_files(labels_file, features_file):
    """准备configs目录下的配置文件链接"""
    logger.info("准备配置文件...")
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # 复制或链接文件到configs目录
    target_labels = configs_dir / "labels_15m.parquet"
    target_features = configs_dir / "selected_features_15m.parquet"
    
    # 读取并保存到configs
    labels_df = read_dataframe(labels_file)
    labels_df.to_parquet(target_labels)
    logger.info(f"✅ 标签文件已复制到: {target_labels}")
    
    features_df = read_dataframe(features_file)
    # 防洩漏：若特徵集含有 label，先移除再寫入 configs
    if 'label' in features_df.columns:
        features_df = features_df.drop(columns=['label'])
    features_df.to_parquet(target_features)
    logger.info(f"✅ 特征文件已复制到: {target_features}")
    
    # 复制cleaned文件
    cleaned_priorities = [
        Path("data/processed/cleaned/BTCUSDT_15m/cleaned_ohlcv.pkl"),
        Path("data/processed/cleaned/BTCUSDT_15m/cleaned_ohlcv.parquet")
    ]
    for cleaned_source in cleaned_priorities:
        if cleaned_source.exists():
            cleaned_target = configs_dir / "cleaned_ohlcv_15m.parquet"
            cleaned_df = read_dataframe(cleaned_source)
            cleaned_df.to_parquet(cleaned_target)
            logger.info(f"✅ 清洗文件已复制到: {cleaned_target}")
            break

def run_layer3_optimization(n_trials_per_model: int = 250):
    """运行Layer3优化"""
    logger.info("="*80)
    logger.info("开始Layer3五模型超参数优化...")
    logger.info("="*80)
    
    try:
        from optuna_system.optimizers.optuna_model import ModelOptimizer
        
        # 创建优化器
        version_sub = get_latest_version()
        results_base = Path("optuna_system/results/BTCUSDT_15m")
        results_path = results_base / version_sub if version_sub else results_base
        results_path.mkdir(parents=True, exist_ok=True)

        optimizer = ModelOptimizer(
            data_path='data',
            config_path='configs',
            results_path=str(results_path),
            symbol='BTCUSDT',
            timeframe='15m'
        )
        
        logger.info(f"\n配置参数:")
        logger.info(f"  - 数据路径: data")
        logger.info(f"  - 配置路径: configs")
        logger.info(f"  - 结果路径: {results_path}")
        logger.info(f"  - 交易对: BTCUSDT")
        logger.info(f"  - 时间框架: 15m")
        logger.info(f"  - 每个模型trials: {n_trials_per_model}")
        logger.info(f"  - 总计trials: {n_trials_per_model * len(optimizer.models_to_train)}")
        logger.info(f"  - 模型列表: {optimizer.models_to_train}")
        logger.info("")
        
        # 开始优化
        start_time = datetime.now()
        logger.info(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        result = optimizer.optimize(n_trials=n_trials_per_model)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("="*80)
        logger.info("🎉 Layer3优化完成！")
        logger.info("="*80)
        logger.info(f"⏰ 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️ 总耗时: {duration/60:.1f}分钟 ({duration:.0f}秒)")
        logger.info(f"🏆 最佳模型: {result.get('best_model_type', 'N/A')}")
        logger.info(f"📊 最佳得分: {result.get('best_score', 0):.4f}")
        
        if 'all_models' in result:
            logger.info("\n各模型性能:")
            for model_name, model_result in result['all_models'].items():
                logger.info(f"  - {model_name}: {model_result['best_score']:.4f}")
        
        # 检查生成的文件
        logger.info("\n生成的文件:")
        results_path = Path("optuna_system/results/BTCUSDT_15m")
        for pred_file in results_path.glob("*_predictions.parquet"):
            size_kb = os.path.getsize(pred_file) / 1024
            df = pd.read_parquet(pred_file)
            logger.info(f"  ✅ {pred_file.name}: {size_kb:.1f}KB, 形状: {df.shape}")
        
        config_file = Path("configs/model_params.json")
        if config_file.exists():
            size_kb = os.path.getsize(config_file) / 1024
            logger.info(f"  ✅ model_params.json: {size_kb:.1f}KB")
        
        logger.info("\n🎊 Layer3优化成功完成！可以继续执行Layer4-9。")
        return True
        
    except Exception as e:
        logger.error(f"❌ Layer3优化失败: {e}", exc_info=True)
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Layer3 优化脚本")
    parser.add_argument("--trials", type=int, default=50,
                        help="每个模型的 trials 数（默认 50）")
    return parser.parse_args()


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("Layer3 五模型集成优化启动器")
    logger.info("="*80)
    logger.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"工作目录: {Path.cwd()}")
    logger.info("")

    args = parse_args()
    trials_per_model = args.trials
    
    # 步骤1: 检查前置条件
    ready, labels_file, features_file = check_prerequisites()
    if not ready:
        logger.error("前置条件不满足，无法继续")
        return False
    
    # 步骤2: 准备配置文件
    prepare_config_files(labels_file, features_file)
    
    # 步骤3: 运行优化
    logger.info(f"本次使用每模型 trials = {trials_per_model}，约等于总计 {trials_per_model * len(['lightgbm','xgboost','catboost','randomforest','extratrees'])} trials")
    success = run_layer3_optimization(n_trials_per_model=trials_per_model)
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("✅ 所有步骤成功完成！")
        logger.info("="*80)
        logger.info("\n下一步:")
        logger.info("1. 查看日志文件了解详细信息")
        logger.info("2. 检查optuna_system/results/BTCUSDT_15m/目录下的预测文件")
        logger.info("3. 运行Layer6集成优化: python run_9layers_optimization.py")
    else:
        logger.error("\n" + "="*80)
        logger.error("❌ 优化过程出现错误")
        logger.error("="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

