"""
LSTM深度學習模型
專為時間序列預測設計的LSTM神經網絡模型
支持多種LSTM架構和自定義配置
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 機器學習基礎庫（獨立於TensorFlow）
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 深度學習框架
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks  # type: ignore
    from tensorflow.keras.models import Sequential, Model  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional  # type: ignore
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Attention  # type: ignore
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore
    from tensorflow.keras.regularizers import l1, l2, l1_l2  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告: TensorFlow未安裝，LSTM模型將不可用")
    
    # 定義假的類型以避免導入錯誤
    Model = None
    Sequential = None
    optimizers = None
    callbacks = None
    
    # TensorFlow組件的假類型
    LSTM = None
    Dense = None
    Dropout = None
    BatchNormalization = None
    Bidirectional = None
    Conv1D = None
    MaxPooling1D = None
    GlobalAveragePooling1D = None
    Attention = None
    Adam = None
    RMSprop = None
    SGD = None
    EarlyStopping = None
    ReduceLROnPlateau = None
    ModelCheckpoint = None
    l1 = None
    l2 = None
    l1_l2 = None

from src.models.base_model import BaseModel, ModelMetrics
from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class LSTMModel(BaseModel):
    """LSTM深度學習模型類"""

    def __init__(self, symbol: str, timeframe: str, version: str = "1.0.0", **kwargs):
        """
        初始化LSTM模型

        Args:
            symbol: 交易品種
            timeframe: 時間框架
            version: 模型版本
            **kwargs: LSTM參數
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("LSTM模型需要安裝TensorFlow")

        super().__init__(symbol, timeframe, version, **kwargs)

        self.model_type = "regression"  # 或 "classification"

        # LSTM特定參數
        self.sequence_length = kwargs.get('sequence_length', 60)  # 序列長度
        self.lstm_units = kwargs.get('lstm_units', [64, 32])  # LSTM層單元數
        self.dense_units = kwargs.get('dense_units', [32, 16])  # 全連接層單元數
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)  # Dropout比率
        self.learning_rate = kwargs.get('learning_rate', 0.001)  # 學習率
        self.batch_size = kwargs.get('batch_size', 32)  # 批次大小
        self.epochs = kwargs.get('epochs', 100)  # 訓練輪數
        self.validation_split = kwargs.get('validation_split', 0.2)  # 驗證集比例

        # 高級參數
        self.use_bidirectional = kwargs.get('use_bidirectional', True)  # 雙向LSTM
        self.use_attention = kwargs.get('use_attention', False)  # 注意力機制
        self.use_cnn = kwargs.get('use_cnn', False)  # CNN-LSTM結合
        self.recurrent_dropout = kwargs.get('recurrent_dropout', 0.1)  # 循環Dropout
        self.l1_reg = kwargs.get('l1_reg', 0.0)  # L1正則化
        self.l2_reg = kwargs.get('l2_reg', 0.01)  # L2正則化

        # 數據預處理
        self.scaler_X = None
        self.scaler_y = None
        self.use_normalization = kwargs.get('use_normalization', True)
        self.scaling_method = kwargs.get('scaling_method', 'minmax')  # 'minmax', 'standard'

        # 模型架構
        self.architecture = kwargs.get('architecture', 'vanilla')  # 'vanilla', 'stacked', 'bidirectional', 'cnn_lstm', 'attention'

        # 訓練控制
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.reduce_lr_patience = kwargs.get('reduce_lr_patience', 5)
        self.min_lr = kwargs.get('min_lr', 1e-7)

        logger.info(f"初始化LSTM模型: {self.architecture}架構，序列長度={self.sequence_length}")

    def _create_model(self) -> 'Model':
        """創建LSTM模型架構"""
        try:
            if self.architecture == 'vanilla':
                model = self._create_vanilla_lstm()
            elif self.architecture == 'stacked':
                model = self._create_stacked_lstm()
            elif self.architecture == 'bidirectional':
                model = self._create_bidirectional_lstm()
            elif self.architecture == 'cnn_lstm':
                model = self._create_cnn_lstm()
            elif self.architecture == 'attention':
                model = self._create_attention_lstm()
            else:
                logger.warning(f"未知架構 {self.architecture}，使用默認vanilla LSTM")
                model = self._create_vanilla_lstm()

            # 編譯模型
            optimizer = self._get_optimizer()
            loss, metrics = self._get_loss_and_metrics()

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

            logger.debug(f"LSTM模型創建完成，參數量: {model.count_params()}")
            return model

        except Exception as e:
            logger.error(f"創建LSTM模型失敗: {e}")
            raise

    def _create_vanilla_lstm(self) -> 'Model':
        """創建基礎LSTM模型"""
        model = Sequential([
            LSTM(
                units=self.lstm_units[0],
                return_sequences=len(self.lstm_units) > 1,
                input_shape=(self.sequence_length, self.feature_count),
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)
            ),
            BatchNormalization(),
        ])

        # 額外LSTM層
        for i, units in enumerate(self.lstm_units[1:], 1):
            model.add(LSTM(
                units=units,
                return_sequences=(i < len(self.lstm_units) - 1),
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)
            ))
            model.add(BatchNormalization())

        # 全連接層
        for units in self.dense_units:
            model.add(Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)
            ))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())

        # 輸出層
        model.add(Dense(self.output_dim, activation=self.output_activation))

        return model

    def _create_stacked_lstm(self) -> 'Model':
        """創建堆疊LSTM模型"""
        model = Sequential()

        # 第一層LSTM
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.feature_count),
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout
        ))
        model.add(BatchNormalization())

        # 中間LSTM層
        for units in self.lstm_units[1:-1]:
            model.add(LSTM(
                units=units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout
            ))
            model.add(BatchNormalization())

        # 最後一層LSTM
        model.add(LSTM(
            units=self.lstm_units[-1],
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout
        ))
        model.add(BatchNormalization())

        # 全連接層
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())

        model.add(Dense(self.output_dim, activation=self.output_activation))

        return model

    def _create_bidirectional_lstm(self) -> 'Model':
        """創建雙向LSTM模型"""
        model = Sequential()

        # 雙向LSTM層
        for i, units in enumerate(self.lstm_units):
            model.add(Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=(i < len(self.lstm_units) - 1),
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout
                ),
                input_shape=(self.sequence_length, self.feature_count) if i == 0 else None
            ))
            model.add(BatchNormalization())

        # 全連接層
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())

        model.add(Dense(self.output_dim, activation=self.output_activation))

        return model

    def _create_cnn_lstm(self) -> 'Model':
        """創建CNN-LSTM組合模型"""
        model = Sequential([
            # CNN層用於特徵提取
            Conv1D(filters=64, kernel_size=3, activation='relu',
                   input_shape=(self.sequence_length, self.feature_count)),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),

            # LSTM層
            LSTM(self.lstm_units[0], return_sequences=True, dropout=self.dropout_rate),
            BatchNormalization(),
            LSTM(self.lstm_units[-1], dropout=self.dropout_rate),
            BatchNormalization(),

            # 全連接層
            Dense(self.dense_units[0], activation='relu'),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            Dense(self.output_dim, activation=self.output_activation)
        ])

        return model

    def _create_attention_lstm(self) -> 'Model':
        """創建帶注意力機制的LSTM模型"""
        # 輸入層
        inputs = keras.Input(shape=(self.sequence_length, self.feature_count))

        # LSTM層
        lstm_out = LSTM(
            self.lstm_units[0],
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.recurrent_dropout
        )(inputs)
        lstm_out = BatchNormalization()(lstm_out)

        # 注意力層 (簡化版)
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation('softmax')(attention)
        attention = keras.layers.RepeatVector(self.lstm_units[0])(attention)
        attention = keras.layers.Permute([2, 1])(attention)

        # 應用注意力權重
        sent_representation = keras.layers.multiply([lstm_out, attention])
        sent_representation = keras.layers.Lambda(
            lambda xin: keras.backend.sum(xin, axis=1)
        )(sent_representation)

        # 全連接層
        dense = sent_representation
        for units in self.dense_units:
            dense = Dense(units, activation='relu')(dense)
            dense = Dropout(self.dropout_rate)(dense)
            dense = BatchNormalization()(dense)

        # 輸出層
        outputs = Dense(self.output_dim, activation=self.output_activation)(dense)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def _get_optimizer(self) -> 'optimizers.Optimizer':
        """獲取優化器"""
        optimizer_name = self.hyperparameters.get('optimizer', 'adam').lower()

        if optimizer_name == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            return Adam(learning_rate=self.learning_rate)

    def _get_loss_and_metrics(self) -> Tuple[str, List[str]]:
        """獲取損失函數和評估指標"""
        if self.model_type == "classification":
            if self.output_dim == 1:
                loss = 'binary_crossentropy'
                metrics = ['accuracy', 'precision', 'recall']
            else:
                loss = 'categorical_crossentropy'
                metrics = ['accuracy', 'top_k_categorical_accuracy']
        else:  # regression
            loss = 'mse'
            metrics = ['mae', 'mape']

        return loss, metrics

    def _prepare_sequences(self, data: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """準備LSTM序列數據"""
        try:
            X_sequences = []
            y_sequences = []

            for i in range(self.sequence_length, len(data)):
                # 輸入序列
                X_sequences.append(data[i-self.sequence_length:i])

                # 目標值
                if target is not None:
                    y_sequences.append(target[i])

            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences) if y_sequences else None

            logger.debug(f"準備序列數據: X={X_sequences.shape}, y={y_sequences.shape if y_sequences is not None else None}")
            return X_sequences, y_sequences

        except Exception as e:
            logger.error(f"準備序列數據失敗: {e}")
            raise

    def _scale_data(self, X: pd.DataFrame, y: pd.Series = None, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """數據標準化"""
        try:
            if not self.use_normalization:
                X_scaled = X.values
                y_scaled = y.values if y is not None else None
                return X_scaled, y_scaled

            # 特徵縮放
            if fit_scaler:
                if self.scaling_method == 'minmax':
                    self.scaler_X = MinMaxScaler()
                else:
                    self.scaler_X = StandardScaler()

                X_scaled = self.scaler_X.fit_transform(X)
            else:
                if self.scaler_X is None:
                    raise ValueError("縮放器未初始化，請先訓練模型")
                X_scaled = self.scaler_X.transform(X)

            # 目標變量縮放
            y_scaled = None
            if y is not None:
                y_array = y.values.reshape(-1, 1)

                if fit_scaler:
                    if self.scaling_method == 'minmax':
                        self.scaler_y = MinMaxScaler()
                    else:
                        self.scaler_y = StandardScaler()

                    y_scaled = self.scaler_y.fit_transform(y_array).flatten()
                else:
                    if self.scaler_y is None:
                        raise ValueError("目標縮放器未初始化")
                    y_scaled = self.scaler_y.transform(y_array).flatten()

            return X_scaled, y_scaled

        except Exception as e:
            logger.error(f"數據縮放失敗: {e}")
            raise

    def _unscale_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """反標準化預測結果"""
        if not self.use_normalization or self.scaler_y is None:
            return predictions

        try:
            predictions_reshaped = predictions.reshape(-1, 1)
            unscaled = self.scaler_y.inverse_transform(predictions_reshaped)
            return unscaled.flatten()
        except Exception as e:
            logger.error(f"反標準化失敗: {e}")
            return predictions

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'LSTMModel':
        """訓練LSTM模型"""
        try:
            logger.info("開始訓練LSTM模型")
            start_time = datetime.now()

            # 驗證數據
            if not self.validate_input_data(X, y):
                raise ValueError("輸入數據驗證失敗")

            # 保存特徵信息
            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, 'name') else 'target'
            self.feature_count = len(self.feature_names)

            # 設置輸出維度
            if self.model_type == "classification":
                unique_classes = len(np.unique(y))
                self.output_dim = 1 if unique_classes == 2 else unique_classes
                self.output_activation = 'sigmoid' if unique_classes == 2 else 'softmax'
            else:
                self.output_dim = 1
                self.output_activation = 'linear'

            # 數據縮放
            X_scaled, y_scaled = self._scale_data(X, y, fit_scaler=True)

            # 準備序列數據
            X_sequences, y_sequences = self._prepare_sequences(X_scaled, y_scaled)

            if len(X_sequences) < self.sequence_length:
                raise ValueError(f"數據長度不足，需要至少{self.sequence_length}個樣本")

            # 驗證集處理
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_scaled, y_val_scaled = self._scale_data(X_val, y_val, fit_scaler=False)
                X_val_sequences, y_val_sequences = self._prepare_sequences(X_val_scaled, y_val_scaled)
                validation_data = (X_val_sequences, y_val_sequences)

            # 創建模型
            self.model = self._create_model()

            # 設置回調
            callbacks_list = self._get_callbacks()

            # 訓練模型
            logger.info(f"開始訓練，數據形狀: {X_sequences.shape}")

            history = self.model.fit(
                X_sequences, y_sequences,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=validation_data,
                validation_split=self.validation_split if validation_data is None else 0,
                callbacks=callbacks_list,
                verbose=1
            )

            # 更新模型狀態
            self.is_fitted = True
            training_time = (datetime.now() - start_time).total_seconds()

            # 更新元數據
            self.metadata.update({
                'trained_at': datetime.now(),
                'training_time': training_time,
                'data_shape': X.shape,
                'feature_count': len(self.feature_names),
                'sequence_length': self.sequence_length,
                'model_architecture': self.architecture,
                'total_parameters': self.model.count_params(),
                'training_history': {
                    'loss': history.history['loss'][-10:],  # 保存最後10個epoch的loss
                    'final_loss': history.history['loss'][-1],
                    'epochs_trained': len(history.history['loss'])
                }
            })

            logger.info(f"LSTM模型訓練完成，用時: {training_time:.2f}秒")

            # 評估模型
            self.evaluate(X, y, "training")
            if validation_data:
                val_X_df = pd.DataFrame(X_val_scaled[:len(X_val_sequences)], columns=self.feature_names)
                self.evaluate(val_X_df, pd.Series(y_val_scaled[:len(y_val_sequences)]), "validation")

            return self

        except Exception as e:
            logger.error(f"LSTM模型訓練失敗: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """預測"""
        try:
            if not self.is_fitted:
                raise ValueError("模型尚未訓練")

            # 數據縮放
            X_scaled, _ = self._scale_data(X, fit_scaler=False)

            # 準備序列數據
            X_sequences, _ = self._prepare_sequences(X_scaled)

            if len(X_sequences) == 0:
                logger.warning("序列數據不足，無法進行預測")
                return np.array([])

            # 模型預測
            predictions_scaled = self.model.predict(X_sequences, verbose=0)

            # 反標準化
            predictions = self._unscale_predictions(predictions_scaled.flatten())

            return predictions

        except Exception as e:
            logger.error(f"LSTM預測失敗: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """預測概率（分類模型）"""
        if self.model_type != "classification":
            raise ValueError("概率預測僅適用於分類模型")

        try:
            if not self.is_fitted:
                raise ValueError("模型尚未訓練")

            X_scaled, _ = self._scale_data(X, fit_scaler=False)
            X_sequences, _ = self._prepare_sequences(X_scaled)

            if len(X_sequences) == 0:
                return np.array([])

            probabilities = self.model.predict(X_sequences, verbose=0)

            # 對於二分類，確保返回兩列概率
            if self.output_dim == 1:
                probabilities = np.column_stack([1 - probabilities, probabilities])

            return probabilities

        except Exception as e:
            logger.error(f"LSTM概率預測失敗: {e}")
            raise

    def _get_callbacks(self) -> List['callbacks.Callback']:
        """獲取訓練回調"""
        callbacks_list = []

        # 早停回調
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)

        # 學習率調整
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
            verbose=1
        )
        callbacks_list.append(reduce_lr)

        # 模型檢查點（可選）
        if self.hyperparameters.get('save_checkpoints', False):
            checkpoint_path = f"checkpoints/{self.model_name}_{self.symbol}_{self.timeframe}_checkpoint.h5"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            callbacks_list.append(checkpoint)

        return callbacks_list

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """LSTM模型無法直接獲取特徵重要性"""
        logger.info("LSTM模型不支持特徵重要性計算")
        return None

    def save_model(self, directory: Optional[Path] = None) -> Path:
        """保存LSTM模型"""
        try:
            if directory is None:
                directory = config.get_model_path(self.symbol, self.timeframe, self.version)

            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)

            # 模型文件路徑
            model_file = directory / f"{self.model_name}_{self.symbol}_{self.timeframe}_v{self.version}.h5"
            metadata_file = directory / f"{self.model_name}_{self.symbol}_{self.timeframe}_v{self.version}_metadata.json"
            scaler_file = directory / f"{self.model_name}_{self.symbol}_{self.timeframe}_v{self.version}_scalers.pkl"

            # 保存Keras模型
            if self.model is not None:
                self.model.save(model_file)
                self.metadata['model_size'] = model_file.stat().st_size

            # 保存縮放器
            if self.scaler_X is not None or self.scaler_y is not None:
                import pickle
                with open(scaler_file, 'wb') as f:
                    pickle.dump({
                        'scaler_X': self.scaler_X,
                        'scaler_y': self.scaler_y,
                        'scaling_method': self.scaling_method,
                        'use_normalization': self.use_normalization
                    }, f)

            # 保存元數據
            metadata = self._prepare_metadata_for_save()
            with open(metadata_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"LSTM模型已保存到: {model_file}")
            return model_file

        except Exception as e:
            logger.error(f"保存LSTM模型失敗: {e}")
            raise

    def load_model(self, model_path: Path) -> 'LSTMModel':
        """加載LSTM模型"""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            # 加載Keras模型
            self.model = keras.models.load_model(model_path)
            self.is_fitted = True

            # 加載縮放器
            scaler_file = model_path.parent / f"{model_path.stem}_scalers.pkl"
            if scaler_file.exists():
                import pickle
                with open(scaler_file, 'rb') as f:
                    scaler_data = pickle.load(f)
                    self.scaler_X = scaler_data.get('scaler_X')
                    self.scaler_y = scaler_data.get('scaler_y')
                    self.scaling_method = scaler_data.get('scaling_method', 'minmax')
                    self.use_normalization = scaler_data.get('use_normalization', True)

            # 加載元數據
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    import json
                    loaded_metadata = json.load(f)
                self._load_metadata_from_dict(loaded_metadata)

            logger.info(f"LSTM模型已從 {model_path} 加載")
            return self

        except Exception as e:
            logger.error(f"加載LSTM模型失敗: {e}")
            raise

    def get_model_architecture(self) -> str:
        """獲取模型架構描述"""
        if self.model is None:
            return "模型未初始化"

        try:
            # 使用StringIO捕獲model.summary()的輸出
            import io
            string_buffer = io.StringIO()
            self.model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
            return string_buffer.getvalue()
        except:
            return f"LSTM模型: {self.architecture}架構"

    def plot_training_history(self, save_path: Optional[Path] = None):
        """繪製訓練歷史"""
        try:
            training_history = self.metadata.get('training_history', {})

            if not training_history:
                logger.warning("沒有訓練歷史數據")
                return

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'LSTM Training History - {self.symbol} {self.timeframe}', fontsize=16)

            # Loss曲線
            if 'loss' in training_history:
                axes[0, 0].plot(training_history['loss'])
                axes[0, 0].set_title('Model Loss')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_xlabel('Epoch')

            # 其他指標...

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"訓練歷史圖表已保存到: {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"繪製訓練歷史失敗: {e}")
