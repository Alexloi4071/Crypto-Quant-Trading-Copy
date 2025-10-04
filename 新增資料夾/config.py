from optuna_system.coordinator import OptunaCoordinator

coordinator = OptunaCoordinator(
    symbol='BTCUSDT',
    timeframe='15m',
    data_path='data'
)

print('--- Layer0 ---')
layer0_result = coordinator.run_layer0_data_cleaning(n_trials=25)
print(layer0_result)

print('--- Layer1 ---')
layer1_result = coordinator.run_layer1_label_optimization(n_trials=150)
print(layer1_result)

print('--- Layer2 ---')
layer2_result = coordinator.run_layer2_feature_optimization(n_trials=100)
print(layer2_result)