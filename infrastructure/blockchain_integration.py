"""
Blockchain Integration
区块链集成系统，支持主流区块链网络的交互和智能合约部署
实现以太坊、比特币、BSC等区块链的统一接口和DeFi协议集成
"""

import asyncio
import aiohttp
from web3 import Web3
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
from decimal import Decimal
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class BlockchainNetwork(Enum):
    """区块链网络"""
    ETHEREUM_MAINNET = "ethereum_mainnet"
    ETHEREUM_SEPOLIA = "ethereum_sepolia"
    BITCOIN_MAINNET = "bitcoin_mainnet"
    BITCOIN_TESTNET = "bitcoin_testnet"
    BSC_MAINNET = "bsc_mainnet"
    BSC_TESTNET = "bsc_testnet"
    POLYGON_MAINNET = "polygon_mainnet"
    ARBITRUM_MAINNET = "arbitrum_mainnet"
    OPTIMISM_MAINNET = "optimism_mainnet"

class TransactionStatus(Enum):
    """交易状态"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ContractType(Enum):
    """合约类型"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    DEFI_EXCHANGE = "defi_exchange"
    DEFI_LENDING = "defi_lending"
    DEFI_YIELD_FARM = "defi_yield_farm"
    CUSTOM = "custom"

@dataclass

class BlockchainConfig:
    """区块链配置"""
    network: BlockchainNetwork
    rpc_url: str
    chain_id: int

    # 代币信息
    native_token: str
    block_explorer_url: str

    # 网络参数
    average_block_time: float  # 秒
    gas_price_gwei: float = 20.0

    def to_dict(self) -> dict:
        return {
            'network': self.network.value,
            'rpc_url': self.rpc_url,
            'chain_id': self.chain_id,
            'native_token': self.native_token,
            'block_explorer_url': self.block_explorer_url,
            'average_block_time': self.average_block_time,
            'gas_price_gwei': self.gas_price_gwei
        }

@dataclass

class Transaction:
    """区块链交易"""
    tx_hash: str
    network: BlockchainNetwork

    # 交易基本信息
    from_address: str
    to_address: str
    value: Decimal
    gas_price: int
    gas_limit: int
    gas_used: Optional[int] = None

    # 状态信息
    status: TransactionStatus = TransactionStatus.PENDING
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    transaction_index: Optional[int] = None

    # 时间信息
    timestamp: datetime = field(default_factory=datetime.now)
    block_timestamp: Optional[datetime] = None

    # 成本信息
    transaction_fee: Optional[Decimal] = None

    # 合约相关
    contract_address: Optional[str] = None
    input_data: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'tx_hash': self.tx_hash,
            'network': self.network.value,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value': str(self.value),
            'gas_price': self.gas_price,
            'gas_limit': self.gas_limit,
            'gas_used': self.gas_used,
            'status': self.status.value,
            'block_number': self.block_number,
            'block_hash': self.block_hash,
            'timestamp': self.timestamp.isoformat(),
            'block_timestamp': self.block_timestamp.isoformat() if self.block_timestamp else None,
            'transaction_fee': str(self.transaction_fee) if self.transaction_fee else None,
            'contract_address': self.contract_address,
            'logs_count': len(self.logs)
        }

@dataclass

class SmartContract:
    """智能合约"""
    contract_address: str
    contract_name: str
    contract_type: ContractType
    network: BlockchainNetwork

    # 合约信息
    abi: List[Dict[str, Any]]
    bytecode: Optional[str] = None
    source_code: Optional[str] = None

    # 部署信息
    deployer: Optional[str] = None
    deployment_tx: Optional[str] = None
    deployment_block: Optional[int] = None
    deployed_at: datetime = field(default_factory=datetime.now)

    # 验证信息
    is_verified: bool = False
    compiler_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'contract_address': self.contract_address,
            'contract_name': self.contract_name,
            'contract_type': self.contract_type.value,
            'network': self.network.value,
            'deployer': self.deployer,
            'deployment_tx': self.deployment_tx,
            'deployment_block': self.deployment_block,
            'deployed_at': self.deployed_at.isoformat(),
            'is_verified': self.is_verified,
            'compiler_version': self.compiler_version,
            'has_abi': len(self.abi) > 0,
            'has_source': self.source_code is not None
        }

@dataclass

class TokenInfo:
    """代币信息"""
    contract_address: str
    symbol: str
    name: str
    decimals: int

    # 供应量信息
    total_supply: Optional[Decimal] = None
    circulating_supply: Optional[Decimal] = None

    # 价格信息
    price_usd: Optional[Decimal] = None
    market_cap: Optional[Decimal] = None

    # 网络信息
    network: BlockchainNetwork = BlockchainNetwork.ETHEREUM_MAINNET

    def to_dict(self) -> dict:
        return {
            'contract_address': self.contract_address,
            'symbol': self.symbol,
            'name': self.name,
            'decimals': self.decimals,
            'total_supply': str(self.total_supply) if self.total_supply else None,
            'circulating_supply': str(self.circulating_supply) if self.circulating_supply else None,
            'price_usd': str(self.price_usd) if self.price_usd else None,
            'market_cap': str(self.market_cap) if self.market_cap else None,
            'network': self.network.value
        }

class BlockchainClient:
    """区块链客户端基类"""

    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.connected = False

    async def connect(self) -> bool:
        """连接到区块链网络"""
        raise NotImplementedError

    async def get_balance(self, address: str, token_address: Optional[str] = None) -> Decimal:
        """获取余额"""
        raise NotImplementedError

    async def send_transaction(self, tx_data: Dict[str, Any]) -> Transaction:
        """发送交易"""
        raise NotImplementedError

    async def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """获取交易信息"""
        raise NotImplementedError

    async def call_contract(self, contract_address: str, function_name: str,
                          *args, **kwargs) -> Any:
        """调用智能合约函数"""
        raise NotImplementedError

class EthereumClient(BlockchainClient):
    """以太坊客户端"""

    def __init__(self, config: BlockchainConfig, private_key: Optional[str] = None):
        super().__init__(config)
        self.web3 = None
        self.account = None
        self.private_key = private_key

        # 合约缓存
        self.contract_cache = {}

        # 常用ABI
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]

    async def connect(self) -> bool:
        """连接到以太坊网络"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))

            # 检查连接
            is_connected = self.web3.is_connected()

            if is_connected:
                # 设置账户（如果有私钥）
                if self.private_key:
                    self.account = self.web3.eth.account.from_key(self.private_key)
                    self.web3.eth.default_account = self.account.address

                self.connected = True
                logger.info(f"连接到 {self.config.network.value} 成功")

                # 获取网络信息
                chain_id = self.web3.eth.chain_id
                block_number = self.web3.eth.block_number

                logger.info(f"Chain ID: {chain_id}, 最新区块: {block_number}")

                return True
            else:
                logger.error("无法连接到以太坊网络")
                return False

        except Exception as e:
            logger.error(f"连接以太坊网络失败: {e}")
            return False

    async def get_balance(self, address: str, token_address: Optional[str] = None) -> Decimal:
        """获取余额"""
        if not self.connected:
            raise ValueError("未连接到区块链网络")

        try:
            if token_address:
                # ERC20代币余额
                contract = self._get_erc20_contract(token_address)
                balance_wei = contract.functions.balanceOf(address).call()
                decimals = contract.functions.decimals().call()
                balance = Decimal(balance_wei) / Decimal(10 ** decimals)
            else:
                # ETH余额
                balance_wei = self.web3.eth.get_balance(address)
                balance = Decimal(self.web3.from_wei(balance_wei, 'ether'))

            return balance

        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return Decimal(0)

    def _get_erc20_contract(self, token_address: str):
        """获取ERC20合约对象"""
        if token_address not in self.contract_cache:
            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.erc20_abi
            )
            self.contract_cache[token_address] = contract

        return self.contract_cache[token_address]

    async def get_token_info(self, token_address: str) -> Optional[TokenInfo]:
        """获取代币信息"""
        if not self.connected:
            return None

        try:
            contract = self._get_erc20_contract(token_address)

            name = contract.functions.name().call()
            symbol = contract.functions.symbol().call()
            decimals = contract.functions.decimals().call()
            total_supply_wei = contract.functions.totalSupply().call()
            total_supply = Decimal(total_supply_wei) / Decimal(10 ** decimals)

            token_info = TokenInfo(
                contract_address=token_address,
                symbol=symbol,
                name=name,
                decimals=decimals,
                total_supply=total_supply,
                network=self.config.network
            )

            return token_info

        except Exception as e:
            logger.error(f"获取代币信息失败: {e}")
            return None

    async def send_transaction(self, tx_data: Dict[str, Any]) -> Transaction:
        """发送交易"""
        if not self.connected or not self.account:
            raise ValueError("未连接到区块链网络或未设置账户")

        try:
            # 构建交易
            transaction = {
                'from': self.account.address,
                'to': tx_data['to'],
                'value': self.web3.to_wei(tx_data.get('value', 0), 'ether'),
                'gas': tx_data.get('gas', 21000),
                'gasPrice': self.web3.to_wei(tx_data.get('gasPrice', self.config.gas_price_gwei), 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            }

            if 'data' in tx_data:
                transaction['data'] = tx_data['data']

            # 签名交易
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.private_key)

            # 发送交易
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = tx_hash.hex()

            # 创建交易对象
            tx = Transaction(
                tx_hash=tx_hash_hex,
                network=self.config.network,
                from_address=self.account.address,
                to_address=tx_data['to'],
                value=Decimal(str(tx_data.get('value', 0))),
                gas_price=int(self.web3.to_wei(tx_data.get('gasPrice', self.config.gas_price_gwei), 'gwei')),
                gas_limit=tx_data.get('gas', 21000)
            )

            logger.info(f"交易已发送: {tx_hash_hex}")
            return tx

        except Exception as e:
            logger.error(f"发送交易失败: {e}")
            raise

    async def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """获取交易信息"""
        if not self.connected:
            return None

        try:
            tx = self.web3.eth.get_transaction(tx_hash)

            # 尝试获取交易收据
            try:
                receipt = self.web3.eth.get_transaction_receipt(tx_hash)
                status = TransactionStatus.CONFIRMED if receipt.status == 1 else TransactionStatus.FAILED
                gas_used = receipt.gasUsed
                block_number = receipt.blockNumber
                block_hash = receipt.blockHash.hex()
                transaction_fee = Decimal(receipt.gasUsed * tx.gasPrice) / Decimal(10**18)
                logs = [dict(log) for log in receipt.logs]
            except:
                # 交易可能还在pending状态
                status = TransactionStatus.PENDING
                gas_used = None
                block_number = None
                block_hash = None
                transaction_fee = None
                logs = []

            # 获取区块时间戳
            block_timestamp = None
            if block_number:
                try:
                    block = self.web3.eth.get_block(block_number)
                    block_timestamp = datetime.fromtimestamp(block.timestamp)
                except:
                    pass

            transaction = Transaction(
                tx_hash=tx_hash,
                network=self.config.network,
                from_address=tx['from'],
                to_address=tx['to'],
                value=Decimal(self.web3.from_wei(tx.value, 'ether')),
                gas_price=tx.gasPrice,
                gas_limit=tx.gas,
                gas_used=gas_used,
                status=status,
                block_number=block_number,
                block_hash=block_hash,
                transaction_fee=transaction_fee,
                input_data=tx.input.hex() if tx.input else None,
                logs=logs,
                block_timestamp=block_timestamp
            )

            return transaction

        except Exception as e:
            logger.error(f"获取交易信息失败: {e}")
            return None

    async def call_contract(self, contract_address: str, function_name: str,
                          abi: Optional[List] = None, *args, **kwargs) -> Any:
        """调用智能合约函数"""
        if not self.connected:
            raise ValueError("未连接到区块链网络")

        try:
            if abi is None:
                # 如果是ERC20合约，使用默认ABI
                abi = self.erc20_abi

            contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=abi
            )

            result = getattr(contract.functions, function_name)(*args).call()
            return result

        except Exception as e:
            logger.error(f"调用合约函数失败: {e}")
            raise

    async def deploy_contract(self, contract_abi: List, contract_bytecode: str,
                            constructor_args: Tuple = ()) -> Optional[SmartContract]:
        """部署智能合约"""
        if not self.connected or not self.account:
            raise ValueError("未连接到区块链网络或未设置账户")

        try:
            # 创建合约对象
            contract = self.web3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)

            # 构建部署交易
            deploy_txn = contract.constructor(*constructor_args).build_transaction({
                'from': self.account.address,
                'gas': 2000000,  # 默认gas limit
                'gasPrice': self.web3.to_wei(self.config.gas_price_gwei, 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.account.address)
            })

            # 签名并发送交易
            signed_txn = self.web3.eth.account.sign_transaction(deploy_txn, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)

            # 等待交易确认
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                # 部署成功
                contract_address = receipt.contractAddress

                smart_contract = SmartContract(
                    contract_address=contract_address,
                    contract_name="Deployed Contract",
                    contract_type=ContractType.CUSTOM,
                    network=self.config.network,
                    abi=contract_abi,
                    bytecode=contract_bytecode,
                    deployer=self.account.address,
                    deployment_tx=tx_hash.hex(),
                    deployment_block=receipt.blockNumber
                )

                logger.info(f"合约部署成功: {contract_address}")
                return smart_contract
            else:
                logger.error("合约部署失败")
                return None

        except Exception as e:
            logger.error(f"部署合约失败: {e}")
            return None

class DeFiIntegrator:
    """DeFi协议集成器"""

    def __init__(self, blockchain_client: BlockchainClient):
        self.client = blockchain_client

        # 知名DeFi协议地址（以太坊主网）
        self.protocol_addresses = {
            'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'uniswap_v2_factory': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
            'sushiswap_router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'compound_comptroller': '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B',
            'aave_lending_pool': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9'
        }

        # Uniswap V2 Router ABI (简化版)
        self.uniswap_router_abi = [
            {
                "constant": True,
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "type": "function"
            }
        ]

    async def get_token_price(self, token_address: str, base_token: str = None) -> Optional[Decimal]:
        """获取代币价格（通过Uniswap）"""
        if not base_token:
            # 默认使用USDC作为基础代币
            base_token = '0xA0b86a33E6417c7e67c86C04c8C8906185c3Cc52'  # USDC

        try:
            router_address = self.protocol_addresses['uniswap_v2_router']

            # 构建交易路径
            path = [Web3.to_checksum_address(token_address), Web3.to_checksum_address(base_token)]

            # 查询价格（1个代币能换多少USDC）
            amount_in = 10 ** 18  # 1个代币（假设18位小数）

            result = await self.client.call_contract(
                router_address,
                'getAmountsOut',
                self.uniswap_router_abi,
                amount_in,
                path
            )

            if result and len(result) >= 2:
                amount_out = result[-1]  # 最后一个是输出数量
                price = Decimal(amount_out) / Decimal(10 ** 6)  # USDC是6位小数
                return price

        except Exception as e:
            logger.error(f"获取代币价格失败: {e}")

        return None

    async def get_pool_info(self, token_a: str, token_b: str) -> Dict[str, Any]:
        """获取流动性池信息"""
        # 简化实现，实际需要调用具体的DeFi协议
        return {
            'pool_address': None,
            'reserves': [0, 0],
            'total_supply': 0,
            'fee_rate': 0.003  # 0.3%
        }

    async def estimate_swap(self, token_in: str, token_out: str,
                          amount_in: Decimal) -> Dict[str, Any]:
        """估算交换结果"""
        try:
            router_address = self.protocol_addresses['uniswap_v2_router']
            path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]

            amount_in_wei = int(amount_in * Decimal(10 ** 18))

            result = await self.client.call_contract(
                router_address,
                'getAmountsOut',
                self.uniswap_router_abi,
                amount_in_wei,
                path
            )

            if result and len(result) >= 2:
                amount_out_wei = result[-1]
                amount_out = Decimal(amount_out_wei) / Decimal(10 ** 18)

                return {
                    'amount_in': amount_in,
                    'amount_out': amount_out,
                    'price_impact': 0.01,  # 简化计算
                    'minimum_amount_out': amount_out * Decimal(0.95),  # 5%滑点保护
                    'path': path
                }

        except Exception as e:
            logger.error(f"估算交换失败: {e}")

        return {}

class BlockchainIntegrationSystem:
    """区块链集成系统主类"""

    def __init__(self):
        self.clients = {}
        self.contracts = {}
        self.transactions = {}

        # 预定义网络配置
        self.network_configs = {
            BlockchainNetwork.ETHEREUM_MAINNET: BlockchainConfig(
                network=BlockchainNetwork.ETHEREUM_MAINNET,
                rpc_url="https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
                chain_id=1,
                native_token="ETH",
                block_explorer_url="https://etherscan.io",
                average_block_time=15.0,
                gas_price_gwei=20.0
            ),
            BlockchainNetwork.BSC_MAINNET: BlockchainConfig(
                network=BlockchainNetwork.BSC_MAINNET,
                rpc_url="https://bsc-dataseed1.binance.org/",
                chain_id=56,
                native_token="BNB",
                block_explorer_url="https://bscscan.com",
                average_block_time=3.0,
                gas_price_gwei=5.0
            ),
            BlockchainNetwork.POLYGON_MAINNET: BlockchainConfig(
                network=BlockchainNetwork.POLYGON_MAINNET,
                rpc_url="https://polygon-rpc.com/",
                chain_id=137,
                native_token="MATIC",
                block_explorer_url="https://polygonscan.com",
                average_block_time=2.0,
                gas_price_gwei=30.0
            )
        }

        # DeFi集成器
        self.defi_integrators = {}

        # 统计信息
        self.stats = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'total_contracts': 0,
            'networks_connected': 0
        }

        logger.info("区块链集成系统初始化完成")

    async def connect_to_network(self, network: BlockchainNetwork,
                               private_key: Optional[str] = None,
                               custom_config: Optional[BlockchainConfig] = None) -> bool:
        """连接到区块链网络"""

        config = custom_config or self.network_configs.get(network)
        if not config:
            logger.error(f"不支持的区块链网络: {network}")
            return False

        try:
            # 创建客户端
            if network.value.startswith('ethereum') or network.value.startswith('bsc') or network.value.startswith('polygon'):
                client = EthereumClient(config, private_key)
            else:
                logger.error(f"暂不支持的区块链类型: {network}")
                return False

            # 连接
            success = await client.connect()

            if success:
                self.clients[network] = client

                # 创建DeFi集成器
                self.defi_integrators[network] = DeFiIntegrator(client)

                self.stats['networks_connected'] += 1
                logger.info(f"成功连接到 {network.value}")

            return success

        except Exception as e:
            logger.error(f"连接到 {network.value} 失败: {e}")
            return False

    async def get_balance(self, network: BlockchainNetwork, address: str,
                        token_address: Optional[str] = None) -> Decimal:
        """获取余额"""
        client = self.clients.get(network)
        if not client:
            logger.error(f"未连接到网络: {network}")
            return Decimal(0)

        return await client.get_balance(address, token_address)

    async def send_transaction(self, network: BlockchainNetwork,
                             tx_data: Dict[str, Any]) -> Optional[Transaction]:
        """发送交易"""
        client = self.clients.get(network)
        if not client:
            logger.error(f"未连接到网络: {network}")
            return None

        try:
            transaction = await client.send_transaction(tx_data)

            # 记录交易
            self.transactions[transaction.tx_hash] = transaction
            self.stats['total_transactions'] += 1

            return transaction

        except Exception as e:
            logger.error(f"发送交易失败: {e}")
            self.stats['failed_transactions'] += 1
            return None

    async def get_transaction_status(self, network: BlockchainNetwork,
                                   tx_hash: str) -> Optional[Transaction]:
        """获取交易状态"""
        client = self.clients.get(network)
        if not client:
            return None

        # 首先从缓存获取
        if tx_hash in self.transactions:
            cached_tx = self.transactions[tx_hash]
            if cached_tx.status != TransactionStatus.PENDING:
                return cached_tx

        # 从区块链获取最新状态
        transaction = await client.get_transaction(tx_hash)

        if transaction:
            # 更新缓存
            self.transactions[tx_hash] = transaction

            # 更新统计
            if transaction.status == TransactionStatus.CONFIRMED:
                self.stats['successful_transactions'] += 1
            elif transaction.status == TransactionStatus.FAILED:
                self.stats['failed_transactions'] += 1

        return transaction

    async def get_token_info(self, network: BlockchainNetwork,
                           token_address: str) -> Optional[TokenInfo]:
        """获取代币信息"""
        client = self.clients.get(network)
        if not client or not isinstance(client, EthereumClient):
            return None

        return await client.get_token_info(token_address)

    async def get_token_price(self, network: BlockchainNetwork,
                            token_address: str) -> Optional[Decimal]:
        """获取代币价格"""
        defi_integrator = self.defi_integrators.get(network)
        if not defi_integrator:
            return None

        return await defi_integrator.get_token_price(token_address)

    async def deploy_contract(self, network: BlockchainNetwork,
                            contract_name: str, abi: List, bytecode: str,
                            constructor_args: Tuple = ()) -> Optional[SmartContract]:
        """部署智能合约"""
        client = self.clients.get(network)
        if not client or not isinstance(client, EthereumClient):
            logger.error(f"不支持在 {network} 部署合约或未连接")
            return None

        contract = await client.deploy_contract(abi, bytecode, constructor_args)

        if contract:
            contract.contract_name = contract_name
            self.contracts[contract.contract_address] = contract
            self.stats['total_contracts'] += 1

        return contract

    async def monitor_transactions(self, tx_hashes: List[str],
                                 network: BlockchainNetwork) -> Dict[str, Transaction]:
        """监控交易状态"""
        results = {}

        for tx_hash in tx_hashes:
            transaction = await self.get_transaction_status(network, tx_hash)
            if transaction:
                results[tx_hash] = transaction

        return results

    def get_supported_networks(self) -> List[Dict[str, Any]]:
        """获取支持的网络列表"""
        networks = []

        for network, config in self.network_configs.items():
            network_info = config.to_dict()
            network_info['connected'] = network in self.clients
            networks.append(network_info)

        return networks

    def get_portfolio_summary(self, address: str) -> Dict[str, Any]:
        """获取投资组合摘要"""
        summary = {
            'address': address,
            'networks': {},
            'total_value_usd': Decimal(0),
            'token_holdings': []
        }

        # 这里需要异步操作，简化处理
        for network in self.clients.keys():
            summary['networks'][network.value] = {
                'connected': True,
                'native_balance': '0',
                'tokens': []
            }

        return summary

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            'stats': self.stats,
            'connected_networks': list(self.clients.keys()),
            'total_contracts': len(self.contracts),
            'recent_transactions': [
                tx.to_dict() for tx in list(self.transactions.values())[-5:]
            ]
        }

# 便利函数

def get_network_config(network: BlockchainNetwork) -> Optional[BlockchainConfig]:
    """获取网络配置"""
    configs = {
        BlockchainNetwork.ETHEREUM_MAINNET: BlockchainConfig(
            network=BlockchainNetwork.ETHEREUM_MAINNET,
            rpc_url="https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
            chain_id=1,
            native_token="ETH",
            block_explorer_url="https://etherscan.io",
            average_block_time=15.0
        )
    }
    return configs.get(network)

# 全局实例
_blockchain_integration_system_instance = None

def get_blockchain_integration_system() -> BlockchainIntegrationSystem:
    """获取区块链集成系统实例"""
    global _blockchain_integration_system_instance
    if _blockchain_integration_system_instance is None:
        _blockchain_integration_system_instance = BlockchainIntegrationSystem()
    return _blockchain_integration_system_instance
