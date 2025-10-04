"""
Exchange Manager Module
Handles connections and operations with multiple cryptocurrency exchanges
Supports Binance, Coinbase, Kraken with unified interface
"""

import asyncio
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
from decimal import Decimal, ROUND_DOWN
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import retry_decorator, timing_decorator

logger = setup_logger(__name__)

class ExchangeManager:
    """Unified exchange management for multiple cryptocurrency exchanges"""

    def __init__(self):
        self.exchanges = {}
        self.exchange_configs = {
            'binance': {
                'class': ccxt.binance,
                'async_class': ccxt_async.binance,
                'testnet': config.get('BINANCE_TESTNET', True),
                'sandbox': config.get('BINANCE_TESTNET', True)
            },
            'coinbase': {
                'class': ccxt.coinbasepro,
                'async_class': ccxt_async.coinbasepro,
                'testnet': False,  # Coinbase doesn't have testnet
                'sandbox': True
            },
            'kraken': {
                'class': ccxt.kraken,
                'async_class': ccxt_async.kraken,
                'testnet': False,
                'sandbox': False
            }
        }

        self.rate_limits = {}
        self.last_requests = {}
        self.exchange_info = {}

    async def initialize_exchanges(self, exchanges: List[str] = None) -> bool:
        """Initialize specified exchanges or all available"""
        try:
            if exchanges is None:
                exchanges = ['binance']  # Default to Binance

            for exchange_name in exchanges:
                success = await self._initialize_exchange(exchange_name)
                if not success:
                    logger.error(f"Failed to initialize {exchange_name}")
                    return False

            logger.info(f"Successfully initialized {len(self.exchanges)} exchanges")
            return True

        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            return False

    async def _initialize_exchange(self, exchange_name: str) -> bool:
        """Initialize a specific exchange"""
        try:
            if exchange_name not in self.exchange_configs:
                logger.error(f"Unsupported exchange: {exchange_name}")
                return False

            config_data = self.exchange_configs[exchange_name]

            # Get API credentials
            api_key = config.get(f'{exchange_name.upper()}_API_KEY')
            api_secret = config.get(f'{exchange_name.upper()}_API_SECRET')

            if not api_key or not api_secret:
                logger.error(f"Missing API credentials for {exchange_name}")
                return False

            # Exchange configuration
            exchange_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'rateLimit': 1200,  # milliseconds
                'timeout': 30000,   # 30 seconds
            }

            # Add exchange-specific settings
            if exchange_name == 'binance':
                if config_data['testnet']:
                    exchange_config['sandbox'] = True
                    exchange_config['urls'] = {
                        'api': 'https://testnet.binance.vision',
                        'test': 'https://testnet.binance.vision'
                    }

            elif exchange_name == 'coinbase':
                passphrase = config.get('COINBASE_PASSPHRASE')
                if passphrase:
                    exchange_config['password'] = passphrase
                if config_data['sandbox']:
                    exchange_config['sandbox'] = True

            # Initialize both sync and async versions
            sync_exchange = config_data['class'](exchange_config)
            async_exchange = config_data['async_class'](exchange_config)

            # Test connection
            await self._test_exchange_connection(async_exchange, exchange_name)

            # Store exchanges
            self.exchanges[exchange_name] = {
                'sync': sync_exchange,
                'async': async_exchange,
                'config': exchange_config,
                'last_used': time.time()
            }

            # Load exchange info
            await self._load_exchange_info(exchange_name)

            # Initialize rate limiting
            self.rate_limits[exchange_name] = {
                'requests_per_minute': 1200,
                'requests_this_minute': 0,
                'minute_start': time.time()
            }

            logger.info(f"Successfully initialized {exchange_name} exchange")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {exchange_name}: {e}")
            return False

    async def _test_exchange_connection(self, exchange, exchange_name: str):
        """Test exchange connection"""
        try:
            # Test basic connectivity
            await exchange.load_markets()

            # Test account access (if not sandbox)
            if not getattr(exchange, 'sandbox', False):
                balance = await exchange.fetch_balance()
                logger.info(f"{exchange_name} connection test successful")
            else:
                logger.info(f"{exchange_name} testnet connection successful")

        except Exception as e:
            logger.warning(f"{exchange_name} connection test failed: {e}")
            raise

    async def _load_exchange_info(self, exchange_name: str):
        """Load exchange information and trading rules"""
        try:
            exchange = self.exchanges[exchange_name]['async']

            # Load markets
            markets = await exchange.load_markets()

            # Get exchange info
            exchange_info = {
                'markets': markets,
                'symbols': list(markets.keys()),
                'base_currencies': list(set([market['base'] for market in markets.values()])),
                'quote_currencies': list(set([market['quote'] for market in markets.values()])),
                'trading_fees': await self._get_trading_fees(exchange, exchange_name),
                'min_order_sizes': self._extract_min_order_sizes(markets),
                'price_precisions': self._extract_price_precisions(markets),
                'amount_precisions': self._extract_amount_precisions(markets)
            }

            self.exchange_info[exchange_name] = exchange_info
            logger.info(f"Loaded info for {len(markets)} markets on {exchange_name}")

        except Exception as e:
            logger.error(f"Failed to load exchange info for {exchange_name}: {e}")

    async def _get_trading_fees(self, exchange, exchange_name: str) -> Dict[str, float]:
        """Get trading fees for the exchange"""
        try:
            if hasattr(exchange, 'fetch_trading_fees'):
                fees = await exchange.fetch_trading_fees()
                return {
                    'maker': fees.get('maker', 0.001),
                    'taker': fees.get('taker', 0.001)
                }
            else:
                # Default fees by exchange
                default_fees = {
                    'binance': {'maker': 0.001, 'taker': 0.001},
                    'coinbase': {'maker': 0.005, 'taker': 0.005},
                    'kraken': {'maker': 0.0016, 'taker': 0.0026}
                }
                return default_fees.get(exchange_name, {'maker': 0.001, 'taker': 0.001})

        except Exception as e:
            logger.warning(f"Could not fetch trading fees for {exchange_name}: {e}")
            return {'maker': 0.001, 'taker': 0.001}

    def _extract_min_order_sizes(self, markets: Dict) -> Dict[str, float]:
        """Extract minimum order sizes for each symbol"""
        min_sizes = {}
        for symbol, market in markets.items():
            try:
                min_size = market.get('limits', {}).get('amount', {}).get('min', 0.001)
                min_sizes[symbol] = float(min_size) if min_size else 0.001
            except:
                min_sizes[symbol] = 0.001
        return min_sizes

    def _extract_price_precisions(self, markets: Dict) -> Dict[str, int]:
        """Extract price precisions for each symbol"""
        precisions = {}
        for symbol, market in markets.items():
            try:
                precision = market.get('precision', {}).get('price', 8)
                precisions[symbol] = int(precision) if precision else 8
            except:
                precisions[symbol] = 8
        return precisions

    def _extract_amount_precisions(self, markets: Dict) -> Dict[str, int]:
        """Extract amount precisions for each symbol"""
        precisions = {}
        for symbol, market in markets.items():
            try:
                precision = market.get('precision', {}).get('amount', 8)
                precisions[symbol] = int(precision) if precision else 8
            except:
                precisions[symbol] = 8
        return precisions

    @retry_decorator(max_retries=3, delay=1.0)

    async def fetch_ohlcv(self, symbol: str, timeframe: str,
                         since: datetime = None, limit: int = 1000,
                         exchange: str = 'binance') -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")

            # Rate limiting
            await self._check_rate_limit(exchange)

            exchange_obj = self.exchanges[exchange]['async']

            # Convert datetime to timestamp
            since_timestamp = None
            if since:
                since_timestamp = int(since.timestamp() * 1000)

            # Fetch data
            ohlcv_data = await exchange_obj.fetch_ohlcv(
                symbol, timeframe, since_timestamp, limit
            )

            if not ohlcv_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.debug(f"Fetched {len(df)} {timeframe} candles for {symbol} from {exchange}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data: {e}")
            return pd.DataFrame()

    @retry_decorator(max_retries=3, delay=0.5)

    async def fetch_ticker(self, symbol: str, exchange: str = 'binance') -> Dict[str, Any]:
        """Fetch current ticker data"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")

            await self._check_rate_limit(exchange)

            exchange_obj = self.exchanges[exchange]['async']
            ticker = await exchange_obj.fetch_ticker(symbol)

            return {
                'symbol': symbol,
                'price': float(ticker['last']),
                'bid': float(ticker['bid']) if ticker['bid'] else None,
                'ask': float(ticker['ask']) if ticker['ask'] else None,
                'volume': float(ticker['baseVolume']) if ticker['baseVolume'] else 0,
                'change': float(ticker['change']) if ticker['change'] else 0,
                'percentage': float(ticker['percentage']) if ticker['percentage'] else 0,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {}

    @retry_decorator(max_retries=3, delay=0.5)

    async def fetch_balance(self, exchange: str = 'binance') -> Dict[str, Any]:
        """Fetch account balance"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")

            await self._check_rate_limit(exchange)

            exchange_obj = self.exchanges[exchange]['async']
            balance = await exchange_obj.fetch_balance()

            # Process balance data
            processed_balance = {
                'total': {},
                'free': {},
                'used': {},
                'timestamp': datetime.now()
            }

            for currency, amounts in balance.items():
                if currency not in ['info', 'timestamp', 'datetime', 'free', 'used', 'total']:
                    if isinstance(amounts, dict):
                        processed_balance['total'][currency] = float(amounts.get('total', 0))
                        processed_balance['free'][currency] = float(amounts.get('free', 0))
                        processed_balance['used'][currency] = float(amounts.get('used', 0))

            return processed_balance

        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    async def place_order(self, symbol: str, side: str, amount: float,
                         order_type: str = 'market', price: float = None,
                         exchange: str = 'binance', **kwargs) -> Dict[str, Any]:
        """Place a trading order"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")

            # Validate order parameters
            if not self._validate_order_params(symbol, side, amount, order_type, price, exchange):
                return {'status': 'failed', 'error': 'Invalid order parameters'}

            # Check if trading is enabled
            if not config.get('ENABLE_LIVE_TRADING', False):
                logger.warning("Live trading is disabled - order not placed")
                return {
                    'id': f'paper_{int(time.time())}',
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'type': order_type,
                    'price': price,
                    'status': 'simulated',
                    'timestamp': datetime.now()
                }

            await self._check_rate_limit(exchange)

            exchange_obj = self.exchanges[exchange]['async']

            # Adjust amount precision
            amount = self._adjust_order_precision(symbol, amount, exchange, 'amount')
            if price:
                price = self._adjust_order_precision(symbol, price, exchange, 'price')

            # Place order
            if order_type.lower() == 'market':
                order = await exchange_obj.create_market_order(symbol, side, amount)
            elif order_type.lower() == 'limit':
                if not price:
                    raise ValueError("Price required for limit orders")
                order = await exchange_obj.create_limit_order(symbol, side, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Process order response
            processed_order = {
                'id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'amount': float(order['amount']),
                'type': order['type'],
                'price': float(order['price']) if order['price'] else None,
                'filled': float(order['filled']) if order['filled'] else 0,
                'remaining': float(order['remaining']) if order['remaining'] else 0,
                'status': order['status'],
                'cost': float(order['cost']) if order['cost'] else 0,
                'fee': order.get('fee', {}),
                'timestamp': datetime.now(),
                'exchange': exchange
            }

            logger.info(f"Order placed: {processed_order}")
            return processed_order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def cancel_order(self, order_id: str, symbol: str,
                          exchange: str = 'binance') -> Dict[str, Any]:
        """Cancel an existing order"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")

            if not config.get('ENABLE_LIVE_TRADING', False):
                logger.warning("Live trading is disabled - order cancellation simulated")
                return {
                    'id': order_id,
                    'status': 'canceled',
                    'timestamp': datetime.now()
                }

            await self._check_rate_limit(exchange)

            exchange_obj = self.exchanges[exchange]['async']
            result = await exchange_obj.cancel_order(order_id, symbol)

            logger.info(f"Order {order_id} canceled")
            return result

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def fetch_order_status(self, order_id: str, symbol: str,
                                exchange: str = 'binance') -> Dict[str, Any]:
        """Fetch order status"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")

            await self._check_rate_limit(exchange)

            exchange_obj = self.exchanges[exchange]['async']
            order = await exchange_obj.fetch_order(order_id, symbol)

            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'status': order['status'],
                'filled': float(order['filled']) if order['filled'] else 0,
                'remaining': float(order['remaining']) if order['remaining'] else 0,
                'cost': float(order['cost']) if order['cost'] else 0,
                'average': float(order['average']) if order['average'] else None,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to fetch order status: {e}")
            return {}

    async def fetch_open_orders(self, symbol: str = None,
                               exchange: str = 'binance') -> List[Dict[str, Any]]:
        """Fetch open orders"""
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")

            await self._check_rate_limit(exchange)

            exchange_obj = self.exchanges[exchange]['async']
            orders = await exchange_obj.fetch_open_orders(symbol)

            return [
                {
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'amount': float(order['amount']),
                    'price': float(order['price']) if order['price'] else None,
                    'type': order['type'],
                    'status': order['status'],
                    'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000) if order['timestamp'] else None
                }
                for order in orders
            ]

        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    def _validate_order_params(self, symbol: str, side: str, amount: float,
                              order_type: str, price: float, exchange: str) -> bool:
        """Validate order parameters"""
        try:
            # Check symbol exists
            if exchange not in self.exchange_info:
                return False

            if symbol not in self.exchange_info[exchange]['symbols']:
                logger.error(f"Symbol {symbol} not found on {exchange}")
                return False

            # Check side
            if side.lower() not in ['buy', 'sell']:
                logger.error(f"Invalid side: {side}")
                return False

            # Check amount
            min_amount = self.exchange_info[exchange]['min_order_sizes'].get(symbol, 0.001)
            if amount < min_amount:
                logger.error(f"Amount {amount} below minimum {min_amount} for {symbol}")
                return False

            # Check order type
            if order_type.lower() not in ['market', 'limit']:
                logger.error(f"Unsupported order type: {order_type}")
                return False

            # Check price for limit orders
            if order_type.lower() == 'limit' and not price:
                logger.error("Price required for limit orders")
                return False

            return True

        except Exception as e:
            logger.error(f"Order validation failed: {e}")
            return False

    def _adjust_order_precision(self, symbol: str, value: float,
                               exchange: str, precision_type: str) -> float:
        """Adjust order value to exchange precision requirements"""
        try:
            if exchange not in self.exchange_info:
                return value

            if precision_type == 'amount':
                precision = self.exchange_info[exchange]['amount_precisions'].get(symbol, 8)
            else:  # price
                precision = self.exchange_info[exchange]['price_precisions'].get(symbol, 8)

            # Round down to required precision
            multiplier = 10 ** precision
            return float(int(value * multiplier) / multiplier)

        except Exception as e:
            logger.error(f"Precision adjustment failed: {e}")
            return value

    async def _check_rate_limit(self, exchange: str):
        """Check and enforce rate limits"""
        try:
            if exchange not in self.rate_limits:
                return

            rate_limit = self.rate_limits[exchange]
            current_time = time.time()

            # Reset counter if a minute has passed
            if current_time - rate_limit['minute_start'] >= 60:
                rate_limit['requests_this_minute'] = 0
                rate_limit['minute_start'] = current_time

            # Check if we're over the limit
            if rate_limit['requests_this_minute'] >= rate_limit['requests_per_minute']:
                sleep_time = 60 - (current_time - rate_limit['minute_start'])
                logger.warning(f"Rate limit reached for {exchange}, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

                # Reset after sleep
                rate_limit['requests_this_minute'] = 0
                rate_limit['minute_start'] = time.time()

            rate_limit['requests_this_minute'] += 1

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")

    async def get_exchange_status(self, exchange: str) -> Dict[str, Any]:
        """Get exchange connection and status information"""
        try:
            if exchange not in self.exchanges:
                return {'status': 'not_initialized'}

            exchange_obj = self.exchanges[exchange]['async']

            # Test connection
            try:
                await exchange_obj.fetch_status()
                status = 'online'
            except:
                status = 'offline'

            # Get last used time
            last_used = self.exchanges[exchange]['last_used']

            # Get rate limit info
            rate_limit_info = self.rate_limits.get(exchange, {})

            return {
                'status': status,
                'last_used': datetime.fromtimestamp(last_used),
                'rate_limit': rate_limit_info,
                'markets_count': len(self.exchange_info.get(exchange, {}).get('symbols', [])),
                'testnet': self.exchanges[exchange]['config'].get('sandbox', False)
            }

        except Exception as e:
            logger.error(f"Failed to get exchange status: {e}")
            return {'status': 'error', 'error': str(e)}

    async def close_all_connections(self):
        """Close all exchange connections"""
        try:
            for exchange_name, exchange_data in self.exchanges.items():
                try:
                    await exchange_data['async'].close()
                    logger.info(f"Closed connection to {exchange_name}")
                except Exception as e:
                    logger.error(f"Failed to close {exchange_name}: {e}")

            self.exchanges.clear()
            logger.info("All exchange connections closed")

        except Exception as e:
            logger.error(f"Failed to close connections: {e}")

    def get_supported_symbols(self, exchange: str = 'binance') -> List[str]:
        """Get list of supported trading symbols"""
        if exchange in self.exchange_info:
            return self.exchange_info[exchange]['symbols']
        return []

    def get_exchange_info_summary(self) -> Dict[str, Any]:
        """Get summary of all exchange information"""
        summary = {}

        for exchange_name, info in self.exchange_info.items():
            summary[exchange_name] = {
                'symbols_count': len(info['symbols']),
                'base_currencies': len(info['base_currencies']),
                'quote_currencies': len(info['quote_currencies']),
                'trading_fees': info['trading_fees'],
                'status': 'initialized' if exchange_name in self.exchanges else 'not_initialized'
            }

        return summary

# Convenience functions

async def create_exchange_manager(exchanges: List[str] = None) -> ExchangeManager:
    """Create and initialize exchange manager"""
    manager = ExchangeManager()
    await manager.initialize_exchanges(exchanges)
    return manager

async def fetch_latest_price(symbol: str, exchange: str = 'binance') -> float:
    """Quick function to fetch latest price"""
    manager = ExchangeManager()
    await manager.initialize_exchanges([exchange])

    ticker = await manager.fetch_ticker(symbol, exchange)
    await manager.close_all_connections()

    return ticker.get('price', 0.0)

# Usage example
if __name__ == "__main__":

    async def test_exchange_manager():
        # Create exchange manager
        manager = ExchangeManager()

        # Initialize Binance
        await manager.initialize_exchanges(['binance'])

        # Fetch some data
        ticker = await manager.fetch_ticker('BTC/USDT')
        print(f"BTC/USDT price: {ticker.get('price', 'N/A')}")

        # Fetch OHLCV data
        ohlcv = await manager.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        print(f"Fetched {len(ohlcv)} OHLCV records")

        # Get balance (if API keys are configured)
        try:
            balance = await manager.fetch_balance()
            print(f"Account balance: {len(balance.get('total', {}))} currencies")
        except Exception as e:
            print(f"Balance fetch failed (expected if using testnet): {e}")

        # Get exchange status
        status = await manager.get_exchange_status('binance')
        print(f"Exchange status: {status}")

        # Close connections
        await manager.close_all_connections()
        print("Exchange manager test completed!")

    # Run test
    asyncio.run(test_exchange_manager())
