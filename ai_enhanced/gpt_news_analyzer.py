"""
GPT News Analyzer
基于GPT的新闻分析器，利用大语言模型分析金融新闻和市场事件
提供情感分析、事件提取、影响评估和投资建议生成功能
"""

import asyncio
import aiohttp
import openai
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import json
import re
import numpy as np
from urllib.parse import urlparse
import hashlib

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class NewsCategory(Enum):
    """新闻类别"""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    ECONOMIC_DATA = "economic_data"
    CORPORATE_NEWS = "corporate_news"
    MARKET_ANALYSIS = "market_analysis"
    GEOPOLITICAL = "geopolitical"
    TECHNOLOGY = "technology"
    ENERGY = "energy"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    OTHER = "other"

class Sentiment(Enum):
    """情感倾向"""
    VERY_POSITIVE = 2
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    VERY_NEGATIVE = -2

class ImpactLevel(Enum):
    """影响级别"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

@dataclass

class NewsArticle:
    """新闻文章"""
    article_id: str
    title: str
    content: str
    source: str
    author: Optional[str] = None

    # 时间信息
    published_at: datetime = field(default_factory=datetime.now)
    scraped_at: datetime = field(default_factory=datetime.now)

    # 元数据
    url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mentioned_symbols: List[str] = field(default_factory=list)

    # 质量指标
    credibility_score: float = 0.5
    word_count: int = 0
    language: str = "en"

    def to_dict(self) -> dict:
        return {
            'article_id': self.article_id,
            'title': self.title,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'source': self.source,
            'author': self.author,
            'published_at': self.published_at.isoformat(),
            'scraped_at': self.scraped_at.isoformat(),
            'url': self.url,
            'tags': self.tags,
            'mentioned_symbols': self.mentioned_symbols,
            'credibility_score': self.credibility_score,
            'word_count': self.word_count,
            'language': self.language
        }

@dataclass

class NewsAnalysis:
    """新闻分析结果"""
    article_id: str
    analysis_timestamp: datetime

    # 基础分析
    category: NewsCategory
    sentiment: Sentiment
    sentiment_score: float  # -1 到 1
    confidence: float  # 0 到 1

    # 实体提取
    companies: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    financial_metrics: Dict[str, float] = field(default_factory=dict)

    # 事件分析
    key_events: List[str] = field(default_factory=list)
    event_timeline: List[Dict[str, Any]] = field(default_factory=list)

    # 影响评估
    market_impact: ImpactLevel = ImpactLevel.MINIMAL
    affected_sectors: List[str] = field(default_factory=list)
    affected_stocks: List[str] = field(default_factory=list)

    # GPT生成内容
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    investment_implications: str = ""
    risk_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'article_id': self.article_id,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'category': self.category.value,
            'sentiment': self.sentiment.value,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'companies': self.companies,
            'people': self.people,
            'locations': self.locations,
            'financial_metrics': self.financial_metrics,
            'key_events': self.key_events,
            'event_timeline': self.event_timeline,
            'market_impact': self.market_impact.value,
            'affected_sectors': self.affected_sectors,
            'affected_stocks': self.affected_stocks,
            'summary': self.summary,
            'key_points': self.key_points,
            'investment_implications': self.investment_implications,
            'risk_factors': self.risk_factors
        }

class GPTClient:
    """GPT客户端"""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.api_key = api_key or config.get('openai', {}).get('api_key')
        self.model = model
        self.client = None

        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            logger.warning("未配置OpenAI API密钥，将使用模拟分析")

        # 请求统计
        self.request_count = 0
        self.token_usage = {'prompt': 0, 'completion': 0, 'total': 0}
        self.error_count = 0

    async def analyze_news(self, article: NewsArticle, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """分析新闻"""
        if not self.client:
            return self._mock_analysis(article)

        try:
            # 构建提示词
            prompt = self._build_prompt(article, analysis_type)

            # 调用GPT API
            response = await self._call_gpt_api(prompt)

            # 解析响应
            analysis_result = self._parse_response(response)

            # 更新统计
            self.request_count += 1
            if hasattr(response, 'usage'):
                self.token_usage['prompt'] += response.usage.prompt_tokens
                self.token_usage['completion'] += response.usage.completion_tokens
                self.token_usage['total'] += response.usage.total_tokens

            return analysis_result

        except Exception as e:
            logger.error(f"GPT分析失败: {e}")
            self.error_count += 1
            return self._mock_analysis(article)

    def _build_prompt(self, article: NewsArticle, analysis_type: str) -> str:
        """构建提示词"""
        base_prompt = f"""
        请分析以下金融新闻文章，并提供详细的分析报告：

        标题: {article.title}
        来源: {article.source}
        发布时间: {article.published_at.strftime('%Y-%m-%d %H:%M:%S')}

        内容:
        {article.content[:2000]}...

        请按照以下JSON格式提供分析结果：
        {{
            "category": "新闻类别 (earnings/merger_acquisition/regulatory/economic_data/corporate_news/market_analysis/geopolitical/technology/energy/healthcare/financial/other)",
            "sentiment": "情感倾向 (-2到2的整数)",
            "sentiment_score": "情感分数 (-1.0到1.0的浮点数)",
            "confidence": "分析置信度 (0.0到1.0的浮点数)",
            "companies": ["提到的公司名称列表"],
            "people": ["提到的人名列表"],
            "locations": ["提到的地点列表"],
            "financial_metrics": {{"指标名": 数值}},
            "key_events": ["关键事件列表"],
            "market_impact": "市场影响级别 (high/medium/low/minimal)",
            "affected_sectors": ["受影响的行业列表"],
            "affected_stocks": ["可能受影响的股票代码列表"],
            "summary": "新闻摘要 (100-200字)",
            "key_points": ["关键要点列表 (3-5个要点)"],
            "investment_implications": "投资影响分析 (150-300字)",
            "risk_factors": ["风险因素列表"]
        }}

        分析要求：
        1. 准确识别新闻类别和情感倾向
        2. 提取所有相关的公司、人名和地点
        3. 识别关键数字指标和财务数据
        4. 评估对市场和特定行业的影响
        5. 提供客观的投资建议和风险提示
        """

        if analysis_type == "sentiment_only":
            base_prompt += "\n特别关注情感分析和市场情绪评估。"
        elif analysis_type == "event_extraction":
            base_prompt += "\n特别关注事件提取和时间线分析。"
        elif analysis_type == "impact_assessment":
            base_prompt += "\n特别关注市场影响评估和行业分析。"

        return base_prompt

    async def _call_gpt_api(self, prompt: str) -> Any:
        """调用GPT API"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的金融分析师，擅长分析新闻对市场的影响。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response
        except Exception as e:
            logger.error(f"GPT API调用失败: {e}")
            raise

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """解析GPT响应"""
        try:
            content = response.choices[0].message.content

            # 提取JSON部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # 如果没有找到JSON，尝试解析文本
                return self._parse_text_response(content)

        except Exception as e:
            logger.error(f"解析GPT响应失败: {e}")
            return {}

    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """解析文本响应"""
        # 简单的文本解析逻辑
        result = {
            "category": "other",
            "sentiment": 0,
            "sentiment_score": 0.0,
            "confidence": 0.5,
            "companies": [],
            "people": [],
            "locations": [],
            "financial_metrics": {},
            "key_events": [],
            "market_impact": "low",
            "affected_sectors": [],
            "affected_stocks": [],
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "key_points": [],
            "investment_implications": "",
            "risk_factors": []
        }

        # 简单的情感分析
        positive_words = ['上涨', '增长', '盈利', '成功', '突破', '超预期']
        negative_words = ['下跌', '亏损', '失败', '危机', '风险', '下调']

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count:
            result['sentiment'] = 1
            result['sentiment_score'] = 0.6
        elif neg_count > pos_count:
            result['sentiment'] = -1
            result['sentiment_score'] = -0.6

        return result

    def _mock_analysis(self, article: NewsArticle) -> Dict[str, Any]:
        """模拟分析结果"""
        # 简单的关键词匹配
        title_content = f"{article.title} {article.content}".lower()

        # 确定类别
        category_keywords = {
            'earnings': ['财报', '业绩', '收益', '盈利'],
            'merger_acquisition': ['收购', '合并', '并购'],
            'regulatory': ['监管', '政策', '法规'],
            'economic_data': ['gdp', 'cpi', '经济数据', '通胀'],
            'technology': ['科技', '技术', '创新', 'ai', '人工智能']
        }

        category = NewsCategory.OTHER
        for cat, keywords in category_keywords.items():
            if any(keyword in title_content for keyword in keywords):
                category = NewsCategory(cat)
                break

        # 简单情感分析
        positive_keywords = ['上涨', '增长', '盈利', '成功', '利好']
        negative_keywords = ['下跌', '亏损', '风险', '危机', '利空']

        pos_score = sum(1 for kw in positive_keywords if kw in title_content)
        neg_score = sum(1 for kw in negative_keywords if kw in title_content)

        sentiment_score = (pos_score - neg_score) / max(pos_score + neg_score, 1) * 0.8
        sentiment = Sentiment.NEUTRAL
        if sentiment_score > 0.3:
            sentiment = Sentiment.POSITIVE
        elif sentiment_score < -0.3:
            sentiment = Sentiment.NEGATIVE

        return {
            "category": category.value,
            "sentiment": sentiment.value,
            "sentiment_score": sentiment_score,
            "confidence": 0.6,
            "companies": article.mentioned_symbols[:3],
            "people": [],
            "locations": [],
            "financial_metrics": {},
            "key_events": [article.title],
            "market_impact": "medium" if abs(sentiment_score) > 0.5 else "low",
            "affected_sectors": [],
            "affected_stocks": article.mentioned_symbols,
            "summary": article.title,
            "key_points": [article.title],
            "investment_implications": "需要进一步关注市场反应",
            "risk_factors": ["市场波动风险"]
        }

class NewsSource:
    """新闻源"""

    def __init__(self, name: str, base_url: str, credibility_score: float = 0.5):
        self.name = name
        self.base_url = base_url
        self.credibility_score = credibility_score
        self.article_count = 0
        self.last_fetch_time = None
        self.rate_limit = 100  # 每小时最大请求数
        self.request_history = deque(maxlen=self.rate_limit)

    async def fetch_articles(self, limit: int = 10) -> List[NewsArticle]:
        """获取新闻文章"""
        # 检查速率限制
        if not self._can_make_request():
            logger.warning(f"新闻源 {self.name} 达到速率限制")
            return []

        # 模拟获取新闻
        articles = []
        for i in range(limit):
            article = self._create_mock_article(i)
            articles.append(article)

        self.article_count += len(articles)
        self.last_fetch_time = datetime.now()
        self.request_history.append(datetime.now())

        return articles

    def _can_make_request(self) -> bool:
        """检查是否可以发起请求"""
        now = datetime.now()
        cutoff_time = now - timedelta(hours=1)

        # 清理过期的请求记录
        while self.request_history and self.request_history[0] < cutoff_time:
            self.request_history.popleft()

        return len(self.request_history) < self.rate_limit

    def _create_mock_article(self, index: int) -> NewsArticle:
        """创建模拟新闻文章"""
        mock_titles = [
            "苹果公司第四季度业绩超预期，股价盘后上涨5%",
            "美联储官员暗示可能进一步加息以控制通胀",
            "特斯拉宣布新一代电池技术突破，续航里程提升40%",
            "中概股集体上涨，阿里巴巴和腾讯领涨",
            "油价因地缘政治紧张局势升至年内新高",
            "比特币突破5万美元关口，市场信心回暖",
            "AI芯片需求激增，英伟达股价创历史新高",
            "银行业监管新规出台，金融股普遍下跌",
            "新能源汽车销量数据亮眼，相关概念股走强",
            "房地产市场出现回暖迹象，地产股反弹"
        ]

        title = mock_titles[index % len(mock_titles)]

        # 提取可能的股票代码
        symbols = []
        symbol_mapping = {
            '苹果': ['AAPL'],
            '特斯拉': ['TSLA'],
            '阿里巴巴': ['BABA'],
            '腾讯': ['00700.HK'],
            '英伟达': ['NVDA']
        }

        for company, codes in symbol_mapping.items():
            if company in title:
                symbols.extend(codes)

        article_id = f"{self.name}_{int(datetime.now().timestamp())}_{index}"

        return NewsArticle(
            article_id=article_id,
            title=title,
            content=f"这是关于{title}的详细报道内容。" + "模拟新闻内容。" * 20,
            source=self.name,
            published_at=datetime.now() - timedelta(minutes=np.random.randint(1, 1440)),
            mentioned_symbols=symbols,
            credibility_score=self.credibility_score,
            word_count=len(title) * 10
        )

class NewsAggregator:
    """新闻聚合器"""

    def __init__(self):
        self.sources = {}
        self.article_cache = {}  # article_id -> NewsArticle
        self.analysis_cache = {}  # article_id -> NewsAnalysis

        # 添加默认新闻源
        self._setup_default_sources()

    def _setup_default_sources(self):
        """设置默认新闻源"""
        default_sources = [
            NewsSource("财经新闻网", "https://finance.example.com", 0.8),
            NewsSource("市场观察", "https://market.example.com", 0.7),
            NewsSource("投资资讯", "https://invest.example.com", 0.6),
            NewsSource("经济日报", "https://economic.example.com", 0.9),
            NewsSource("证券时报", "https://securities.example.com", 0.8)
        ]

        for source in default_sources:
            self.sources[source.name] = source

    async def fetch_all_articles(self, limit_per_source: int = 10) -> List[NewsArticle]:
        """从所有源获取新闻"""
        all_articles = []

        tasks = []
        for source in self.sources.values():
            task = source.fetch_articles(limit_per_source)
            tasks.append(task)

        # 并行获取
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source_name = list(self.sources.keys())[i]
                logger.error(f"从 {source_name} 获取新闻失败: {result}")
                continue

            for article in result:
                # 去重检查
                article_hash = hashlib.md5(f"{article.title}{article.content[:100]}".encode()).hexdigest()
                if article_hash not in [hashlib.md5(f"{a.title}{a.content[:100]}".encode()).hexdigest()
                                       for a in all_articles]:
                    all_articles.append(article)
                    self.article_cache[article.article_id] = article

        logger.info(f"获取到 {len(all_articles)} 篇新闻文章")
        return all_articles

    def get_articles_by_symbol(self, symbol: str) -> List[NewsArticle]:
        """根据股票代码获取相关新闻"""
        return [
            article for article in self.article_cache.values()
            if symbol.upper() in [s.upper() for s in article.mentioned_symbols]
        ]

    def get_articles_by_timerange(self, start_time: datetime, end_time: datetime) -> List[NewsArticle]:
        """根据时间范围获取新闻"""
        return [
            article for article in self.article_cache.values()
            if start_time <= article.published_at <= end_time
        ]

class GPTNewsAnalyzer:
    """GPT新闻分析器主类"""

    def __init__(self, api_key: str = None):
        self.gpt_client = GPTClient(api_key)
        self.news_aggregator = NewsAggregator()

        # 分析历史
        self.analysis_history = deque(maxlen=10000)

        # 统计信息
        self.stats = {
            'total_articles_analyzed': 0,
            'total_gpt_requests': 0,
            'analysis_by_category': defaultdict(int),
            'analysis_by_sentiment': defaultdict(int),
            'average_confidence': 0.0,
            'error_count': 0
        }

        logger.info("GPT新闻分析器初始化完成")

    async def analyze_article(self, article: NewsArticle,
                            analysis_type: str = "comprehensive") -> NewsAnalysis:
        """分析单篇新闻"""
        try:
            # 调用GPT分析
            gpt_result = await self.gpt_client.analyze_news(article, analysis_type)

            # 创建分析结果
            analysis = NewsAnalysis(
                article_id=article.article_id,
                analysis_timestamp=datetime.now(),
                category=NewsCategory(gpt_result.get('category', 'other')),
                sentiment=Sentiment(gpt_result.get('sentiment', 0)),
                sentiment_score=gpt_result.get('sentiment_score', 0.0),
                confidence=gpt_result.get('confidence', 0.5),
                companies=gpt_result.get('companies', []),
                people=gpt_result.get('people', []),
                locations=gpt_result.get('locations', []),
                financial_metrics=gpt_result.get('financial_metrics', {}),
                key_events=gpt_result.get('key_events', []),
                market_impact=ImpactLevel(gpt_result.get('market_impact', 'minimal')),
                affected_sectors=gpt_result.get('affected_sectors', []),
                affected_stocks=gpt_result.get('affected_stocks', []),
                summary=gpt_result.get('summary', ''),
                key_points=gpt_result.get('key_points', []),
                investment_implications=gpt_result.get('investment_implications', ''),
                risk_factors=gpt_result.get('risk_factors', [])
            )

            # 缓存分析结果
            self.news_aggregator.analysis_cache[article.article_id] = analysis

            # 添加到历史
            self.analysis_history.append(analysis)

            # 更新统计
            self._update_stats(analysis)

            logger.debug(f"分析完成: {article.title[:50]}... ({analysis.sentiment.name})")
            return analysis

        except Exception as e:
            logger.error(f"分析文章失败 {article.article_id}: {e}")
            self.stats['error_count'] += 1

            # 返回默认分析结果
            return NewsAnalysis(
                article_id=article.article_id,
                analysis_timestamp=datetime.now(),
                category=NewsCategory.OTHER,
                sentiment=Sentiment.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.0
            )

    async def analyze_batch(self, articles: List[NewsArticle],
                          max_concurrent: int = 5) -> List[NewsAnalysis]:
        """批量分析新闻"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(article):
            async with semaphore:
                return await self.analyze_article(article)

        tasks = [analyze_with_semaphore(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        analyses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量分析中的错误: {result}")
                continue
            analyses.append(result)

        logger.info(f"批量分析完成: {len(analyses)} 篇文章")
        return analyses

    async def fetch_and_analyze_news(self, limit_per_source: int = 10) -> List[NewsAnalysis]:
        """获取并分析新闻"""
        # 获取新闻
        articles = await self.news_aggregator.fetch_all_articles(limit_per_source)

        if not articles:
            logger.warning("未获取到新闻文章")
            return []

        # 批量分析
        analyses = await self.analyze_batch(articles)

        return analyses

    def get_market_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """获取市场情绪摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # 过滤最近的分析
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if analysis.analysis_timestamp >= cutoff_time
        ]

        if not recent_analyses:
            return {'message': '暂无最近的分析数据'}

        # 计算情绪统计
        sentiment_scores = [a.sentiment_score for a in recent_analyses]
        sentiment_distribution = defaultdict(int)
        impact_distribution = defaultdict(int)
        category_distribution = defaultdict(int)

        for analysis in recent_analyses:
            sentiment_distribution[analysis.sentiment.name] += 1
            impact_distribution[analysis.market_impact.value] += 1
            category_distribution[analysis.category.value] += 1

        # 获取高影响文章
        high_impact_analyses = [
            a for a in recent_analyses
            if a.market_impact in [ImpactLevel.HIGH, ImpactLevel.MEDIUM]
        ]

        return {
            'time_period_hours': hours_back,
            'total_articles': len(recent_analyses),
            'average_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
            'sentiment_std': np.std(sentiment_scores) if sentiment_scores else 0,
            'sentiment_distribution': dict(sentiment_distribution),
            'impact_distribution': dict(impact_distribution),
            'category_distribution': dict(category_distribution),
            'high_impact_count': len(high_impact_analyses),
            'top_high_impact_articles': [
                {
                    'article_id': a.article_id,
                    'summary': a.summary,
                    'sentiment_score': a.sentiment_score,
                    'market_impact': a.market_impact.value,
                    'affected_stocks': a.affected_stocks
                }
                for a in sorted(high_impact_analyses,
                               key=lambda x: abs(x.sentiment_score), reverse=True)[:5]
            ]
        }

    def get_symbol_news_summary(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """获取特定股票的新闻摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # 获取相关分析
        relevant_analyses = []
        for analysis in self.analysis_history:
            if (analysis.analysis_timestamp >= cutoff_time and
                (symbol.upper() in [s.upper() for s in analysis.affected_stocks] or
                 symbol.upper() in [c.upper() for c in analysis.companies])):
                relevant_analyses.append(analysis)

        if not relevant_analyses:
            return {'symbol': symbol, 'message': '暂无相关新闻'}

        # 计算统计
        sentiment_scores = [a.sentiment_score for a in relevant_analyses]

        return {
            'symbol': symbol,
            'time_period_hours': hours_back,
            'total_mentions': len(relevant_analyses),
            'average_sentiment': np.mean(sentiment_scores),
            'sentiment_trend': 'positive' if np.mean(sentiment_scores) > 0.1 else
                            'negative' if np.mean(sentiment_scores) < -0.1 else 'neutral',
            'recent_articles': [
                {
                    'summary': a.summary,
                    'sentiment_score': a.sentiment_score,
                    'key_points': a.key_points,
                    'investment_implications': a.investment_implications,
                    'timestamp': a.analysis_timestamp.isoformat()
                }
                for a in sorted(relevant_analyses,
                               key=lambda x: x.analysis_timestamp, reverse=True)[:5]
            ]
        }

    def _update_stats(self, analysis: NewsAnalysis):
        """更新统计信息"""
        self.stats['total_articles_analyzed'] += 1
        self.stats['analysis_by_category'][analysis.category.value] += 1
        self.stats['analysis_by_sentiment'][analysis.sentiment.name] += 1

        # 更新平均置信度
        total = self.stats['total_articles_analyzed']
        current_avg = self.stats['average_confidence']
        self.stats['average_confidence'] = (
            (current_avg * (total - 1) + analysis.confidence) / total
        )

    def get_analyzer_stats(self) -> Dict[str, Any]:
        """获取分析器统计"""
        return {
            'stats': self.stats,
            'gpt_client_stats': {
                'request_count': self.gpt_client.request_count,
                'token_usage': self.gpt_client.token_usage,
                'error_count': self.gpt_client.error_count
            },
            'news_sources': [
                {
                    'name': source.name,
                    'credibility_score': source.credibility_score,
                    'article_count': source.article_count,
                    'last_fetch_time': source.last_fetch_time.isoformat() if source.last_fetch_time else None
                }
                for source in self.news_aggregator.sources.values()
            ],
            'analysis_history_size': len(self.analysis_history),
            'cache_sizes': {
                'articles': len(self.news_aggregator.article_cache),
                'analyses': len(self.news_aggregator.analysis_cache)
            }
        }

# 全局实例
_gpt_news_analyzer_instance = None

def get_gpt_news_analyzer() -> GPTNewsAnalyzer:
    """获取GPT新闻分析器实例"""
    global _gpt_news_analyzer_instance
    if _gpt_news_analyzer_instance is None:
        _gpt_news_analyzer_instance = GPTNewsAnalyzer()
    return _gpt_news_analyzer_instance
