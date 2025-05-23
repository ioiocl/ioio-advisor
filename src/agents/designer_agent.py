from typing import Dict, Any, Optional, List, Tuple
import os
import base64
import io
import logging
import aiohttp
import json
from functools import lru_cache
from datetime import datetime, timedelta
from PIL import Image
from dataclasses import dataclass
from enum import Enum

class ChartType(Enum):
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    MIXED = "mixed"
    SCATTER = "scatter"
    HEATMAP = "heatmap"

class VisualizationStyle(Enum):
    MODERN = "modern"
    CLASSIC = "classic"
    MINIMALIST = "minimalist"
    DARK = "dark"
    LIGHT = "light"

@dataclass
class ChartConfig:
    type: ChartType
    elements: List[str]
    colors: List[str]
    indicators: List[str]
    interactive: bool = True
    animations: bool = True
    tooltips: bool = True

logger = logging.getLogger(__name__)
from ..ports.agent_port import DesignerAgent
from ..infrastructure.storage.local_storage import LocalStorageService

class StableDiffusionDesignerAgent(DesignerAgent):
    """Implementation of the designer agent using Stable Diffusion with advanced visualization capabilities."""
    
    def __init__(self):
        self.api_key = os.getenv("STABILITY_API_KEY")
        self.api_host = "https://api.stability.ai"
        self.engine_id = "stable-diffusion-xl-1024-v1-0"
        self.storage = LocalStorageService()
        self.cache_duration = timedelta(hours=1)
        self._initialize_cache()
        
        # Enhanced chart templates with interactive features
        self.chart_templates = {
            "stock": ChartConfig(
                type=ChartType.CANDLESTICK,
                elements=["price", "volume", "moving_averages", "bollinger_bands", "fibonacci_levels"],
                colors=["#2E7D32", "#C62828", "#1565C0", "#FFA000", "#6200EA"],
                indicators=["RSI", "MACD", "ATR", "OBV", "Stochastic"],
                interactive=True,
                animations=True,
                tooltips=True
            ),
            "currency": ChartConfig(
                type=ChartType.LINE,
                elements=["exchange_rate", "bid_ask_spread", "volatility_bands", "support_resistance"],
                colors=["#1565C0", "#FFA000", "#2E7D32", "#C62828"],
                indicators=["volatility", "trend", "momentum", "strength_index"],
                interactive=True,
                animations=True,
                tooltips=True
            ),
            "interest": ChartConfig(
                type=ChartType.BAR,
                elements=["rates", "terms", "payments", "comparison", "historical"],
                colors=["#2E7D32", "#1565C0", "#FFA000", "#6200EA", "#C62828"],
                indicators=["APR", "monthly_payment", "total_interest", "amortization"],
                interactive=True,
                animations=True,
                tooltips=True
            ),
            "inflation": ChartConfig(
                type=ChartType.AREA,
                elements=["cpi", "categories", "purchasing_power", "wage_growth", "sector_impact"],
                colors=["#C62828", "#F57C00", "#1565C0", "#2E7D32", "#6200EA"],
                indicators=["yoy_change", "core_inflation", "real_rate", "velocity"],
                interactive=True,
                animations=True,
                tooltips=True
            ),
            "investment": ChartConfig(
                type=ChartType.MIXED,
                elements=["allocation", "performance", "risk", "correlation", "efficient_frontier"],
                colors=["#1565C0", "#2E7D32", "#FFA000", "#6200EA", "#C62828"],
                indicators=["returns", "volatility", "sharpe_ratio", "alpha", "beta"],
                interactive=True,
                animations=True,
                tooltips=True
            ),
            "default": ChartConfig(
                type=ChartType.LINE,
                elements=["trend", "comparison", "forecast"],
                colors=["#1565C0", "#2E7D32", "#FFA000"],
                indicators=["change", "average", "prediction"],
                interactive=True,
                animations=True,
                tooltips=True
            )
        }
        
        # Initialize the visualization cache
        self._visualization_cache = {}
        self._cache_timestamps = {}
    
    async def generate_visualization(
        self,
        context: Dict[str, Any],
        text: str
    ) -> Dict[str, Any]:
        """Generate both a data-driven visualization and an AI-generated image.
        Returns a dictionary containing:
        - visualization_url: URL to the data visualization
        - image_url: URL to the AI-generated image
        - metadata: Dictionary with metadata about both visualizations
        """
        
        # Get the topic and style
        topic = context.get("intent", {}).get("main_topic", "default")
        if not topic or topic == "general_finance":
            # Try to detect topic from text
            topic = self._detect_topic_from_text(text)
        
        # Extract data points from text
        data_points = self._extract_data_points(text)
        
        # Generate chart data
        chart_data = self._generate_chart_data(topic, data_points)
        
        try:
            # Create visualization using matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
            
            # Set style
            plt.style.use('seaborn')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get chart configuration
            chart_config = self.chart_templates[topic]
            
            if chart_config.type == ChartType.BAR:
                # Create bar chart
                bars = ax.bar(chart_data["labels"], chart_data["data"])
                for i, bar in enumerate(bars):
                    bar.set_color(chart_config.colors[i % len(chart_config.colors)])
                ax.set_xticklabels(chart_data["labels"], rotation=45, ha='right')
                
            elif chart_config.type == ChartType.LINE or chart_config.type == ChartType.AREA:
                # Create line/area chart
                if chart_config.type == ChartType.AREA:
                    ax.fill_between(range(len(chart_data["data"])), chart_data["data"],
                                  color=chart_config.colors[0], alpha=0.3)
                ax.plot(range(len(chart_data["data"])), chart_data["data"],
                        color=chart_config.colors[0], linewidth=2)
                if chart_data["labels"]:
                    ax.set_xticks(range(len(chart_data["labels"])))
                    ax.set_xticklabels(chart_data["labels"], rotation=45, ha='right')
            
            elif chart_config.type == ChartType.MIXED:
                # Create combination chart
                for i, series in enumerate(chart_data["series"]):
                    ax.plot(range(len(series["data"])), series["data"],
                            label=series["name"], color=chart_config.colors[i])
                ax.legend()
            
            # Add trend indicators if available
            if "trend" in chart_data["indicators"]:
                trends = chart_data["indicators"]["trend"]
                for i, trend in enumerate(trends):
                    color = "#2E7D32" if trend == "up" else "#C62828"
                    ax.axvspan(i-0.2, i+0.2, color=color, alpha=0.1)
            
            # Customize appearance
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(f'Análisis Financiero: {topic.title()}', pad=20)
            
            # Save plot to memory
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            plt.close()
            
            # Save the chart and get its URL
            chart_path = await self.storage.save_image(buf.getvalue(), extension='png')
            chart_url = self.storage.get_image_url(chart_path)
            
            # Generate DALL-E image
            import openai
            from os import getenv
            
            openai.api_key = getenv('OPENAI_API_KEY')
            
            # Create a descriptive prompt based on the topic and data
            prompt = self._generate_dalle_prompt(topic, data_points, chart_data)
            
            # Generate image with DALL-E
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            
            # Get the generated image URL
            dalle_image_url = response['data'][0]['url']
            
            # Download and save the DALL-E image
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(dalle_image_url)
                image_data = response.content
            
            # Save the DALL-E image and get its URL
            image_path = await self.storage.save_image(image_data, extension='png')
            image_url = self.storage.get_image_url(image_path)
            
            # Prepare metadata
            metadata = {
                "topic": topic,
                "chart": {
                    "type": chart_config.type.value,
                    "data_points": {
                        "values": len(data_points["values"]),
                        "categories": len(data_points["categories"]),
                        "trends": len(data_points["trends"])
                    }
                },
                "dalle_image": {
                    "prompt": prompt
                },
                "generated_at": datetime.now().isoformat()
            }
            
            return {
                "chart_url": chart_url,
                "image_url": image_url,
                "metadata": metadata
            }
            
        except Exception as e:
            error_msg = f"Error generating visualization: {str(e)}"
            print(error_msg)
            # Re-raise with more context if it's a value error
            if isinstance(e, ValueError):
                raise ValueError(error_msg) from e
            return {
                "image_url": None,
                "image_metadata": {
                    "error": str(e),
                    "topic": topic
                }
            }
    
    def _extract_data_points(self, text: str) -> dict:
        """Extract numerical data and trends from the response text."""
        data_points = {
            "values": [],
            "percentages": [],
            "trends": [],
            "categories": [],
            "time_periods": []
        }
        
        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Extract numbers and percentages
            words = sentence_lower.split()
            for i, word in enumerate(words):
                # Look for percentages
                if "%" in word:
                    try:
                        value = float(word.replace("%", "").strip())
                        data_points["percentages"].append({
                            "value": value,
                            "context": " ".join(words[max(0, i-2):min(len(words), i+3)])
                        })
                    except ValueError:
                        continue
                
                # Look for numerical values
                elif word.replace(".", "").replace(",", "").isdigit():
                    try:
                        value = float(word.replace(",", ""))
                        data_points["values"].append({
                            "value": value,
                            "context": " ".join(words[max(0, i-2):min(len(words), i+3)])
                        })
                    except ValueError:
                        continue
            
            # Extract trends
            trend_words = ["aumentó", "disminuyó", "subió", "bajó", "creció", "cayó"]
            for trend in trend_words:
                if trend in sentence_lower:
                    data_points["trends"].append({
                        "direction": "up" if trend in ["aumentó", "subió", "creció"] else "down",
                        "context": sentence
                    })
            
            # Extract categories
            if ":" in sentence:
                category = sentence.split(":")[0].strip()
                if len(category.split()) <= 3:  # Reasonable category name length
                    data_points["categories"].append(category)
            
            # Extract time periods
            time_indicators = ["año", "mes", "semana", "trimestre", "semestre"]
            if any(indicator in sentence_lower for indicator in time_indicators):
                data_points["time_periods"].append(sentence)
        
        return data_points
    
    def _detect_topic_from_text(self, text: str) -> str:
        """Detect the main financial topic from the text content."""
        text_lower = text.lower()
        
        # Topic detection rules
        if any(word in text_lower for word in ["aapl", "googl", "msft", "acciones", "bolsa"]):
            return "stock"
        elif any(word in text_lower for word in ["dólar", "euro", "cambio", "divisa"]):
            return "currency"
        elif any(word in text_lower for word in ["tasa", "interés", "préstamo", "hipoteca"]):
            return "interest"
        elif any(word in text_lower for word in ["inflación", "precios", "aumento"]):
            return "inflation"
        elif any(word in text_lower for word in ["inversión", "invertir", "riesgo", "rendimiento"]):
            return "investment"
        
        return "default"
    
    def _get_topic_visualization(self, topic: str, key_concepts: list) -> str:
        """Get topic-specific visualization elements."""
        visualizations = {
            "stock": """
            - Add stock price chart with time series data
            - Include company logos and ticker symbols
            - Show price changes with up/down arrows
            - Use candlestick patterns for price movements
            - Add volume bars below the main chart""",
            
            "currency": """
            - Create a world map with major currency zones
            - Show exchange rate flows with animated arrows
            - Include currency symbols and flags
            - Display real-time rate changes
            - Add mini trend charts for key pairs""",
            
            "interest": """
            - Compare loan types with bar charts
            - Show APR ranges and terms
            - Include monthly payment calculations
            - Add loan type icons (house, car, etc.)
            - Display amortization schedule snippet""",
            
            "inflation": """
            - Create price trend charts over time
            - Show category-wise inflation breakdown
            - Include common consumer goods icons
            - Add cost comparison timeline
            - Display purchasing power indicators""",
            
            "investment": """
            - Show portfolio allocation pie chart
            - Include risk-return scatter plot
            - Add asset class performance bars
            - Display investment timeline
            - Show diversification benefits""",
            
            "default": """
            - Create multi-panel financial dashboard
            - Include key performance indicators
            - Add trend analysis charts
            - Show relevant financial metrics
            - Include summary statistics"""
        }
        
        return visualizations.get(topic, visualizations["default"])
    
    def _generate_chart_data(self, topic: str, data_points: dict) -> dict:
        """Generate chart data based on extracted data points."""
        chart_config = self.chart_templates.get(topic, self.chart_templates["default"])
        
        # Initialize chart data structure
        chart_data = {
            "type": chart_config.type.value,
            "data": [],
            "labels": [],
            "series": [],
            "colors": chart_config.colors,
            "indicators": {}
        }
        
        # Process data points based on chart type
        if chart_config.type == ChartType.BAR:
            # Use categories and their corresponding values/percentages
            for cat in data_points["categories"]:
                chart_data["labels"].append(cat)
                # Find corresponding value
                value = next((p["value"] for p in data_points["percentages"] 
                           if cat.lower() in p["context"].lower()), 0)
                chart_data["data"].append(value)
        
        elif chart_config.type == ChartType.LINE or chart_config.type == ChartType.AREA:
            # Use time series data if available
            if data_points["time_periods"]:
                chart_data["labels"] = data_points["time_periods"]
                chart_data["data"] = [p["value"] for p in data_points["values"]]
        
        elif chart_config.type == ChartType.MIXED:
            # Combine different types of data
            chart_data["series"] = [
                {
                    "name": "Valores",
                    "data": [p["value"] for p in data_points["values"]]
                },
                {
                    "name": "Porcentajes",
                    "data": [p["value"] for p in data_points["percentages"]]
                }
            ]
        
        # Add indicators
        for indicator in chart_config.indicators:
            if indicator == "trend":
                chart_data["indicators"]["trend"] = [
                    t["direction"] for t in data_points["trends"]
                ]
        
        return chart_data
    
    def _get_visualization_elements(self, topic: str) -> list:
        """Get the list of visualization elements for a topic."""
        elements_map = {
            "stock": ["chart", "candlestick", "company_logos", "price_indicators", "volume_bars"],
            "currency": ["world_map", "exchange_rates", "currency_symbols", "trend_charts", "flags"],
            "interest": ["loan_comparison", "apr_charts", "loan_types", "payment_calculator", "amortization"],
            "inflation": ["price_trends", "category_breakdown", "consumer_goods", "timeline", "indicators"],
            "investment": ["portfolio_allocation", "risk_return", "asset_classes", "timeline", "diversification"],
            "default": ["dashboard", "charts", "indicators", "metrics", "statistics"]
        }
        return elements_map.get(topic, elements_map["default"])
    
    def _get_style_metadata(self, topic: str) -> dict:
        """Get the style metadata for a topic."""
        styles = {
            "stock": {
                "color_scheme": "dark_theme_with_neon",
                "layout": "data_centric",
                "design_elements": ["glowing_accents", "grid_lines", "sharp_edges"]
            },
            "currency": {
                "color_scheme": "professional_light_gold",
                "layout": "geographic",
                "design_elements": ["flowing_arrows", "gradient_fills", "shadows"]
            },
            "interest": {
                "color_scheme": "clean_white_green",
                "layout": "comparison_focused",
                "design_elements": ["icons", "data_tables", "clean_lines"]
            },
            "inflation": {
                "color_scheme": "warm_alert",
                "layout": "category_grid",
                "design_elements": ["price_tags", "trend_arrows", "product_icons"]
            },
            "investment": {
                "color_scheme": "professional_blue",
                "layout": "portfolio_view",
                "design_elements": ["pie_charts", "scatter_plots", "progress_bars"]
            },
            "default": {
                "color_scheme": "neutral_professional",
                "layout": "dashboard",
                "design_elements": ["charts", "cards", "icons"]
            }
        }
        return styles.get(topic, styles["default"])
    
    def _generate_dalle_prompt(self, topic: str, data_points: Dict[str, Any], chart_data: Dict[str, Any]) -> str:
        """Generate a descriptive prompt for DALL-E based on the financial topic and data.
        
        Args:
            topic: The financial topic (e.g., 'stocks', 'inflation')
            data_points: Dictionary containing extracted numerical values, trends, etc.
            chart_data: Dictionary containing processed chart data
            
        Returns:
            A detailed prompt string for DALL-E image generation
        """
        # Base prompts for different financial topics
        topic_prompts = {
            "stocks": "A modern, professional illustration of a stock market trading floor or digital trading interface",
            "inflation": "A creative visualization of inflation and price changes, with currency symbols and price tags",
            "investment": "A conceptual image of investment growth and wealth management",
            "interest": "A visual representation of interest rates and banking concepts",
            "currency": "An artistic composition of global currencies and exchange rates",
            "default": "A professional financial concept illustration"
        }
        
        # Get base prompt
        base_prompt = topic_prompts.get(topic, topic_prompts["default"])
        
        # Add trend information if available
        trend_text = ""
        if "trend" in chart_data.get("indicators", {}):
            trends = chart_data["indicators"]["trend"]
            if all(t == "up" for t in trends):
                trend_text = ", showing upward growth and positive momentum"
            elif all(t == "down" for t in trends):
                trend_text = ", depicting market decline or economic challenges"
            else:
                trend_text = ", illustrating market volatility and fluctuation"
        
        # Add style elements
        style_elements = [
            "professional lighting",
            "clean design",
            "financial district atmosphere",
            "modern business aesthetic",
            "high-quality digital art style"
        ]
        
        # Combine all elements
        prompt = f"{base_prompt}{trend_text}, {', '.join(style_elements)}"
        
        return prompt
    
    def _get_prompt_summary(self, prompt: str) -> dict:
        """Create a summary of the generation prompt."""
        return {
            "length": len(prompt),
            "key_elements": [
                element.strip()
                for element in prompt.split("\n")
                if element.strip().startswith("-")
            ][:5],  # First 5 key elements
            "style_focus": "data_visualization",
            "language": "spanish"
        }
    
    async def _call_stability_api(self, prompt: str, retries: int = 3) -> bytes:
        """Call the Stability AI API to generate the image with retries and enhanced error handling."""
        if not self.api_key:
            raise ValueError("Missing Stability API key")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 50,
            "image_format": "jpeg"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Non-200 response: {await response.text()}")
                
                response_data = await response.json()
                if 'artifacts' not in response_data or not response_data['artifacts']:
                    raise Exception("No image generated in response")
                
                # Get the base64 image data from the first artifact
                raw_image_data = base64.b64decode(response_data['artifacts'][0]['base64'])
                logger.info(f"Received raw image data size: {len(raw_image_data)} bytes")
                
                # Validate and convert image format
                try:
                    # Open the image using PIL
                    image = Image.open(io.BytesIO(raw_image_data))
                    logger.info(f"Original image format: {image.format}, size: {image.size}, mode: {image.mode}")
                    
                    # Convert to RGB if needed (in case it's RGBA)
                    if image.mode in ('RGBA', 'LA'):
                        image = image.convert('RGB')
                        logger.info("Converted image to RGB mode")
                    
                    # Create a new image with minimum dimensions
                    min_width = max(800, image.width)
                    min_height = max(600, image.height)
                    if min_width > image.width or min_height > image.height:
                        image = image.resize((min_width, min_height), Image.Resampling.LANCZOS)
                        logger.info(f"Resized image to {min_width}x{min_height}")
                    
                    # Save as high-quality JPEG in memory
                    output_buffer = io.BytesIO()
                    image.save(output_buffer, format='JPEG', quality=100, optimize=True)
                    image_data = output_buffer.getvalue()
                    
                    # Validate size
                    image_size = len(image_data)
                    logger.info(f"Final image size: {image_size} bytes")
                    
                    # In test environment, we accept smaller images
                    min_size = 30000 if os.getenv('TESTING') else 50000
                    if image_size < min_size:
                        raise ValueError(f"Generated image is too small ({image_size} bytes). Minimum size: {min_size} bytes")
                    
                    return image_data
                except Exception as e:
                    raise ValueError(f"Failed to process image: {str(e)}")
    
    def _initialize_cache(self):
        """Initialize the visualization cache system."""
        self._visualization_cache = {}
        self._cache_timestamps = {}
    
    def _get_cache_key(self, context: Dict[str, Any], text: str) -> str:
        """Generate a unique cache key for the visualization."""
        cache_data = {
            "context": context,
            "text": text,
            "timestamp": datetime.now().strftime("%Y%m%d")
        }
        return base64.b64encode(json.dumps(cache_data).encode()).decode()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if the cached visualization is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        age = datetime.now() - self._cache_timestamps[cache_key]
        return age < self.cache_duration
    
    @lru_cache(maxsize=100)
    def _get_chart_style(self, topic: str, user_level: str) -> Dict[str, Any]:
        """Get cached chart style configuration."""
        return self._get_style_metadata(topic)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the designer agent with caching and error handling."""
        try:
            context = input_data.get("context", {})
            text = input_data.get("response", "")
            
            # Check cache first
            cache_key = self._get_cache_key(context, text)
            if self._is_cache_valid(cache_key) and cache_key in self._visualization_cache:
                return {"visualization": self._visualization_cache[cache_key]}
            
            # Generate new visualization
            visualization = await self.generate_visualization(context, text)
            
            # Cache the result
            self._visualization_cache[cache_key] = visualization
            self._cache_timestamps[cache_key] = datetime.now()
            
            return {"visualization": visualization}
        except Exception as e:
            logger.error(f"Error in designer agent: {str(e)}")
            return {
                "visualization": None,
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            }
