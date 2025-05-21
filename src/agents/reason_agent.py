from typing import Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..ports.agent_port import ReasonAgent

class Mistral7BReasonAgent(ReasonAgent):
    """Implementation of the reasoning agent using Mistral 7B."""
    
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    async def analyze(
        self,
        query: str,
        retrieved_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze information using Chain of Thought reasoning."""
        
        # Prepare the analysis prompt
        prompt = self._generate_analysis_prompt(query, retrieved_info)
        
        # Generate analysis using Mistral 7B
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1
            )
        
        analysis_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse and structure the analysis
        return self._parse_analysis(analysis_text)
    
    def _generate_analysis_prompt(
        self,
        query: str,
        retrieved_info: Dict[str, Any]
    ) -> str:
        """Generate a prompt for Chain of Thought analysis."""
        
        # Extract key information
        metrics = retrieved_info.get("processed_info", {}).get("key_metrics", [])
        trends = retrieved_info.get("processed_info", {}).get("trends", [])
        
        return f"""Analyze the financial implications of the following query using Chain of Thought reasoning:

Query: {query}

Available Information:
Key Metrics:
{self._format_list(metrics)}

Trends:
{self._format_list(trends)}

Let's approach this step by step:

1) First, let's understand the key factors:
[Your analysis of key factors]

2) Now, let's examine the relationships between these factors:
[Your analysis of relationships]

3) Let's consider the implications:
[Your analysis of implications]

4) Finally, let's draw conclusions:
[Your conclusions]

Based on this analysis, provide:
1. Key findings
2. Short-term implications
3. Medium-term outlook
4. Recommended actions
5. Confidence level

Response:"""
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as a bulleted string."""
        return "\n".join(f"- {item}" for item in items)
    
    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the analysis text into structured data."""
        sections = {
            "key_findings": [],
            "short_term_implications": [],
            "medium_term_outlook": [],
            "recommended_actions": [],
            "confidence_level": 0.0,
            "reasoning_chain": []
        }
        
        current_section = None
        
        for line in analysis_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if "Key findings" in line:
                current_section = "key_findings"
            elif "Short-term implications" in line:
                current_section = "short_term_implications"
            elif "Medium-term outlook" in line:
                current_section = "medium_term_outlook"
            elif "Recommended actions" in line:
                current_section = "recommended_actions"
            elif "Confidence level" in line:
                # Extract confidence level (assumed to be a percentage or decimal)
                try:
                    confidence = float(line.split(":")[-1].strip().rstrip("%")) / 100
                    sections["confidence_level"] = confidence
                except ValueError:
                    sections["confidence_level"] = 0.8  # Default confidence
            elif line.startswith(("1)", "2)", "3)", "4)")):
                # Capture reasoning chain steps
                sections["reasoning_chain"].append(line)
            elif current_section and current_section != "confidence_level":
                # Add content to current section
                if line.startswith("- "):
                    sections[current_section].append(line[2:])
                else:
                    sections[current_section].append(line)
        
        return sections
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through the reason agent."""
        query = input_data.get("query", "")
        retrieved_info = input_data.get("retrieved_info", {})
        return await self.analyze(query, retrieved_info)
