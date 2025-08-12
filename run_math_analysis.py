#!/usr/bin/env python3
"""
Mathematical Federation - Complete Pipeline Runner
Runs all 10 nodes in sequence for any mathematical problem

Usage:
    python run_math_analysis.py "Your mathematical problem here"
    python run_math_analysis.py "Find all integer solutions to x² + y² = 169"
    python run_math_analysis.py --file problem.md
    python run_math_analysis.py --file complex_problem.md --output result.md
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import all mathematical nodes
from math_node1 import MathNode1ProblemClassifier
from math_node2 import SpecializedMathematicalNode as MathNode2AlgebraicSolver
from math_node3 import SpecializedMathematicalNode as MathNode3GeometricAnalyzer
from math_node4 import SpecializedMathematicalNode as MathNode4CombinatorialAnalyzer
from math_node5 import SpecializedMathematicalNode as MathNode5NumberTheoryAnalyzer
from math_node6 import SpecializedMathematicalNode as MathNode6CalculusAnalyzer
from math_node7 import SpecializedMathematicalNode as MathNode7DiscreteAnalyzer
from math_node8 import SpecializedMathematicalNode as MathNode8SymbolicVerifier
from math_node9 import SpecializedMathematicalNode as MathNode9AlternativeAnalyzer
from math_node10 import MathNode10ConsensusSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteMathematicalFederationPipeline:
    """Complete pipeline orchestrator for federated mathematical analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize all nodes and orchestrator"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize all 10 mathematical nodes
        self.nodes = {
            "node1": MathNode1ProblemClassifier(self.api_key),
            "node2": MathNode2AlgebraicSolver(self.api_key),
            "node3": MathNode3GeometricAnalyzer(self.api_key),
            "node4": MathNode4CombinatorialAnalyzer(self.api_key),
            "node5": MathNode5NumberTheoryAnalyzer(self.api_key),
            "node6": MathNode6CalculusAnalyzer(self.api_key),
            "node7": MathNode7DiscreteAnalyzer(self.api_key),
            "node8": MathNode8SymbolicVerifier(self.api_key),
            "node9": MathNode9AlternativeAnalyzer(self.api_key),
            "node10": MathNode10ConsensusSynthesizer(self.api_key)
        }
        
        # Track results
        self.results = {}
        self.output_files = {
            "node1": "math_node1_output.json",
            "node2": "math_node2_output.json", 
            "node3": "math_node3_output.json",
            "node4": "math_node4_output.json",
            "node5": "math_node5_output.json",
            "node6": "math_node6_output.json",
            "node7": "math_node7_output.json",
            "node8": "math_node8_output.json",
            "node9": "math_node9_output.json",
            "node10": "math_node10_output.json",
            "federation": "math_federation_final_answer.json"
        }
        
        # Clean up any old duplicate files
        old_files = ["math_node_1_classification_output.json"]
        for old_file in old_files:
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                except:
                    pass
        
    def load_problem_from_markdown(self, file_path: str) -> str:
        """Load mathematical problem from markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract problem from markdown (could be enhanced with more sophisticated parsing)
            # For now, we'll use the entire content as the problem
            if content.strip():
                logger.info(f"Loaded problem from {file_path} ({len(content)} characters)")
                return content.strip()
            else:
                raise ValueError(f"Empty file: {file_path}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Problem file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading problem file {file_path}: {str(e)}")
    
    def save_results_to_markdown(self, final_result: Dict[str, Any], output_file: str = "math_federation_results.md") -> str:
        """Save federation results to markdown file"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build markdown content
        markdown_content = f"""# Mathematical Federation Analysis Results

**Generated:** {timestamp}  
**System:** 10-Node Mathematical Federation

---

## Problem

```
{final_result['question']}
```

---

## Solution

### Answer Type
**{final_result['answer_type']}**

### Primary Answer
{final_result.get('primary_answer', 'No primary answer provided')}

### System Confidence
**{final_result['system_confidence']:.1%}**

### Human Guidance Needed
**{'Yes' if final_result['human_guidance_needed'] else 'No'}**

---

## Mathematical Reasoning

{final_result['reasoning']}

---

## Federation Analysis Summary

### Performance Assessment
{final_result['federation_performance']}

### Node Execution Summary
"""
        
        # Add node results summary
        for node_id in ["node1", "node2", "node3", "node4", "node5", "node6", "node7", "node8", "node9", "node10"]:
            if node_id in self.results:
                result = self.results[node_id]
                if isinstance(result, dict):
                    confidence = result.get('confidence', 'N/A')
                    if node_id == "node1":
                        domain = result.get('problem_domain', 'Unknown')
                        complexity = result.get('complexity_assessment', {}).get('overall_complexity', 'Unknown')
                        markdown_content += f"\n- **Node {node_id[-1]}**: Problem Classification - Domain: {domain}, Complexity: {complexity}, Confidence: {confidence}"
                    elif node_id == "node10":
                        answer_type = result.get('responsible_answer', {}).get('answer_type', 'Unknown')
                        markdown_content += f"\n- **Node 10**: Consensus Synthesis - Answer Type: {answer_type}"
                    else:
                        approach = result.get('mathematical_approach', 'Unknown approach')
                        markdown_content += f"\n- **Node {node_id[-1]}**: {approach} - Confidence: {confidence}"
        
        markdown_content += f"""

---

## Technical Details

### Generated Files
"""
        
        # List generated files
        for node_id, filename in self.output_files.items():
            if Path(filename).exists():
                file_size = Path(filename).stat().st_size
                markdown_content += f"- `{filename}` ({file_size:,} bytes)\n"
        
        markdown_content += f"""
### Execution Environment
- **OpenAI Model**: GPT-5-mini
- **Federation Architecture**: 10 Specialized Mathematical Nodes
- **Processing Mode**: Sequential Analysis with Cross-Node Coordination

---

*Generated by Mathematical Federation AI System*
"""
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Results saved to markdown file: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to save markdown results: {str(e)}")
            raise
    
    def save_node_result(self, node_id: str, result: Any) -> None:
        """Save node result to JSON file"""
        output_file = self.output_files[node_id]
        
        # Convert Pydantic model to dict if needed
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        else:
            result_dict = result
            
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved {node_id} results to {output_file}")
        self.results[node_id] = result_dict
    
    async def run_node1(self, problem: str) -> Dict[str, Any]:
        """Run Node 1: Problem Classification"""
        logger.info("🔍 Starting Node 1: Problem Classification")
        
        result = await self.nodes["node1"].classify_problem(problem)
        self.save_node_result("node1", result)
        
        logger.info(f"Node 1 Complete - Domain: {result.problem_domain}, "
                   f"Complexity: {result.complexity_assessment.overall_complexity}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node1"]
    
    async def run_node2(self, problem: str) -> Dict[str, Any]:
        """Run Node 2: Algebraic Analysis"""
        logger.info("📊 Starting Node 2: Algebraic Analysis")
        
        # Node 2 needs Node 1's output
        node1_data = self.results["node1"]
        result = await self.nodes["node2"].analyze_mathematically(problem)
        self.save_node_result("node2", result)
        
        logger.info(f"Node 2 Complete - Steps: {len(result.reasoning_steps)}, "
                   f"Insights: {len(result.key_insights)}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node2"]
    
    async def run_node3(self, problem: str) -> Dict[str, Any]:
        """Run Node 3: Geometric Analysis"""
        logger.info("📐 Starting Node 3: Geometric Analysis")
        
        result = await self.nodes["node3"].analyze_mathematically(problem)
        self.save_node_result("node3", result)
        
        logger.info(f"Node 3 Complete - Approach: {result.mathematical_approach}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node3"]
    
    async def run_node4(self, problem: str) -> Dict[str, Any]:
        """Run Node 4: Combinatorial Analysis"""
        logger.info("🔢 Starting Node 4: Combinatorial Analysis")
        
        result = await self.nodes["node4"].analyze_mathematically(problem)
        self.save_node_result("node4", result)
        
        logger.info(f"Node 4 Complete - Approach: {result.mathematical_approach}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node4"]
    
    async def run_node5(self, problem: str) -> Dict[str, Any]:
        """Run Node 5: Number Theory Analysis"""
        logger.info("🔢 Starting Node 5: Number Theory Analysis")
        
        result = await self.nodes["node5"].analyze_mathematically(problem)
        self.save_node_result("node5", result)
        
        logger.info(f"Node 5 Complete - Approach: {result.mathematical_approach}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node5"]
    
    async def run_node6(self, problem: str) -> Dict[str, Any]:
        """Run Node 6: Calculus & Analysis"""
        logger.info("∫ Starting Node 6: Calculus & Analysis")
        
        result = await self.nodes["node6"].analyze_mathematically(problem)
        self.save_node_result("node6", result)
        
        logger.info(f"Node 6 Complete - Approach: {result.mathematical_approach}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node6"]
    
    async def run_node7(self, problem: str) -> Dict[str, Any]:
        """Run Node 7: Discrete Mathematics"""
        logger.info("🔗 Starting Node 7: Discrete Mathematics")
        
        result = await self.nodes["node7"].analyze_mathematically(problem)
        self.save_node_result("node7", result)
        
        logger.info(f"Node 7 Complete - Approach: {result.mathematical_approach}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node7"]
    
    async def run_node8(self, problem: str) -> Dict[str, Any]:
        """Run Node 8: Symbolic Verification"""
        logger.info("✓ Starting Node 8: Symbolic Verification")
        
        result = await self.nodes["node8"].analyze_mathematically(problem)
        self.save_node_result("node8", result)
        
        logger.info(f"Node 8 Complete - Approach: {result.mathematical_approach}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node8"]
    
    async def run_node9(self, problem: str) -> Dict[str, Any]:
        """Run Node 9: Alternative Methods"""
        logger.info("🧩 Starting Node 9: Alternative Methods")
        
        result = await self.nodes["node9"].analyze_mathematically(problem)
        self.save_node_result("node9", result)
        
        logger.info(f"Node 9 Complete - Approach: {result.mathematical_approach}, "
                   f"Confidence: {result.confidence:.2f}")
        return self.results["node9"]
    
    async def run_node10(self, problem: str) -> Dict[str, Any]:
        """Run Node 10: Consensus Synthesis"""
        logger.info("🤝 Starting Node 10: Consensus Synthesis")
        # Node 10 doesn't need the problem - it synthesizes existing results
        
        result = await self.nodes["node10"].synthesize_final_mathematical_answer(problem)
        self.save_node_result("node10", result)
        
        # Save the final federation answer
        ra = result.responsible_answer
        final_answer = {
            "question": problem,
            "answer_type": ra.answer_type,
            "primary_answer": ra.primary_answer,
            "confidence_range": getattr(ra, 'confidence_range', None),
            "system_confidence": ra.system_confidence,
            "reasoning": ra.mathematical_reasoning_summary,
            "human_guidance_needed": ra.human_guidance_needed,
            "federation_performance": self._assess_federation_performance()
        }
        
        with open(self.output_files["federation"], 'w') as f:
            json.dump(final_answer, f, indent=2)
        
        logger.info(f"Node 10 Complete - Answer Type: {ra.answer_type}, "
                   f"System Confidence: {ra.system_confidence:.3f}, "
                   f"Human Needed: {ra.human_guidance_needed}")
        
        return final_answer
    
    async def run_complete_analysis(self, problem: str, output_markdown: str = None) -> Dict[str, Any]:
        """Run the complete federated mathematical analysis pipeline"""
        
        logger.info(f"🚀 Starting Complete Mathematical Federation Analysis")
        logger.info(f"📝 Problem: {problem[:100]}..." if len(problem) > 100 else f"📝 Problem: {problem}")
        logger.info("=" * 60)
        
        try:
            # Run all nodes in sequence
            await self.run_node1(problem)
            await self.run_node2(problem)
            await self.run_node3(problem)
            await self.run_node4(problem)
            await self.run_node5(problem)
            await self.run_node6(problem)
            await self.run_node7(problem)
            await self.run_node8(problem)
            await self.run_node9(problem)
            final_result = await self.run_node10(problem)
            
            logger.info("=" * 60)
            logger.info("🎉 FEDERATION ANALYSIS COMPLETE!")
            logger.info("=" * 60)
            
            # Print final summary
            self.print_final_summary(final_result)
            
            # Save markdown results if requested
            if output_markdown:
                markdown_file = self.save_results_to_markdown(final_result, output_markdown)
                logger.info(f"📄 Markdown results saved to: {markdown_file}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def print_final_summary(self, final_result: Dict[str, Any]) -> None:
        """Print a formatted summary of results"""
        
        print("\n" + "=" * 60)
        print("📋 FEDERATED MATHEMATICAL ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"🎯 Question: {final_result['question']}")
        print(f"📊 Answer Type: {final_result['answer_type']}")
        
        if final_result['primary_answer']:
            print(f"💡 Primary Answer: {final_result['primary_answer']}")
        
        if final_result['confidence_range']:
            print(f"📈 Confidence Range: {final_result['confidence_range']}")
        
        print(f"🎗️ System Confidence: {final_result['system_confidence']:.1%}")
        print(f"👤 Human Guidance Needed: {'Yes' if final_result['human_guidance_needed'] else 'No'}")
        
        print(f"\n🔍 Reasoning:")
        print(f"   {final_result['reasoning']}")
        
        print(f"\n⚡ Federation Performance:")
        print(f"   {final_result['federation_performance']}")
        
        print("\n📁 Generated Files:")
        for node_id, filename in self.output_files.items():
            if Path(filename).exists():
                print(f"   ✅ {filename}")
        
        print("=" * 60)
    
    def _assess_federation_performance(self) -> str:
        """Assess how well the federation performed"""
        if len(self.results) < 9:
            return "Incomplete federation execution"
        
        confidences = []
        for node_id in ["node1", "node2", "node3", "node4", "node5", "node6", "node7", "node8", "node9"]:
            if node_id in self.results:
                result = self.results[node_id]
                if isinstance(result, dict):
                    conf = result.get('confidence', 0.0)
                    if isinstance(conf, (int, float)):
                        confidences.append(float(conf))
        
        if not confidences:
            return "No confidence data available"
        
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        if confidence_variance > 0.1:
            return "High disagreement detected - effective contradiction identification"
        elif avg_confidence > 0.8:
            return "High consensus - reliable mathematical analysis"
        else:
            return "Moderate consensus - some uncertainty present"

def main():
    """Main entry point for command-line usage"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="10-Node Mathematical Federation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_math_analysis.py "Find all integer solutions to x² + y² = 169"
  python run_math_analysis.py --file complex_problem.md
  python run_math_analysis.py --file problem.md --output results.md
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("problem", nargs="?", help="Mathematical problem as text")
    group.add_argument("--file", "-f", help="Load problem from markdown file")
    
    parser.add_argument("--output", "-o", help="Save results to markdown file")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    try:
        # Create pipeline
        pipeline = CompleteMathematicalFederationPipeline()
        
        # Determine problem source
        if args.file:
            problem = pipeline.load_problem_from_markdown(args.file)
            print(f"📁 Loaded problem from: {args.file}")
        else:
            problem = args.problem
        
        # Run analysis
        result = asyncio.run(pipeline.run_complete_analysis(problem, args.output))
        
        # Output completion message
        if args.output:
            print(f"\n✅ Mathematical analysis complete!")
            print(f"📄 Results saved to: {args.output}")
            print(f"📁 JSON files: math_federation_final_answer.json + individual node outputs")
        else:
            print(f"\n✅ Mathematical analysis complete!")
            print(f"📁 Check math_federation_final_answer.json for results.")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()