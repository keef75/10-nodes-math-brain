"""
Mathematical Node 1: Problem Classification & Domain Analysis
Analyzes mathematical problems and provides federation guidance
"""

import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolutionStrategy(BaseModel):
    """Individual solution strategy assessment"""
    strategy_name: str = Field(description="Name of the solution strategy")
    mathematical_domain: str = Field(description="Primary mathematical domain")
    approach_description: str = Field(description="Detailed description of the approach")
    feasibility_score: float = Field(ge=0.0, le=1.0, description="How feasible is this strategy?")
    estimated_difficulty: str = Field(description="EASY, MEDIUM, HARD, VERY_HARD")
    required_techniques: List[str] = Field(description="Mathematical techniques needed")
    expected_steps: int = Field(ge=1, le=20, description="Expected number of solution steps")
    
    model_config = {
        "json_schema_extra": {
            "required": ["strategy_name", "mathematical_domain", "approach_description", 
                        "feasibility_score", "estimated_difficulty", "required_techniques", "expected_steps"]
        }
    }

class FederationGuidance(BaseModel):
    """Specific guidance for each node in the federation"""
    node_id: str = Field(description="Target node identifier")
    node_specialization: str = Field(description="Node's mathematical specialization")
    priority_level: str = Field(description="HIGH, MEDIUM, LOW priority for this problem")
    specific_guidance: str = Field(description="Detailed guidance for this node")
    key_focus_areas: List[str] = Field(description="Key areas this node should focus on")
    expected_contribution: str = Field(description="What we expect this node to contribute")
    
    model_config = {
        "json_schema_extra": {
            "required": ["node_id", "node_specialization", "priority_level", 
                        "specific_guidance", "key_focus_areas", "expected_contribution"]
        }
    }

class ProblemComplexityAssessment(BaseModel):
    """Detailed complexity assessment of the mathematical problem"""
    overall_complexity: str = Field(description="ELEMENTARY, INTERMEDIATE, ADVANCED, RESEARCH_LEVEL")
    conceptual_difficulty: str = Field(description="How conceptually challenging is this?")
    computational_difficulty: str = Field(description="How computationally intensive is this?")
    proof_complexity: str = Field(description="If proof required, how complex?")
    prerequisite_knowledge: List[str] = Field(description="Required mathematical background")
    estimated_solution_time: str = Field(description="Expected time to solve")
    
    model_config = {
        "json_schema_extra": {
            "required": ["overall_complexity", "conceptual_difficulty", "computational_difficulty",
                        "proof_complexity", "prerequisite_knowledge", "estimated_solution_time"]
        }
    }

class ProblemClassificationOutput(BaseModel):
    """Complete Pydantic model for Node 1 structured output"""
    analysis_type: str = Field(default="mathematical_problem_classification")
    
    # Core Problem Analysis
    problem_domain: str = Field(description="Primary mathematical domain")
    secondary_domains: List[str] = Field(description="Secondary mathematical domains involved")
    problem_type: str = Field(description="Type of mathematical problem (proof, computation, optimization, etc.)")
    
    # Complexity Assessment  
    complexity_assessment: ProblemComplexityAssessment = Field(description="Detailed complexity analysis")
    
    # Solution Strategy Analysis
    solution_strategies: List[SolutionStrategy] = Field(description="Ranked solution strategies")
    recommended_primary_strategy: str = Field(description="Most promising solution approach")
    
    # Mathematical Objects & Structure
    key_mathematical_objects: List[str] = Field(description="Key mathematical objects (equations, functions, etc.)")
    mathematical_structure: str = Field(description="Overall mathematical structure of the problem")
    problem_constraints: List[str] = Field(description="Constraints and conditions in the problem")
    
    # Federation Coordination
    federation_guidance: List[FederationGuidance] = Field(description="Specific guidance for each federation node")
    node_priority_ranking: Dict[str, str] = Field(description="Priority ranking for each specialized node")
    expected_collaboration_patterns: List[str] = Field(description="How nodes should collaborate")
    
    # Meta-Analysis
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification accuracy")
    reasoning_path: str = Field(description="Detailed explanation of classification process")
    potential_pitfalls: List[str] = Field(description="Common errors or pitfalls to avoid")
    success_criteria: List[str] = Field(description="How to judge if the solution is correct")
    
    model_config = {
        "json_schema_extra": {
            "required": ["problem_domain", "secondary_domains", "problem_type", "complexity_assessment",
                        "solution_strategies", "recommended_primary_strategy", "key_mathematical_objects",
                        "mathematical_structure", "problem_constraints", "federation_guidance", 
                        "node_priority_ranking", "expected_collaboration_patterns", "confidence", 
                        "reasoning_path", "potential_pitfalls", "success_criteria"]
        }
    }
    
    @field_validator('solution_strategies')
    @classmethod
    def validate_strategy_ranking(cls, v: List[SolutionStrategy]) -> List[SolutionStrategy]:
        """Ensure solution strategies are properly ranked by feasibility"""
        if not v:
            raise ValueError("At least one solution strategy required")
        
        # Verify strategies are ranked by feasibility (highest first)
        for i in range(len(v) - 1):
            if v[i].feasibility_score < v[i + 1].feasibility_score:
                logger.warning(f"Strategy ranking may be suboptimal: {v[i].strategy_name} has lower feasibility than {v[i+1].strategy_name}")
        
        return v
    
    @field_validator('federation_guidance')
    @classmethod
    def validate_complete_federation_guidance(cls, v: List[FederationGuidance]) -> List[FederationGuidance]:
        """Ensure guidance is provided for all federation nodes"""
        expected_nodes = [f"math_node_{i}" for i in range(2, 11)]  # Nodes 2-10
        provided_nodes = [guidance.node_id for guidance in v]
        
        missing_nodes = set(expected_nodes) - set(provided_nodes)
        if missing_nodes:
            logger.warning(f"Missing federation guidance for nodes: {missing_nodes}")
        
        return v

class MathNode1ProblemClassifier:
    """Node 1: Mathematical Problem Classification and Federation Coordination"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.node_id = "math_node_1_classification"
    
    def _transform_gpt5_response(self, gpt5_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform GPT-5 response structure to match Pydantic schema"""
        
        # Handle simple response format
        if len(gpt5_data) <= 5:  # Simple response
            domain = gpt5_data.get("problem_domain", "algebra") 
            secondary = gpt5_data.get("secondary_domains", [])
            prob_type = gpt5_data.get("problem_type", "proof")
            confidence = gpt5_data.get("confidence", 0.8)
            
            # Build complete response with defaults
            return {
                "problem_domain": domain,
                "secondary_domains": secondary,
                "problem_type": prob_type,
                "complexity_assessment": {
                    "overall_complexity": "ADVANCED",
                    "conceptual_difficulty": "High conceptual challenge",
                    "computational_difficulty": "Moderate computational work",
                    "proof_complexity": "Mathematical proof required",
                    "prerequisite_knowledge": [domain.replace("_", " ").title()],
                    "estimated_solution_time": "15-30 minutes"
                },
                "solution_strategies": [{
                    "strategy_name": f"{domain.replace('_', ' ').title()} Analysis",
                    "mathematical_domain": domain,
                    "approach_description": f"Apply {domain.replace('_', ' ')} techniques",
                    "feasibility_score": 0.8,
                    "estimated_difficulty": "MEDIUM",
                    "required_techniques": ["mathematical reasoning"],
                    "expected_steps": 5
                }],
                "recommended_primary_strategy": f"{domain.replace('_', ' ').title()} Analysis",
                "key_mathematical_objects": ["functions", "equations"],
                "mathematical_structure": f"{domain.replace('_', ' ').title()} problem",
                "problem_constraints": ["Mathematical constraints apply"],
                "federation_guidance": [
                    {
                        "node_id": f"math_node_{i}",
                        "node_specialization": spec,
                        "priority_level": "HIGH" if domain.lower() in spec.lower() else "MEDIUM",
                        "specific_guidance": f"Apply {spec.lower()} techniques",
                        "key_focus_areas": ["mathematical analysis"],
                        "expected_contribution": f"{spec} verification"
                    }
                    for i, spec in enumerate([
                        "Algebraic Analysis", "Geometric Analysis", "Combinatorial Analysis",
                        "Number Theory Analysis", "Calculus Analysis", "Discrete Mathematics",
                        "Symbolic Verification", "Alternative Methods"
                    ], 2)
                ],
                "node_priority_ranking": {f"math_node_{i}": "MEDIUM" for i in range(2, 10)},
                "expected_collaboration_patterns": ["Sequential analysis", "Cross-domain validation"],
                "confidence": confidence,
                "reasoning_path": f"Classified as {domain.replace('_', ' ')} problem",
                "potential_pitfalls": ["Complex mathematical relationships"],
                "success_criteria": ["Correct mathematical solution"]
            }
        
        # Direct mapping if already in correct format
        if "problem_domain" in gpt5_data and len(gpt5_data) > 10:
            return gpt5_data
        
        # Extract information with flexible field names
        def get_nested_value(data: Dict, *possible_keys) -> Any:
            """Get value from nested dict using multiple possible key paths"""
            for key in possible_keys:
                if isinstance(key, str) and key in data:
                    return data[key]
                elif isinstance(key, tuple):
                    current = data
                    try:
                        for k in key:
                            current = current[k]
                        return current
                    except (KeyError, TypeError):
                        continue
            return None
        
        # Extract domain information
        problem_domain = get_nested_value(gpt5_data, 
            "problem_domain", 
            ("domain_identification", "primary_domain"),
            ("problem_analysis", "domain"),
            "domain") or "number_theory"
            
        secondary_domains = get_nested_value(gpt5_data,
            "secondary_domains",
            ("domain_identification", "secondary_domains"),
            ("problem_analysis", "secondary_domains")) or []
            
        problem_type = get_nested_value(gpt5_data,
            "problem_type",
            ("problem_type_classification", "specific_type"),
            ("problem_analysis", "type")) or "computational"
        
        # Extract complexity assessment
        complexity_data = get_nested_value(gpt5_data, "complexity_assessment", "complexity") or {}
        
        complexity_assessment = {
            "overall_complexity": get_nested_value(complexity_data, 
                "overall_complexity", "overall_difficulty", "difficulty") or "ADVANCED",
            "conceptual_difficulty": get_nested_value(complexity_data,
                "conceptual_difficulty", "conceptual") or "High conceptual challenge",
            "computational_difficulty": get_nested_value(complexity_data,
                "computational_difficulty", "computational") or "Moderate computational work",
            "proof_complexity": get_nested_value(complexity_data,
                "proof_complexity", "proof") or "Mathematical proof required",
            "prerequisite_knowledge": get_nested_value(complexity_data,
                "prerequisite_knowledge", "prerequisites") or ["Number theory", "Modular arithmetic"],
            "estimated_solution_time": get_nested_value(complexity_data,
                "estimated_solution_time", "solution_time") or "15-30 minutes"
        }
        
        # Extract solution strategies
        strategies_data = get_nested_value(gpt5_data, "solution_strategies", "strategies") or []
        solution_strategies = []
        
        for i, strategy in enumerate(strategies_data[:3]):  # Limit to 3 strategies
            if isinstance(strategy, dict):
                solution_strategies.append({
                    "strategy_name": strategy.get("strategy_name", f"Strategy {i+1}"),
                    "mathematical_domain": strategy.get("mathematical_domain", problem_domain),
                    "approach_description": strategy.get("approach_description", "Mathematical approach"),
                    "feasibility_score": float(strategy.get("feasibility_score", 0.8)),
                    "estimated_difficulty": strategy.get("estimated_difficulty", "MEDIUM"),
                    "required_techniques": strategy.get("required_techniques", ["analysis"]),
                    "expected_steps": int(strategy.get("expected_steps", 5))
                })
        
        # Ensure at least one strategy
        if not solution_strategies:
            solution_strategies = [{
                "strategy_name": "Direct Analysis",
                "mathematical_domain": problem_domain,
                "approach_description": "Direct mathematical analysis",
                "feasibility_score": 0.8,
                "estimated_difficulty": "MEDIUM",
                "required_techniques": ["mathematical reasoning"],
                "expected_steps": 5
            }]
        
        # Extract mathematical structure
        structure_data = get_nested_value(gpt5_data, "mathematical_structure", "structure") or {}
        
        key_objects = get_nested_value(structure_data, "key_mathematical_objects", "key_objects", "objects") or ["integers", "equations"]
        math_structure = get_nested_value(structure_data, "mathematical_structure", "overall_structure", "structure") or "Number theory problem"
        constraints = get_nested_value(structure_data, "problem_constraints", "constraints") or ["Positive integers"]
        
        # Extract federation guidance
        guidance_data = get_nested_value(gpt5_data, "federation_guidance", "guidance") or []
        federation_guidance = []
        
        for guidance in guidance_data:
            if isinstance(guidance, dict):
                federation_guidance.append({
                    "node_id": guidance.get("node_id", "math_node_2"),
                    "node_specialization": guidance.get("node_specialization", "Mathematical Analysis"),
                    "priority_level": guidance.get("priority_level", "MEDIUM"),
                    "specific_guidance": guidance.get("specific_guidance", "Apply domain techniques"),
                    "key_focus_areas": guidance.get("key_focus_areas", ["analysis"]),
                    "expected_contribution": guidance.get("expected_contribution", "Domain verification")
                })
        
        # Ensure guidance for all nodes 2-9
        existing_nodes = {g["node_id"] for g in federation_guidance}
        node_specs = [
            ("math_node_2", "Algebraic Analysis"),
            ("math_node_3", "Geometric Analysis"), 
            ("math_node_4", "Combinatorial Analysis"),
            ("math_node_5", "Number Theory Analysis"),
            ("math_node_6", "Calculus Analysis"),
            ("math_node_7", "Discrete Mathematics"),
            ("math_node_8", "Symbolic Verification"),
            ("math_node_9", "Alternative Methods")
        ]
        
        for node_id, spec in node_specs:
            if node_id not in existing_nodes:
                federation_guidance.append({
                    "node_id": node_id,
                    "node_specialization": spec,
                    "priority_level": "HIGH" if "number" in spec.lower() else "MEDIUM",
                    "specific_guidance": f"Apply {spec.lower()} techniques",
                    "key_focus_areas": ["mathematical analysis"],
                    "expected_contribution": f"{spec} verification"
                })
        
        # Build final transformed structure
        transformed = {
            "problem_domain": problem_domain,
            "secondary_domains": secondary_domains,
            "problem_type": problem_type,
            "complexity_assessment": complexity_assessment,
            "solution_strategies": solution_strategies,
            "recommended_primary_strategy": solution_strategies[0]["strategy_name"],
            "key_mathematical_objects": key_objects if isinstance(key_objects, list) else [str(key_objects)],
            "mathematical_structure": math_structure,
            "problem_constraints": constraints if isinstance(constraints, list) else [str(constraints)],
            "federation_guidance": federation_guidance,
            "node_priority_ranking": {f"math_node_{i}": "HIGH" if i == 5 else "MEDIUM" for i in range(2, 10)},
            "expected_collaboration_patterns": get_nested_value(gpt5_data, 
                "expected_collaboration_patterns", "collaboration") or ["Sequential analysis", "Cross-domain validation"],
            "confidence": float(get_nested_value(gpt5_data, "confidence", ("meta_analysis", "confidence")) or 0.85),
            "reasoning_path": get_nested_value(gpt5_data, "reasoning_path", ("meta_analysis", "reasoning_path")) or "Mathematical problem classification",
            "potential_pitfalls": get_nested_value(gpt5_data, "potential_pitfalls", ("meta_analysis", "potential_pitfalls")) or ["Complex mathematical relationships"],
            "success_criteria": get_nested_value(gpt5_data, "success_criteria", ("meta_analysis", "success_criteria")) or ["Correct mathematical solution"]
        }
        
        return transformed
        
    def create_classification_prompt(self, problem: str) -> str:
        """Create comprehensive prompt for mathematical problem classification"""
        return f"""You are a mathematical problem classifier. Analyze this problem and respond with JSON.

PROBLEM: {problem}

Respond with JSON containing:
- problem_domain: string (main domain like "functional_equations", "number_theory", etc.)
- secondary_domains: array of strings
- problem_type: string ("proof", "computational", etc.)  
- confidence: number between 0-1

Keep your response simple and focused.""".strip()
    
    async def classify_problem(self, problem: str) -> ProblemClassificationOutput:
        """Perform comprehensive mathematical problem classification"""
        
        try:
            # Create the classification prompt
            prompt = self.create_classification_prompt(problem)
            
            logger.info(f"Math Node 1: Starting comprehensive problem classification")
            logger.info(f"Math Node 1: Problem preview: {problem[:100]}...")
            
            # Define the complete schema for OpenAI API
            schema = {
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string"},
                    "problem_domain": {"type": "string"},
                    "secondary_domains": {"type": "array", "items": {"type": "string"}},
                    "problem_type": {"type": "string"},
                    "complexity_assessment": {
                        "type": "object",
                        "properties": {
                            "overall_complexity": {"type": "string"},
                            "conceptual_difficulty": {"type": "string"},
                            "computational_difficulty": {"type": "string"},
                            "proof_complexity": {"type": "string"},
                            "prerequisite_knowledge": {"type": "array", "items": {"type": "string"}},
                            "estimated_solution_time": {"type": "string"}
                        },
                        "required": ["overall_complexity", "conceptual_difficulty", "computational_difficulty",
                                   "proof_complexity", "prerequisite_knowledge", "estimated_solution_time"]
                    },
                    "solution_strategies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "strategy_name": {"type": "string"},
                                "mathematical_domain": {"type": "string"},
                                "approach_description": {"type": "string"},
                                "feasibility_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "estimated_difficulty": {"type": "string"},
                                "required_techniques": {"type": "array", "items": {"type": "string"}},
                                "expected_steps": {"type": "integer", "minimum": 1, "maximum": 20}
                            },
                            "required": ["strategy_name", "mathematical_domain", "approach_description", 
                                       "feasibility_score", "estimated_difficulty", "required_techniques", "expected_steps"]
                        }
                    },
                    "recommended_primary_strategy": {"type": "string"},
                    "key_mathematical_objects": {"type": "array", "items": {"type": "string"}},
                    "mathematical_structure": {"type": "string"},
                    "problem_constraints": {"type": "array", "items": {"type": "string"}},
                    "federation_guidance": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string"},
                                "node_specialization": {"type": "string"},
                                "priority_level": {"type": "string"},
                                "specific_guidance": {"type": "string"},
                                "key_focus_areas": {"type": "array", "items": {"type": "string"}},
                                "expected_contribution": {"type": "string"}
                            },
                            "required": ["node_id", "node_specialization", "priority_level", 
                                       "specific_guidance", "key_focus_areas", "expected_contribution"]
                        }
                    },
                    "node_priority_ranking": {"type": "object", "additionalProperties": {"type": "string"}},
                    "expected_collaboration_patterns": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasoning_path": {"type": "string"},
                    "potential_pitfalls": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["problem_domain", "secondary_domains", "problem_type", "complexity_assessment",
                           "solution_strategies", "recommended_primary_strategy", "key_mathematical_objects",
                           "mathematical_structure", "problem_constraints", "federation_guidance", 
                           "node_priority_ranking", "expected_collaboration_patterns", "confidence", 
                           "reasoning_path", "potential_pitfalls", "success_criteria"],
                "additionalProperties": False
            }
            
            # Call OpenAI API with structured output
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=3000   # More tokens for comprehensive analysis
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content
            logger.info(f"Math Node 1: Raw response length: {len(response_text) if response_text else 0}")
            
            if not response_text or not response_text.strip():
                raise ValueError("Empty response from GPT-5")
            
            try:
                classification_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Math Node 1: JSON parse error: {e}")
                logger.error(f"Math Node 1: Response text preview: {response_text[:500]}...")
                raise ValueError(f"Invalid JSON response: {e}")
            
            # Transform GPT-5 response to match our Pydantic schema
            transformed_data = self._transform_gpt5_response(classification_data)
            
            # Ensure required defaults
            transformed_data["analysis_type"] = "mathematical_problem_classification"
            
            # Create Pydantic model instance
            classification = ProblemClassificationOutput(**transformed_data)
            
            logger.info(f"Math Node 1: Classification complete. Domain: {classification.problem_domain}, "
                       f"Complexity: {classification.complexity_assessment.overall_complexity}, "
                       f"Strategies: {len(classification.solution_strategies)}, "
                       f"Confidence: {classification.confidence:.2f}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Math Node 1: Error during classification: {str(e)}")
            raise

class MathematicalFederationOrchestrator:
    """Federation orchestrator following the exact cube analysis architecture"""
    
    def __init__(self):
        self.nodes = {}
        self.execution_graph = []
        self.results = {}
        self.problem_statement = None
        
    def register_node(self, node_id: str, node_instance):
        """Register a mathematical node in the federation"""
        self.nodes[node_id] = node_instance
        logger.info(f"Registered mathematical node: {node_id}")
        
    def add_execution_step(self, node_id: str, inputs: Dict[str, Any]):
        """Add a step to the mathematical execution graph"""
        self.execution_graph.append({
            "node_id": node_id,
            "inputs": inputs
        })
        
    async def execute_node(self, node_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single mathematical node and store results"""
        
        if node_id not in self.nodes:
            raise ValueError(f"Mathematical node {node_id} not registered")
            
        node = self.nodes[node_id]
        
        # Route to appropriate node method based on node type
        if node_id == "math_node_1_classification":
            result = await node.classify_problem(inputs["problem"])
        else:
            # For other mathematical nodes, pass problem + previous results
            previous_results = inputs.get("previous_results", {})
            result = await node.analyze_mathematically(inputs["problem"])
            
        # Store result for subsequent nodes
        self.results[node_id] = result
        
        logger.info(f"Executed mathematical node {node_id} successfully")
        return result.model_dump() if hasattr(result, 'model_dump') else result
        
    async def execute_federation(self, problem: str) -> Dict[str, Any]:
        """Execute the complete mathematical federation"""
        
        self.problem_statement = problem
        execution_results = {}
        
        logger.info(f"🧮 Starting Mathematical Federation Analysis")
        logger.info(f"📝 Problem: {problem}")
        logger.info("=" * 70)
        
        for step in self.execution_graph:
            node_id = step["node_id"]
            inputs = step["inputs"]
            
            # Add problem statement and previous results to inputs
            inputs["problem"] = problem
            inputs["previous_results"] = self.results
            
            try:
                result = await self.execute_node(node_id, inputs)
                execution_results[node_id] = result
                
                # Save individual node result
                output_filename = f"{node_id}_output.json"
                with open(output_filename, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"📁 Saved {node_id} results to {output_filename}")
                
            except Exception as e:
                logger.error(f"❌ Error executing {node_id}: {str(e)}")
                # Continue with other nodes rather than failing completely
                execution_results[node_id] = {"error": str(e)}
                
        return execution_results
        
    def get_node_result(self, node_id: str) -> Optional[Any]:
        """Get the result from a specific mathematical node"""
        return self.results.get(node_id)

# Test function for Math Node 1
async def test_math_node1_classification():
    """Test Mathematical Node 1 with a competition-level problem"""
    
    # Initialize Math Node 1
    node1 = MathNode1ProblemClassifier()
    
    # Initialize orchestrator
    orchestrator = MathematicalFederationOrchestrator()
    orchestrator.register_node("math_node_1_classification", node1)
    
    # Test with a challenging mathematical problem
    test_problem = "Find the number of ordered pairs (a,b) of positive integers such that lcm(a,b) + gcd(a,b) = a + b + 144"
    
    # Add execution step
    orchestrator.add_execution_step(
        "math_node_1_classification", 
        {"problem": test_problem}
    )
    
    try:
        # Execute the classification
        results = await orchestrator.execute_federation(test_problem)
        
        # Print comprehensive results
        print("=" * 80)
        print("MATHEMATICAL NODE 1 CLASSIFICATION RESULTS")
        print("=" * 80)
        
        node1_result = results["math_node_1_classification"]
        
        # Core Analysis
        print(f"Analysis Type: {node1_result['analysis_type']}")
        print(f"Primary Domain: {node1_result['problem_domain']}")
        print(f"Secondary Domains: {', '.join(node1_result['secondary_domains'])}")
        print(f"Problem Type: {node1_result['problem_type']}")
        print(f"Confidence: {node1_result['confidence']:.2f}")
        
        # Complexity Assessment
        complexity = node1_result['complexity_assessment']
        print(f"\nComplexity Assessment:")
        print(f"  Overall: {complexity['overall_complexity']}")
        print(f"  Conceptual: {complexity['conceptual_difficulty']}")
        print(f"  Computational: {complexity['computational_difficulty']}")
        print(f"  Proof Complexity: {complexity['proof_complexity']}")
        print(f"  Prerequisites: {', '.join(complexity['prerequisite_knowledge'])}")
        print(f"  Est. Solution Time: {complexity['estimated_solution_time']}")
        
        # Solution Strategies
        print(f"\nSolution Strategies ({len(node1_result['solution_strategies'])}):")
        for i, strategy in enumerate(node1_result['solution_strategies'], 1):
            print(f"  {i}. {strategy['strategy_name']} (Feasibility: {strategy['feasibility_score']:.2f})")
            print(f"     Domain: {strategy['mathematical_domain']}")
            print(f"     Approach: {strategy['approach_description']}")
            print(f"     Difficulty: {strategy['estimated_difficulty']}")
            print(f"     Steps: {strategy['expected_steps']}")
            print(f"     Techniques: {', '.join(strategy['required_techniques'])}")
        
        print(f"\nRecommended Primary Strategy: {node1_result['recommended_primary_strategy']}")
        
        # Mathematical Structure
        print(f"\nMathematical Structure:")
        print(f"  Structure: {node1_result['mathematical_structure']}")
        print(f"  Key Objects: {', '.join(node1_result['key_mathematical_objects'])}")
        print(f"  Constraints: {', '.join(node1_result['problem_constraints'])}")
        
        # Federation Guidance
        print(f"\nFederation Guidance ({len(node1_result['federation_guidance'])} nodes):")
        for guidance in node1_result['federation_guidance']:
            print(f"  {guidance['node_id']} ({guidance['priority_level']} priority):")
            print(f"    Specialization: {guidance['node_specialization']}")
            print(f"    Guidance: {guidance['specific_guidance']}")
            print(f"    Focus Areas: {', '.join(guidance['key_focus_areas'])}")
            print(f"    Expected Contribution: {guidance['expected_contribution']}")
        
        # Node Priority Ranking
        print(f"\nNode Priority Ranking:")
        for node_id, priority in node1_result['node_priority_ranking'].items():
            print(f"  {node_id}: {priority}")
        
        # Collaboration Patterns
        print(f"\nExpected Collaboration Patterns:")
        for pattern in node1_result['expected_collaboration_patterns']:
            print(f"  - {pattern}")
        
        # Success Criteria & Pitfalls
        print(f"\nSuccess Criteria:")
        for criterion in node1_result['success_criteria']:
            print(f"  - {criterion}")
        
        print(f"\nPotential Pitfalls:")
        for pitfall in node1_result['potential_pitfalls']:
            print(f"  - {pitfall}")
        
        print(f"\nReasoning Path:")
        print(f"  {node1_result['reasoning_path']}")
        
        print("\n" + "=" * 80)
        print("Results saved to math_node_1_classification_output.json")
        return node1_result
        
    except Exception as e:
        print(f"Error during Math Node 1 execution: {str(e)}")
        return None

if __name__ == "__main__":
    import asyncio
    
    # Run the test
    result = asyncio.run(test_math_node1_classification())