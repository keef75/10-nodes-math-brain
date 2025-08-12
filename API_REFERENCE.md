# API Reference - 10-Node Mathematical Federation System

## Overview

This document provides comprehensive API documentation for the 10-Node Mathematical Federation System, including class definitions, method signatures, and data models.

---

## Core Classes

### 1. CompleteMathematicalFederationPipeline

**File:** `run_math_analysis.py`

The main orchestrator class that coordinates execution of all mathematical nodes.

#### Constructor
```python
def __init__(self, api_key: Optional[str] = None)
```
- **Parameters:**
  - `api_key`: OpenAI API key (defaults to `OPENAI_API_KEY` environment variable)
- **Raises:**
  - `ValueError`: If no API key is provided or found

#### Methods

##### `load_problem_from_markdown(file_path: str) -> str`
Load mathematical problem from markdown file.
- **Parameters:**
  - `file_path`: Path to markdown file containing the problem
- **Returns:** Problem text as string
- **Raises:** `FileNotFoundError`, `ValueError` for empty files

##### `async run_complete_analysis(problem: str, output_markdown: str = None) -> Dict[str, Any]`
Execute the complete mathematical federation analysis pipeline.
- **Parameters:**
  - `problem`: Mathematical problem as text
  - `output_markdown`: Optional path for markdown output
- **Returns:** Final analysis result dictionary
- **Raises:** Generic `Exception` for pipeline failures

##### `save_results_to_markdown(final_result: Dict[str, Any], output_file: str) -> str`
Save federation results to formatted markdown file.
- **Parameters:**
  - `final_result`: Analysis results from Node 10
  - `output_file`: Output markdown file path
- **Returns:** Path to created markdown file

---

## Node 1: Problem Classification

### MathNode1ProblemClassifier

**File:** `math_node1.py`

Analyzes mathematical problems and provides federation guidance.

#### Data Models

##### `ProblemComplexityAssessment`
```python
class ProblemComplexityAssessment(BaseModel):
    overall_complexity: str  # BASIC, INTERMEDIATE, ADVANCED, EXPERT
    conceptual_difficulty: str
    computational_difficulty: str
    proof_complexity: str
    prerequisite_knowledge: List[str]
    estimated_solution_time: str
```

##### `SolutionStrategy`
```python
class SolutionStrategy(BaseModel):
    strategy_name: str
    mathematical_domain: str
    approach_description: str
    feasibility_score: float  # 0.0 to 1.0
    estimated_difficulty: str  # EASY, MEDIUM, HARD, VERY_HARD
    required_techniques: List[str]
    expected_steps: int  # 1 to 20
```

##### `FederationGuidance`
```python
class FederationGuidance(BaseModel):
    node_id: str
    node_specialization: str
    priority_level: str  # HIGH, MEDIUM, LOW
    specific_guidance: str
    key_focus_areas: List[str]
    expected_contribution: str
```

##### `ProblemClassificationOutput`
```python
class ProblemClassificationOutput(BaseModel):
    analysis_type: str
    problem_domain: str
    secondary_domains: List[str]
    problem_type: str
    complexity_assessment: ProblemComplexityAssessment
    solution_strategies: List[SolutionStrategy]
    federation_guidance: List[FederationGuidance]
    confidence: float  # 0.0 to 1.0
```

#### Methods

##### `async classify_problem(problem: str) -> ProblemClassificationOutput`
Classify mathematical problem and generate federation guidance.
- **Parameters:**
  - `problem`: Mathematical problem text
- **Returns:** Complete problem classification
- **Raises:** Generic `Exception` for API or processing errors

---

## Nodes 2-9: Specialized Mathematical Analysis

### SpecializedMathematicalNode

**Files:** `math_node2.py` through `math_node9.py`

Template-based nodes for domain-specific mathematical analysis.

#### Specialization Configurations

Each node has a unique `SPECIALIZATION_CONFIG`:

##### Node 2 - Algebraic Analysis
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_2",
    "node_name": "Algebraic Analysis",
    "mathematical_domain": "algebra",
    "primary_techniques": ["equation solving", "substitution", "factoring", "algebraic manipulation"],
    "output_filename": "math_node2_output.json"
}
```

##### Node 3 - Geometric Analysis
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_3", 
    "node_name": "Geometric Analysis",
    "mathematical_domain": "geometry",
    "primary_techniques": ["coordinate geometry", "constructions", "similarity", "trigonometry"],
    "output_filename": "math_node3_output.json"
}
```

##### Node 4 - Combinatorial Analysis
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_4",
    "node_name": "Combinatorial Analysis", 
    "mathematical_domain": "combinatorics",
    "primary_techniques": ["counting principles", "permutations", "combinations", "inclusion-exclusion"],
    "output_filename": "math_node4_output.json"
}
```

##### Node 5 - Number Theory Analysis
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_5",
    "node_name": "Number Theory Analysis",
    "mathematical_domain": "number_theory", 
    "primary_techniques": ["modular arithmetic", "divisibility", "prime factorization", "Diophantine equations"],
    "output_filename": "math_node5_output.json"
}
```

##### Node 6 - Calculus Analysis
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_6",
    "node_name": "Calculus Analysis",
    "mathematical_domain": "calculus",
    "primary_techniques": ["differentiation", "integration", "limits", "series"],
    "output_filename": "math_node6_output.json"
}
```

##### Node 7 - Discrete Mathematics
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_7", 
    "node_name": "Discrete Mathematics",
    "mathematical_domain": "discrete_mathematics",
    "primary_techniques": ["logic", "graph theory", "algorithms", "proof techniques"],
    "output_filename": "math_node7_output.json"
}
```

##### Node 8 - Symbolic Verification
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_8",
    "node_name": "Symbolic Verification",
    "mathematical_domain": "symbolic_verification",
    "primary_techniques": ["consistency checking", "algebraic verification", "symbolic computation"],
    "output_filename": "math_node8_output.json"
}
```

##### Node 9 - Alternative Methods
```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_9",
    "node_name": "Alternative Methods", 
    "mathematical_domain": "alternative_methods",
    "primary_techniques": ["creative approaches", "unconventional methods", "heuristics"],
    "output_filename": "math_node9_output.json"
}
```

#### Data Models

##### `MathematicalStep`
```python
class MathematicalStep(BaseModel):
    step_number: int
    operation: str
    justification: str
    result: str
    confidence: float  # 0.0 to 1.0
```

##### `MathematicalInsight`
```python
class MathematicalInsight(BaseModel):
    insight_type: str
    description: str
    mathematical_significance: str
    connections_to_other_areas: List[str]
```

##### `SpecializedMathAnalysisOutput`
```python
class SpecializedMathAnalysisOutput(BaseModel):
    analysis_type: str
    specialization: str
    mathematical_approach: str
    reasoning_steps: List[MathematicalStep]
    key_insights: List[MathematicalInsight]
    final_result: str
    alternative_approaches: List[str]
    confidence: float  # 0.0 to 1.0
    domain_specific_notes: Dict[str, Any]
```

#### Methods

##### `async analyze_mathematically(problem: str) -> SpecializedMathAnalysisOutput`
Perform domain-specific mathematical analysis.
- **Parameters:**
  - `problem`: Mathematical problem text
- **Returns:** Specialized analysis output
- **Uses:** Node 1's classification output for context

---

## Node 10: Consensus Synthesis

### MathNode10ConsensusSynthesizer

**File:** `math_node10.py`

Synthesizes findings from all mathematical approaches and provides responsible final answer.

#### Data Models

##### `MathematicalEvidenceSummary`
```python
class MathematicalEvidenceSummary(BaseModel):
    node_id: str
    mathematical_approach: str
    key_finding: str
    final_result: str
    confidence: float  # 0.0 to 1.0
    supporting_steps: int
    mathematical_rigor: str
    domain_alignment: str
```

##### `ContradictionAnalysis`
```python
class ContradictionAnalysis(BaseModel):
    contradiction_detected: bool
    contradicting_nodes: List[str]
    contradiction_description: str
    resolution_approach: str
    confidence_impact: str
```

##### `ResponsibleMathematicalAnswer`
```python
class ResponsibleMathematicalAnswer(BaseModel):
    answer_type: str  # CONFIDENT_ANSWER, UNCERTAIN_RANGE, DEFER_TO_HUMAN
    primary_answer: Optional[str]
    confidence_range: Optional[str]
    system_confidence: float  # 0.0 to 1.0
    mathematical_reasoning_summary: str
    supporting_evidence: List[str]
    human_guidance_needed: bool
    recommended_verification_steps: List[str]
```

##### `ConsensusSynthesisOutput`
```python
class ConsensusSynthesisOutput(BaseModel):
    synthesis_approach: str
    evidence_summaries: List[MathematicalEvidenceSummary]
    contradiction_analysis: ContradictionAnalysis
    confidence_analysis: str
    responsible_answer: ResponsibleMathematicalAnswer
    federation_performance_assessment: str
```

#### Methods

##### `async synthesize_final_mathematical_answer(problem: str) -> ConsensusSynthesisOutput`
Synthesize findings from all nodes into responsible final answer.
- **Parameters:**
  - `problem`: Original mathematical problem
- **Returns:** Complete consensus synthesis
- **Dependencies:** Requires outputs from all previous nodes

---

## Output Files

### JSON Output Structure

Each node generates a JSON file with its analysis results:

- `math_node1_output.json` - Problem classification
- `math_node2_output.json` through `math_node9_output.json` - Specialized analyses  
- `math_node10_output.json` - Consensus synthesis
- `math_federation_final_answer.json` - Final consolidated answer

### Final Answer Format

The `math_federation_final_answer.json` contains:

```json
{
  "question": "Original mathematical problem",
  "answer_type": "CONFIDENT_ANSWER|UNCERTAIN_RANGE|DEFER_TO_HUMAN",
  "primary_answer": "Main mathematical answer",
  "confidence_range": "Range if uncertain",
  "system_confidence": 0.85,
  "reasoning": "Mathematical reasoning summary",
  "human_guidance_needed": false,
  "federation_performance": "Performance assessment"
}
```

---

## Error Handling

### Common Exceptions

- **ValueError**: Invalid API key or empty input files
- **FileNotFoundError**: Missing problem files or node output dependencies
- **OpenAI API Errors**: Rate limiting, invalid requests, or API failures
- **JSON Parse Errors**: Malformed OpenAI responses or corrupted output files

### Error Recovery

The system implements graceful error handling:
- Continued execution when individual nodes fail
- Fallback data generation for missing dependencies
- Comprehensive logging for debugging
- Structured error reporting in output files

---

## Usage Examples

### Basic Analysis
```python
pipeline = CompleteMathematicalFederationPipeline()
result = await pipeline.run_complete_analysis("Find all integer solutions to x² + y² = 169")
```

### File-Based Analysis
```python
pipeline = CompleteMathematicalFederationPipeline()
problem = pipeline.load_problem_from_markdown("problem.md")
result = await pipeline.run_complete_analysis(problem, "results.md")
```

### Individual Node Usage
```python
node1 = MathNode1ProblemClassifier(api_key)
classification = await node1.classify_problem(problem)

node2 = SpecializedMathematicalNode(api_key)
analysis = await node2.analyze_mathematically(problem)
```