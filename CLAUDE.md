# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **10-Node Mathematical Federation System** - a distributed AI architecture that solves mathematical problems using specialized nodes that work together in sequence. Each node specializes in a specific mathematical domain and contributes to a collaborative solution process designed for **competition-level mathematical analysis**.

## Core Architecture

### Mathematical Federation Pipeline
The system follows a three-phase architecture with **enhanced deep analysis capabilities**:

1. **Problem Classification (Node 1)** → Analyzes problem and provides federation guidance
2. **Specialized Analysis (Nodes 2-9)** → **Competition-level domain-specific mathematical reasoning**
3. **Consensus Synthesis (Node 10)** → Integration and final answer with confidence assessment

### Node Specializations
- **Node 1**: Problem Classification & Federation Coordination (`math_node1.py`)
- **Node 2**: Algebraic Analysis - *Enhanced with competition-level algebraic reasoning* (`math_node2.py`)  
- **Node 3**: Geometric Analysis - *Enhanced with spatial visualization and coordinate geometry* (`math_node3.py`)
- **Node 4**: Combinatorial Analysis - *Enhanced with Pigeonhole Principle and discrete counting* (`math_node4.py`)
- **Node 5**: Number Theory Analysis - *Enhanced with modular arithmetic and advanced techniques* (`math_node5.py`)
- **Node 6**: Calculus Analysis - *Enhanced with continuous optimization and analytical methods* (`math_node6.py`)
- **Node 7**: Discrete Mathematics - *Enhanced with structural analysis and algorithmic reasoning* (`math_node7.py`)
- **Node 8**: Symbolic Verification (`math_node8.py`)
- **Node 9**: Alternative Methods (`math_node9.py`)
- **Node 10**: Consensus Synthesis (`math_node10.py`)

### Enhanced Template Pattern
**Nodes 2-9** now use significantly enhanced template architecture:
- **Competition-level prompting** with detailed domain-specific guidance
- **Rich output models** with proof strategies, constraint analysis, and mathematical connections
- **12,000 token limits** for deep mathematical analysis (vs. original 2,500)
- **Enhanced fallback responses** with meaningful mathematical content

## Essential Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install openai pydantic

# Set OpenAI API key (REQUIRED)
export OPENAI_API_KEY='your-api-key-here'
```

### Running the System
```bash
# Direct problem input (recommended for complex problems)
python run_math_analysis.py "Find all integer solutions to x² + y² = 169"

# Load problem from markdown file (useful for IMO-level problems)
python run_math_analysis.py --file simple_problem.md
python run_math_analysis.py --file complex_problem.md

# Save results to markdown (generates comprehensive analysis report)
python run_math_analysis.py --file problem.md --output results.md
```

### Testing Individual Nodes
```bash
# Test enhanced mathematical nodes individually (all nodes 2-9 are now enhanced)
python math_node2.py  # Algebraic analysis
python math_node3.py  # Geometric analysis
python math_node4.py  # Combinatorial analysis
python math_node5.py  # Number theory analysis
python math_node6.py  # Calculus analysis
python math_node7.py  # Discrete mathematics
python math_node8.py  # Symbolic verification
python math_node9.py  # Alternative methods

# Basic functionality test
python simple_test.py
```

## Enhanced Data Flow & Dependencies

### Sequential Execution with Rich Context
Nodes execute in strict sequence (1→2→3→...→10) with **enhanced cross-node communication**:
- `math_node1_output.json` through `math_node10_output.json`
- `math_federation_final_answer.json` (final consolidated result)
- **Rich previous node referencing** with detailed findings integration

### Enhanced Cross-Node Communication
- **Node 1** provides detailed federation guidance and problem classification
- **Nodes 2-9** reference Node 1's analysis AND previous specialized node findings
- **Enhanced constraint analysis** shared across domains (e.g., p+q < n analyzed algebraically, geometrically, and combinatorially)
- **Node 10** synthesizes comprehensive findings from all enhanced analyses

### Enhanced Output Structure
Each node now produces **significantly richer structured output**:

**Node 1**: `ProblemClassificationOutput`
- Problem domain classification with confidence scoring
- Federation guidance with specific instructions for each node
- Complexity assessment and solution strategies

**Nodes 2-9**: `SpecializedMathAnalysisOutput` *(ENHANCED)*
- `mathematical_approach`: Detailed domain-specific approach (3-4 sentences)
- `proof_strategy`: Specific proof methodology
- `constraint_analysis`: Analysis of mathematical constraints
- `technique_applications`: 4-6 specific techniques applied
- `reasoning_steps`: 5-7 detailed mathematical steps with justifications
- `key_insights`: 3-5 substantial insights with mathematical significance
- `key_observations`: Important mathematical observations
- `mathematical_connections`: Connections to other mathematical areas
- `alternative_approaches`: Alternative methods considered
- Enhanced confidence scoring and domain-specific notes

**Node 10**: `ResponsibleMathematicalAnswer`
- Confidence-based answer classification
- Human guidance recommendations for uncertain problems
- Comprehensive synthesis of all node findings

## Key Implementation Patterns

### Enhanced Template Configuration
**Critical**: Nodes 2-9 follow enhanced template pattern with competition-level configuration:

```python
SPECIALIZATION_CONFIG = {
    "node_id": "math_node_X",
    "node_name": "Domain Name",
    "mathematical_domain": "domain_key", 
    "primary_techniques": ["technique1", "technique2", "advanced_technique3"],  # Enhanced
    "output_filename": "math_nodeX_output.json",
    "specialization_prompt": """
    You are a world-class expert in [domain], specializing in competition-level problems.
    
    Your expertise includes:
    - Advanced technique descriptions
    - Competition-level applications
    - Domain-specific guidance for complex problems
    """,  # Significantly enhanced prompting
    "approach_keywords": ["keyword1", "keyword2", "advanced_keyword3"],
}
```

### Enhanced API Configuration
**All enhanced nodes (2-9) use:**
- `max_completion_tokens=12000` (vs. original 2,500)
- Competition-level prompting with specific mathematical requirements
- **Relevance checks** - nodes assess problem fit before providing detailed analysis
- Enhanced fallback responses with meaningful mathematical content
- Structured JSON output with rich mathematical fields
- JSON parsing fixes for confidence field extraction
- Adaptive specialized prompting (no hardcoded domain-specific content)

### Enhanced Mathematical Analysis Process
1. **Problem Classification**: Node 1 provides detailed domain analysis and federation guidance
2. **Relevance Assessment**: Each specialized node assesses problem fit before deep analysis
3. **Multi-Domain Analysis**: Relevant nodes provide deep, competition-level analysis
4. **Cross-Domain Integration**: Nodes reference and build upon previous findings
5. **Constraint Synthesis**: Mathematical constraints analyzed from multiple perspectives
6. **Consensus Building**: Node 10 synthesizes comprehensive findings into final answer

## Current Enhancement Status

### Enhanced Nodes (Competition-Level) - ALL COMPLETE ✅
- ✅ **Node 2** (Algebraic): GCD/LCM analysis, factorization, algebraic manipulation
- ✅ **Node 3** (Geometric): Coordinate geometry, spatial visualization, lattice analysis  
- ✅ **Node 4** (Combinatorial): Pigeonhole Principle, discrete counting, existence proofs
- ✅ **Node 5** (Number Theory): Modular arithmetic, Pigeonhole Principle, advanced techniques
- ✅ **Node 6** (Calculus): Continuous optimization, analytical bounds, asymptotic analysis
- ✅ **Node 7** (Discrete Mathematics): Graph theory, structural analysis, algorithmic reasoning
- ✅ **Node 8** (Symbolic Verification): Formal verification, computational validation, consistency checking
- ✅ **Node 9** (Alternative Methods): Creative approaches, heuristic reasoning, innovative strategies

### System Architecture
All nodes use OpenAI's `gpt-5-mini` model with structured output via Pydantic models. The system runs asynchronously but executes sequentially to ensure proper dependency handling between nodes.

## File Organization & Data Persistence

- **Node files**: `math_nodeX.py` (X = 1-10) with consistent template architecture
- **Main orchestrator**: `run_math_analysis.py` - complete pipeline execution
- **Output files**: JSON outputs for each node plus consolidated final answer
- **Test problems**: `simple_problem.md` (IMO 1996 Problem 6), `complex_problem.md`
- **Results**: Generated markdown reports with comprehensive analysis

## Problem-Solving Capabilities

The enhanced system can handle:
- **IMO-level competition problems** (International Mathematical Olympiad)
- **Complex constraint analysis** (e.g., p+q < n analyzed across multiple domains)
- **Multi-domain mathematical reasoning** with cross-node synthesis
- **Existence proofs** using various mathematical approaches
- **Advanced number theory**, algebraic manipulation, geometric visualization
- **Competition-level combinatorics** and discrete mathematics

## Extension Points

### Adding New Mathematical Domains
1. Create new node file using enhanced template from Nodes 2-9
2. Configure enhanced `SPECIALIZATION_CONFIG` with competition-level prompting
3. Update orchestrator in `run_math_analysis.py`
4. Add federation guidance in Node 1

### Key Enhancement Features (All Nodes 2-9)
All enhanced nodes include:
1. Enhanced `SPECIALIZATION_CONFIG` with advanced techniques and competition-level prompting
2. **Relevance assessment** - nodes self-evaluate problem fit before detailed analysis
3. API calls with 12,000 token limits and specific mathematical requirements
4. Enhanced fallback responses with meaningful mathematical content
5. Rich Pydantic models with mathematical significance and problem connection fields
6. JSON parsing fixes for confidence field extraction and validation
7. Adaptive specialized prompting that avoids hardcoded domain-specific content

### Mathematical Problem Integration
The system excels with:
- Problems requiring multiple mathematical perspectives
- Competition-level complexity with rigorous proof requirements  
- Constraint analysis across different mathematical domains
- Problems where cross-domain synthesis provides insights