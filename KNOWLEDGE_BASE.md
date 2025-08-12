# Knowledge Base - 10-Node Mathematical Federation System

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [System Components](#system-components)
3. [Development Workflows](#development-workflows)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Best Practices](#best-practices)
6. [Extension Guide](#extension-guide)
7. [Cross-References](#cross-references)

---

## Quick Start Guide

### Prerequisites Checklist
- ✅ Python 3.8+ installed
- ✅ OpenAI API key obtained
- ✅ Virtual environment created
- ✅ Dependencies installed (`openai`, `pydantic`)

### Essential Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install openai pydantic
export OPENAI_API_KEY='your-key-here'

# Run analysis
python run_math_analysis.py "Find all solutions to x^2 + y^2 = 25"
python run_math_analysis.py --file problem.md --output results.md

# Test system
python simple_test.py
```

### First Run Success Indicators
- ✅ All 10 nodes execute without errors
- ✅ JSON output files generated (11 total)  
- ✅ Final answer with confidence score
- ✅ Markdown report (if requested)

---

## System Components

### Core Files Overview

| Component | File | Purpose | Key Classes |
|-----------|------|---------|-------------|
| **Orchestrator** | `run_math_analysis.py` | Main pipeline coordination | `CompleteMathematicalFederationPipeline` |
| **Problem Classifier** | `math_node1.py` | Problem analysis & federation guidance | `MathNode1ProblemClassifier` |
| **Algebraic Specialist** | `math_node2.py` | Equation solving & algebraic manipulation | `SpecializedMathematicalNode` |
| **Geometric Specialist** | `math_node3.py` | Spatial reasoning & constructions | `SpecializedMathematicalNode` |
| **Combinatorial Specialist** | `math_node4.py` | Counting & arrangements | `SpecializedMathematicalNode` |
| **Number Theory Specialist** | `math_node5.py` | Integer properties & modular arithmetic | `SpecializedMathematicalNode` |
| **Calculus Specialist** | `math_node6.py` | Continuous mathematics & analysis | `SpecializedMathematicalNode` |
| **Discrete Math Specialist** | `math_node7.py` | Logic, graphs & algorithms | `SpecializedMathematicalNode` |
| **Verification Specialist** | `math_node8.py` | Consistency checking & validation | `SpecializedMathematicalNode` |
| **Alternative Methods** | `math_node9.py` | Creative & unconventional approaches | `SpecializedMathematicalNode` |
| **Consensus Synthesizer** | `math_node10.py` | Final answer generation | `MathNode10ConsensusSynthesizer` |
| **Testing Utility** | `simple_test.py` | Markdown functionality testing | N/A |

### Data Flow Map

```
Input Problem → Node 1 → Nodes 2-9 → Node 10 → Final Answer
              ↓         ↓           ↓        ↓
            JSON      JSON        JSON    JSON + MD
```

**File Dependencies:**
- `math_node1_output.json` → Required by Nodes 2-9
- `math_node[2-9]_output.json` → Required by Node 10
- `math_node10_output.json` → Contains consensus synthesis
- `math_federation_final_answer.json` → Final consolidated result

---

## Development Workflows

### Adding a New Mathematical Domain

**Example: Adding Statistics Node (Node 11)**

1. **Create Node File**
   ```bash
   cp math_node2.py math_node11.py
   ```

2. **Configure Specialization**
   ```python
   SPECIALIZATION_CONFIG = {
       "node_id": "math_node_11",
       "node_name": "Statistical Analysis",
       "mathematical_domain": "statistics",
       "primary_techniques": ["hypothesis testing", "regression", "probability distributions"],
       "output_filename": "math_node11_output.json",
       "specialization_prompt": "You are an expert in statistical analysis..."
   }
   ```

3. **Update Pipeline Orchestrator**
   ```python
   # In run_math_analysis.py
   from math_node11 import SpecializedMathematicalNode as MathNode11StatisticalAnalyzer
   
   # Add to __init__
   "node11": MathNode11StatisticalAnalyzer(self.api_key)
   
   # Add run_node11 method and update run_complete_analysis
   ```

4. **Extend Federation Guidance**
   ```python
   # In math_node1.py, update classification prompts to include statistics domain
   ```

### Modifying Node Behavior

**Common Customizations:**

1. **Enhanced Prompting**
   ```python
   # Modify specialization_prompt in SPECIALIZATION_CONFIG
   "specialization_prompt": """
   You are a world-class expert in [domain].
   Focus on [specific techniques].
   Consider [domain-specific factors].
   """
   ```

2. **Additional Data Fields**
   ```python
   # Extend SpecializedMathAnalysisOutput model
   class SpecializedMathAnalysisOutput(BaseModel):
       # ... existing fields
       custom_analysis: Optional[str] = None
       domain_metrics: Optional[Dict[str, float]] = None
   ```

3. **Custom Validation Logic**
   ```python
   # Add domain-specific validation in analyze_mathematically method
   if our_domain == "statistics":
       # Custom statistics validation
   ```

### Testing Workflows

**Unit Testing Individual Nodes**
```python
# Test Node 1
async def test_node1():
    classifier = MathNode1ProblemClassifier(api_key)
    result = await classifier.classify_problem("Test problem")
    assert result.confidence > 0.5

# Test Template Node
async def test_node2():
    analyzer = SpecializedMathematicalNode(api_key)
    result = await analyzer.analyze_mathematically("2x + 3 = 7")
    assert "algebraic" in result.mathematical_approach.lower()
```

**Integration Testing**
```python
# Test full pipeline
async def test_pipeline():
    pipeline = CompleteMathematicalFederationPipeline()
    result = await pipeline.run_complete_analysis("Simple test problem")
    assert result["answer_type"] in ["CONFIDENT_ANSWER", "UNCERTAIN_RANGE", "DEFER_TO_HUMAN"]
```

**Markdown Testing**
```bash
python simple_test.py  # Tests without API calls
```

### Debugging Workflows

**Common Debug Steps:**

1. **Check API Key**
   ```bash
   echo $OPENAI_API_KEY  # Should show your key
   ```

2. **Verify Node Outputs**
   ```bash
   ls math_node*_output.json  # Should show 11 files after successful run
   cat math_node1_output.json | head -20  # Check Node 1 output
   ```

3. **Enable Verbose Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Isolate Node Failures**
   ```python
   # Run individual nodes for debugging
   node1 = MathNode1ProblemClassifier(api_key)
   result = await node1.classify_problem(problem)
   ```

---

## Troubleshooting Guide

### Common Issues & Solutions

#### Issue: "OpenAI API key not found"
**Symptoms:** ValueError on startup
**Solutions:**
```bash
export OPENAI_API_KEY='sk-your-key-here'
# Or create .env file with OPENAI_API_KEY=sk-your-key-here
```

#### Issue: "Node X output file not found"  
**Symptoms:** FileNotFoundError during execution
**Solutions:**
- Check if previous node completed successfully
- Verify JSON file permissions
- Run nodes individually to isolate failure point

#### Issue: "JSON parse error"
**Symptoms:** JSON decoding errors
**Solutions:**
- Check OpenAI API response format
- Verify API quota and rate limits
- Examine malformed JSON in output files

#### Issue: "Markdown generation fails"
**Symptoms:** Error in save_results_to_markdown
**Solutions:**
- Check file permissions in output directory
- Verify final_result dictionary structure
- Test with simple_test.py first

#### Issue: "Low confidence scores across all nodes"
**Symptoms:** System confidence < 0.3
**Analysis:**
- Problem may be ambiguous or outside system capabilities
- Check if problem domain matches node specializations
- Review Node 1 classification for domain mismatch

#### Issue: "Inconsistent results between runs"
**Symptoms:** Different answers for identical problems
**Analysis:**
- OpenAI API responses have inherent variability
- Node confidence scores should reflect uncertainty
- Consider multiple runs for important problems

### Performance Issues

#### Slow Execution
**Typical Causes:**
- Network latency to OpenAI API
- Rate limiting or quota restrictions
- Large problem complexity requiring extensive reasoning

**Optimizations:**
- Use structured outputs to reduce token usage
- Implement local caching for repeated problems
- Consider parallel execution for Nodes 2-9

#### Memory Usage
**Monitoring:**
```python
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

**Typical Usage:** 20-50MB during execution

---

## Best Practices

### Development Guidelines

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables or secure key management
   - Rotate keys regularly

2. **Error Handling**
   - Always use try-catch for API calls
   - Provide meaningful error messages
   - Log errors with sufficient context for debugging

3. **Configuration Management**
   - Keep SPECIALIZATION_CONFIG centralized
   - Use consistent naming conventions
   - Document configuration changes

4. **Testing Practices**
   - Test individual nodes before integration
   - Use simple_test.py for non-API testing
   - Create test problems for each mathematical domain

### Mathematical Problem Guidelines

**Well-Suited Problems:**
- ✅ Clear mathematical questions with definite answers
- ✅ Problems within standard mathematical domains
- ✅ Questions requiring step-by-step reasoning
- ✅ Problems benefiting from multiple perspectives

**Challenging Problems:**
- ⚠️ Extremely open-ended or philosophical questions
- ⚠️ Problems requiring visual diagrams or complex notation
- ⚠️ Research-level mathematics beyond standard techniques
- ⚠️ Problems requiring extensive numerical computation

### Output Interpretation

**Confidence Score Guidelines:**
- **0.8-1.0:** High confidence, answer likely correct
- **0.6-0.8:** Moderate confidence, verify with additional methods
- **0.4-0.6:** Low confidence, human review recommended
- **0.0-0.4:** Very low confidence, problem may be beyond system capabilities

**Answer Types:**
- **CONFIDENT_ANSWER:** Single definitive solution
- **UNCERTAIN_RANGE:** Multiple possible answers or ranges
- **DEFER_TO_HUMAN:** Problem requires human mathematical expertise

---

## Extension Guide

### Adding New Mathematical Techniques

**Extending Existing Nodes:**
1. Add techniques to `primary_techniques` in SPECIALIZATION_CONFIG
2. Update prompts to include new techniques
3. Add technique-specific validation logic

**Creating Specialized Sub-Nodes:**
1. Create technique-specific classes inheriting from SpecializedMathematicalNode
2. Override analyze_mathematically with specialized logic
3. Register with main orchestrator

### Integration Enhancements

**External Tool Integration:**
```python
# Example: Adding SymPy integration
import sympy as sp

class EnhancedSymbolicNode(SpecializedMathematicalNode):
    async def analyze_mathematically(self, problem: str):
        # Use SymPy for symbolic computation
        symbolic_result = sp.solve(problem_equation)
        
        # Integrate with existing analysis
        base_result = await super().analyze_mathematically(problem)
        base_result.symbolic_verification = str(symbolic_result)
        return base_result
```

**Database Integration:**
```python
# Store historical results for learning
class DatabaseEnhancedPipeline(CompleteMathematicalFederationPipeline):
    def save_analysis_history(self, problem, result):
        # Store in database for future reference
        pass
```

### UI/UX Enhancements

**Web Interface:**
- Flask/FastAPI wrapper around pipeline
- Problem submission form
- Real-time progress tracking
- Interactive result exploration

**Batch Processing:**
- Multiple problem processing
- Result comparison and analysis
- Performance benchmarking

---

## Cross-References

### Related Documentation

| Document | Purpose | Key Sections |
|----------|---------|--------------|
| **[CLAUDE.md](CLAUDE.md)** | Claude Code integration | Commands, Architecture, Patterns |
| **[PROJECT_INDEX.md](PROJECT_INDEX.md)** | Original system overview | Usage, Architecture, Features |
| **[API_REFERENCE.md](API_REFERENCE.md)** | Complete API documentation | Classes, Methods, Data Models |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Detailed architectural design | Patterns, Components, Flows |

### Code Cross-References

#### Class Inheritance Hierarchy
```
BaseModel (Pydantic)
├── ProblemClassificationOutput
├── SpecializedMathAnalysisOutput  
└── ConsensusSynthesisOutput
    └── ResponsibleMathematicalAnswer
```

#### Configuration Dependencies
```
SPECIALIZATION_CONFIG (Nodes 2-9)
├── node_id → Used in file naming and logging
├── mathematical_domain → Used in prompts and validation
├── primary_techniques → Used in prompt generation
└── output_filename → Used in file I/O operations
```

#### Data Flow Dependencies
```
Node 1 → Node 1 Output JSON → Nodes 2-9 (parallel consumption)
Nodes 2-9 → Individual Output JSONs → Node 10 (synthesis)
Node 10 → Final Answer JSON + Optional Markdown
```

### External Dependencies

**Python Packages:**
- `openai`: OpenAI API client for AI model access
- `pydantic`: Data validation and settings management
- `asyncio`: Asynchronous programming support
- `logging`: Structured logging and debugging
- `json`: Data serialization and file I/O
- `pathlib`: Modern path handling utilities

**Environment Dependencies:**
- `OPENAI_API_KEY`: Required environment variable
- Python 3.8+: Minimum Python version requirement
- Internet connectivity: Required for OpenAI API access

### Performance Metrics

**Typical Execution Times:**
- Simple algebra problems: 3-5 minutes
- Complex multi-domain problems: 10-15 minutes
- Advanced mathematical proofs: 15-20 minutes

**Resource Usage Patterns:**
- API Token Usage: ~2000-5000 tokens per problem
- Memory Usage: ~20-50MB peak during execution
- Storage: ~50-100KB JSON output per analysis

### Support Resources

**Debugging:**
- Check `simple_test.py` for basic functionality
- Review individual JSON outputs for node-specific issues
- Enable debug logging for detailed execution traces

**Community:**
- Mathematical problem examples in `*_problem.md` files
- Result examples in `*_results.md` files  
- Test cases in `simple_test.py`

This knowledge base provides comprehensive guidance for understanding, developing, and extending the 10-Node Mathematical Federation System.