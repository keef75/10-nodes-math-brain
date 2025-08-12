# Startup Guide - 10-Node Mathematical Federation System

> **Best practices for initiating Claude Code sessions and system startup**

---

## 🎯 **PRE-SESSION PREPARATION**

### **1. Environment Verification**

#### **Directory & Location Check**
```bash
# Verify correct project directory
pwd
# Expected: /Users/keithlambert/Desktop/10 nodes math brain

# Quick directory integrity check
ls -la | grep -E "(math_node|run_math|README)"
# Should show all core files present
```

#### **Python Environment Setup**
```bash
# Activate virtual environment
source venv/bin/activate

# Verify Python version and location
python --version  # Should be 3.8+
which python      # Should point to venv/bin/python

# Validate core dependencies
python -c "
import sys
try:
    import openai
    import pydantic
    print(f'✅ Dependencies OK - Python {sys.version.split()[0]}')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"
```

#### **API Configuration Validation**
```bash
# Check API key is set
if [[ -n "$OPENAI_API_KEY" ]]; then
    echo "✅ OpenAI API key configured (${OPENAI_API_KEY:0:8}...)"
else
    echo "❌ OpenAI API key not set"
    echo "Run: export OPENAI_API_KEY='your-key-here'"
fi
```

---

## 🚀 **SYSTEM INITIALIZATION**

### **2. System Health Verification**

#### **Core System Import Test**
```bash
# Test core system can be loaded
python -c "
try:
    from run_math_analysis import CompleteMathematicalFederationPipeline
    from math_node10 import MathNode10ConsensusSynthesizer
    print('✅ Core system modules load successfully')
    
    # Quick instantiation test
    pipeline = CompleteMathematicalFederationPipeline()
    node10 = MathNode10ConsensusSynthesizer()
    print('✅ System components instantiate correctly')
    
    # Check Node 10 optimizations
    has_fallback = hasattr(node10, 'create_fallback_synthesis')
    has_emergency = hasattr(node10, 'create_emergency_fallback_synthesis')
    print(f'✅ Node 10 optimization status: Fallback={has_fallback}, Emergency={has_emergency}')
    
except Exception as e:
    print(f'❌ System initialization error: {e}')
"
```

#### **System Functionality Test**
```bash
# Run lightweight system test
echo "Running system functionality test..."
python simple_test.py

# Verify test results
if [[ $? -eq 0 ]]; then
    echo "✅ System functionality test passed"
else
    echo "⚠️  System functionality test showed warnings (check output above)"
fi
```

#### **Individual Node Verification**
```bash
# Test critical nodes independently
echo "Testing Node 10 (Consensus Synthesis)..."
python -c "
import asyncio
from math_node10 import MathNode10ConsensusSynthesizer

async def test_node10():
    node10 = MathNode10ConsensusSynthesizer()
    # Test emergency fallback (doesn't need API call)
    result = node10.create_emergency_fallback_synthesis('Test problem')
    print(f'✅ Node 10 emergency fallback working: {type(result).__name__}')

asyncio.run(test_node10())
"
```

---

## 📊 **PROJECT STATE ANALYSIS**

### **3. Repository & Documentation Status**

#### **Git Repository State**
```bash
# Check git status and recent history
echo "=== Git Repository Status ==="
git status --porcelain | head -10
git log --oneline -5
git remote -v

# Check branch and synchronization
echo "Current branch: $(git branch --show-current)"
echo "Last commit: $(git rev-parse --short HEAD)"
```

#### **Documentation Currency Check**
```bash
# Verify critical documentation exists and is recent
echo "=== Documentation Status ==="
for doc in CLAUDE.md README.md ARCHITECTURE.md KNOWLEDGE_BASE.md FUTURE_VISION.md; do
    if [[ -f "$doc" ]]; then
        lines=$(wc -l < "$doc")
        echo "✅ $doc ($lines lines)"
    else
        echo "❌ $doc MISSING"
    fi
done
```

#### **System State Assessment**
```bash
# Check for any temporary files or work-in-progress
echo "=== System State ==="
ls -la *.json 2>/dev/null | wc -l | xargs echo "JSON output files:"
ls -la __pycache__/ 2>/dev/null && echo "Python cache present" || echo "No Python cache"
ls -la debug_*.py 2>/dev/null | wc -l | xargs echo "Debug files:"
```

---

## 🧠 **CLAUDE CODE CONTEXT ESTABLISHMENT**

### **4. Session Context Loading**

#### **Project Overview Refresher**
```bash
# Load key project information for context
echo "=== PROJECT CONTEXT ==="
echo "Project: $(head -1 README.md | sed 's/# //')"
echo "Architecture: Sequential federation pipeline (1 → 2-9 → 10)"
echo "Status: Production-ready with optimized Node 10"
echo "GitHub: https://github.com/keef75/10-nodes-math-brain"
```

#### **Recent Work Summary**
```bash
# Show recent significant changes
echo "=== RECENT WORK ==="
git log --oneline -3 --grep="feat\|docs\|enhance\|optimize"
```

#### **System Capabilities Reminder**
```bash
echo "=== SYSTEM CAPABILITIES ==="
echo "✅ Universal problem handling (any mathematical domain)"
echo "✅ Competition-level analysis (IMO/USAMO problems)"
echo "✅ Responsible AI with confidence calibration"
echo "✅ Triple-layered error recovery system"
echo "✅ Cross-domain validation via 8 specialists"
echo "✅ Comprehensive documentation suite"
```

---

## 📋 **SESSION READINESS CHECKLIST**

### **5. Final Startup Verification**

#### **Complete System Check**
```bash
echo "=== STARTUP READINESS CHECKLIST ==="

# Environment
echo -n "Python environment: "
[[ $(which python) == *"venv"* ]] && echo "✅" || echo "❌ Not in venv"

# Dependencies  
echo -n "Dependencies: "
python -c "import openai, pydantic" 2>/dev/null && echo "✅" || echo "❌"

# API Key
echo -n "OpenAI API Key: "
[[ -n "$OPENAI_API_KEY" ]] && echo "✅" || echo "❌"

# Core files
echo -n "Core system files: "
[[ -f "run_math_analysis.py" && -f "math_node10.py" ]] && echo "✅" || echo "❌"

# Documentation
echo -n "Essential documentation: "
[[ -f "CLAUDE.md" && -f "README.md" ]] && echo "✅" || echo "❌"

# Git repository
echo -n "Git repository: "
git rev-parse --git-dir >/dev/null 2>&1 && echo "✅" || echo "❌"

echo ""
echo "🚀 If all items show ✅, system is ready for Claude Code session!"
```

---

## 🎯 **SESSION INITIATION BEST PRACTICES**

### **6. Optimal Session Start**

#### **Context Sharing Strategy**
When starting a new Claude Code session, share this context:

```
PROJECT: 10-Node Mathematical Federation System
LOCATION: /Users/keithlambert/Desktop/10 nodes math brain
STATUS: Production-ready with optimized Node 10 consensus synthesis

ARCHITECTURE:
- Sequential federation pipeline: Classification → Specialists → Synthesis
- 8 mathematical domain specialists (algebra, geometry, number theory, etc.)
- Universal problem handling with responsible AI and error recovery

KEY ACHIEVEMENTS:
- Node 10 optimized with triple-layered fallback system
- Competition-level mathematical reasoning (IMO/USAMO problems)
- Comprehensive documentation suite and GitHub integration
- Future vision: Universal federated AI framework (10 → 10,000+ nodes)

CURRENT FOCUS: [State your session objective here]
```

#### **Documentation Priority Order**
1. **CLAUDE.md** - Essential Claude Code integration patterns
2. **STARTUP.md** - This startup guide (you're reading it!)
3. **README.md** - Comprehensive project overview  
4. **KNOWLEDGE_BASE.md** - Development workflows and troubleshooting
5. **FUTURE_VISION.md** - Universal federation framework roadmap

#### **Common Session Objectives**
- **Enhancement**: "Improving [specific component] for [specific goal]"
- **Debugging**: "Investigating [specific issue] in [specific component]"  
- **Documentation**: "Updating documentation for [specific changes]"
- **Testing**: "Validating [specific functionality] with [specific scenarios]"
- **Research**: "Exploring [specific research direction] for future development"

---

## ⚡ **RAPID STARTUP SEQUENCE**

### **7. Quick Start for Experienced Users**

```bash
# 30-second startup sequence
cd "/Users/keithlambert/Desktop/10 nodes math brain"
source venv/bin/activate
export OPENAI_API_KEY='your-key-here'
python -c "from run_math_analysis import CompleteMathematicalFederationPipeline; print('✅ System Ready')"
git status
```

### **Express System Test**
```bash
# 60-second full system validation
python run_math_analysis.py "What is 2+2?" && echo "✅ System fully operational"
```

---

## 🔍 **TROUBLESHOOTING STARTUP ISSUES**

### **8. Common Startup Problems**

#### **API Key Issues**
```bash
# Fix missing API key
export OPENAI_API_KEY='your-actual-key-here'
# Verify: echo $OPENAI_API_KEY
```

#### **Python Environment Issues**
```bash
# Recreate virtual environment if needed
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install openai pydantic
```

#### **Import Errors**
```bash
# Check and fix Python path issues
python -c "import sys; print('\\n'.join(sys.path))"
# Ensure current directory is in path
export PYTHONPATH=".:$PYTHONPATH"
```

#### **File Permission Issues**
```bash
# Fix file permissions if needed
chmod +x *.py
chmod 644 *.md
```

---

## 💡 **STARTUP SUCCESS INDICATORS**

### **Green Light Criteria**
✅ All environment checks pass  
✅ Core system imports successfully  
✅ Node 10 optimization confirmed  
✅ Git repository synchronized  
✅ Documentation current and accessible  
✅ API configuration validated  

### **Ready for Session When:**
- System responds to basic commands
- No critical errors in startup checks  
- All essential files present and readable
- Git status clean or changes understood
- Clear session objective established

---

*This startup guide ensures consistent, reliable initialization of the 10-Node Mathematical Federation System for productive Claude Code sessions.*