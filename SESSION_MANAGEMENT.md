# Claude Code Session Management Guide
## 10-Node Mathematical Federation System

> **Complete guide for starting and ending Claude Code sessions effectively**

---

## 🚀 **SESSION STARTUP CHECKLIST**

### **1. Essential Pre-Session Preparation**

#### **Directory Verification**
```bash
# Verify you're in the correct directory
pwd
# Should show: /Users/keithlambert/Desktop/10 nodes math brain

# Quick system status check
ls -la *.py *.md
# Should show all 10 math_node files + run_math_analysis.py + documentation
```

#### **Environment Setup**
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# Verify API key is set
echo $OPENAI_API_KEY
# Should show your API key (first few characters)

# Quick dependency check
python -c "import openai, pydantic; print('✅ Dependencies OK')"
```

#### **System Health Check**
```bash
# Quick system test (optional but recommended)
python simple_test.py
# Should complete without critical errors
```

### **2. Documentation Quick Reference**

#### **Always Reference First:**
- **[CLAUDE.md](CLAUDE.md)** - Claude Code integration commands and architecture
- **[SESSION_MANAGEMENT.md](SESSION_MANAGEMENT.md)** - This file (session procedures)
- **[KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md)** - Development workflows and troubleshooting

#### **Context Loading Commands:**
```bash
# Load project overview
head -50 README.md

# Check system status
git status

# Review recent changes
git log --oneline -5

# Check for any pending work
cat TODO.md 2>/dev/null || echo "No TODO file found"
```

### **3. Claude Code Context Establishment**

#### **Key Information to Share:**
1. **Project Type:** "10-Node Mathematical Federation System - distributed AI for mathematical reasoning"
2. **Current Status:** "Production-ready system with optimized Node 10 and comprehensive documentation"
3. **Architecture:** "Sequential pipeline: Classification → 8 Specialists → Consensus Synthesis"
4. **Key Features:** "Competition-level math problems, responsible AI, universal problem handling"

#### **Essential Files Overview:**
```
Core System:
├── run_math_analysis.py     # Main orchestrator
├── math_node1.py           # Problem classifier  
├── math_node2-9.py         # 8 mathematical specialists
└── math_node10.py          # Optimized consensus synthesis

Documentation:
├── CLAUDE.md              # Claude Code integration guide
├── README.md              # Comprehensive project overview
├── ARCHITECTURE.md        # Technical architecture
├── KNOWLEDGE_BASE.md      # Development workflows
└── FUTURE_VISION.md       # Universal federation framework vision
```

---

## 🛡️ **SESSION ENDING CHECKLIST**

### **1. Code & Documentation Status**

#### **File Integrity Check**
```bash
# Verify all core files exist and are readable
for file in math_node{1..10}.py run_math_analysis.py; do
    if [[ -f "$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ $file MISSING"
    fi
done

# Check documentation completeness  
for doc in CLAUDE.md README.md ARCHITECTURE.md KNOWLEDGE_BASE.md; do
    if [[ -f "$doc" ]]; then
        echo "✅ $doc"
    else
        echo "❌ $doc MISSING"
    fi
done
```

#### **System Functionality Verification**
```bash
# Quick functionality test
python -c "
from run_math_analysis import CompleteMathematicalFederationPipeline
pipeline = CompleteMathematicalFederationPipeline()
print('✅ Core system can be imported and instantiated')
"

# Verify Node 10 optimization
python -c "
from math_node10 import MathNode10ConsensusSynthesizer
node10 = MathNode10ConsensusSynthesizer()
print('✅ Optimized Node 10 available')
print('✅ Error recovery methods:', hasattr(node10, 'create_fallback_synthesis'))
"
```

### **2. Git Repository Management**

#### **Working Directory Status**
```bash
# Check for uncommitted changes
git status

# Review recent work
git log --oneline -3

# Check remote synchronization
git remote -v
git branch -v
```

#### **Commit Outstanding Work** (if any changes exist)
```bash
# Stage important changes
git add *.py *.md

# Create meaningful commit message
git commit -m "session: [describe work completed]

- [specific achievement 1]
- [specific achievement 2]  
- [any issues resolved]

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to remote repository
git push origin main
```

### **3. Documentation Updates**

#### **Session Work Documentation**
```bash
# Update SESSION_LOG.md (create if doesn't exist)
echo "## Session $(date '+%Y-%m-%d %H:%M')

### Work Completed:
- [List major accomplishments]
- [Any optimizations or fixes]
- [Documentation updates]

### System Status:
- Core functionality: ✅ Working
- Node 10 optimization: ✅ Complete  
- Documentation: ✅ Current
- Repository: ✅ Synchronized

### Notes for Next Session:
- [Any pending items]
- [Areas for improvement]
- [Research directions]

---" >> SESSION_LOG.md
```

#### **Critical Files Backup Check**
```bash
# Verify critical configurations are preserved
ls -la math_node10.py  # Optimized consensus synthesis
ls -la CLAUDE.md       # Claude Code integration guide
ls -la FUTURE_VISION.md # Universal federation roadmap

# Check file sizes are reasonable (not corrupted)
wc -l *.py *.md | tail -1  # Should show total line counts
```

### **4. System State Preservation**

#### **Environment Documentation**
```bash
# Document current environment state
echo "## Environment Status $(date)
- Python Version: $(python --version)
- Virtual Environment: $(which python)
- API Key Status: $([ -n "$OPENAI_API_KEY" ] && echo "✅ Set" || echo "❌ Not Set")
- Working Directory: $(pwd)
- Git Branch: $(git branch --show-current)
- Git Commit: $(git rev-parse --short HEAD)
" > .env_status
```

#### **Clean Temporary Files**
```bash
# Clean up temporary outputs (preserve in .gitignore)
ls math_node*_output.json 2>/dev/null | wc -l | xargs echo "JSON outputs:"
ls __pycache__ 2>/dev/null && echo "Python cache exists"

# Note: These are preserved by .gitignore but good to be aware
```

---

## 📋 **CRITICAL REFERENCE DOCUMENTS**

### **Always Consult Before Major Changes:**

1. **[CLAUDE.md](CLAUDE.md)** - Claude Code integration patterns and essential commands
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design principles and patterns  
3. **[KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md)** - Development workflows and troubleshooting
4. **[API_REFERENCE.md](API_REFERENCE.md)** - Class definitions and method signatures

### **Project State References:**

1. **[README.md](README.md)** - Comprehensive project overview and research vision
2. **[FUTURE_VISION.md](FUTURE_VISION.md)** - Universal federation framework roadmap
3. **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Navigation to all documentation

---

## 🔧 **COMMON SESSION WORKFLOWS**

### **Debugging Node Issues**
```bash
# Test individual nodes
python math_node1.py   # Problem classification
python math_node10.py  # Consensus synthesis

# Check for common issues
grep -r "ERROR\|Exception" *.py | head -5
```

### **Running Mathematical Analysis**
```bash
# Direct problem input
python run_math_analysis.py "Your mathematical problem here"

# File-based analysis with output
python run_math_analysis.py --file problem.md --output results.md

# Test with provided examples
python run_math_analysis.py --file simple_problem.md
python run_math_analysis.py --file complex_problem.md
```

### **System Enhancement Workflow**
```bash
# 1. Understand current architecture
head -100 ARCHITECTURE.md

# 2. Review enhancement patterns  
grep -A 10 "enhancement" KNOWLEDGE_BASE.md

# 3. Implement with testing
python simple_test.py  # Before changes
# [Make modifications]
python simple_test.py  # After changes

# 4. Update documentation
# [Update relevant .md files]

# 5. Commit with meaningful message
git add . && git commit -m "enhance: [description]"
```

---

## 🎯 **SESSION SUCCESS CRITERIA**

### **Every Session Should End With:**

✅ **System Functional** - Core pipeline works without errors  
✅ **Documentation Current** - All changes reflected in relevant docs  
✅ **Repository Synchronized** - All important work committed and pushed  
✅ **Environment Stable** - Dependencies and configuration preserved  
✅ **Progress Documented** - Clear record of accomplishments and next steps  

### **Session Quality Indicators:**

- **High Quality Session:** All 5 criteria met, significant progress documented
- **Standard Session:** 4/5 criteria met, routine maintenance completed  
- **Incomplete Session:** <4 criteria met, requires immediate follow-up

---

## 💡 **TIPS FOR EFFECTIVE CLAUDE CODE SESSIONS**

### **Session Planning**
1. **Start with clear objectives** - What specific goal are you trying to achieve?
2. **Reference documentation first** - Don't reinvent patterns that already exist
3. **Test frequently** - Use `simple_test.py` and individual node tests
4. **Document as you go** - Update relevant docs immediately after changes

### **Collaboration Patterns**
1. **Leverage existing architecture** - Build on the federation pattern
2. **Follow established conventions** - Use existing naming and structure patterns  
3. **Maintain backwards compatibility** - Preserve existing interfaces
4. **Think in terms of specialization** - Each node should have clear expertise boundaries

### **Quality Maintenance**
1. **Always test Node 10** - It's the critical synthesis component
2. **Validate JSON outputs** - Ensure proper schema compliance
3. **Check error recovery** - Test fallback mechanisms work
4. **Verify documentation accuracy** - Examples should be current and working

---

*This session management guide ensures consistent, high-quality work on the 10-Node Mathematical Federation System. Reference frequently to maintain system integrity and development velocity.*