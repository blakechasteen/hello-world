# 🐷 TORUGH Quick Start Guide

**"From Slop to Prize-Winning Code in 4 Easy Steps!"**

---

## Step 1: Scan Your Entire Codebase 🔍

```bash
# Scan HoloLoom (default)
python torugh.py --max-files 50

# Scan a different directory
python torugh.py --dir ./my_project --max-files 100
```

**What You Get**: TORUGH_REPORT.md

---

## Step 2: Review the Classification 📊

```bash
cat TORUGH_REPORT.md
```

**Look For**:
- Auto-Fixable (Prize Pigs): Safe to fix
- Needs Review (Piglets): Need human judgment
- False Positives (Mud): Can ignore

---

## Step 3: Watch the Demo 🎬

```bash
python demo_complete_torugh.py
```

**You'll See**: Detection → Classification → Fixing → Validation → Git

---

## Step 4: Start Fixing 🔧

Currently 0 auto-fixes (being VERY conservative - good!)

To enable auto-fixes:
1. Add test coverage to files
2. Re-scan with torugh.py
3. xTerminator will be more confident!

---

## (*)<  The Barnyard Brigade

