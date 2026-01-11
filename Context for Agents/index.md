# Context for AI Agents

This folder contains documentation specifically for AI assistants helping with this codebase. Not for human reading.

## Files

### `WORKFLOW_TEST.md`
Complete workflow verification. Explains what happens during:
- Fresh clone
- Running `./setup.sh`
- Creating new projects
- What files are tracked vs ignored
- npm-like workflow comparison

**When to use:** Understanding the full build process and what gets generated where.

### `CHANGES_SUMMARY.md`
History of Windows build system fixes. Explains:
- What problems existed
- Root causes
- Solutions implemented
- Before/after workflow comparison

**When to use:** Understanding why the build system works the way it does.

---

## Quick Reference for AI

### Build System Flow
```
1. User runs ./setup.sh
   → Builds SFML from source
   → Creates _sfml/SFML_DIR.txt (machine-specific, git-ignored)
   → Info only, doesn't build projects anymore

2. User runs cd "Project" && ./runApp.sh  
   → Auto-configures CMake with SFML_DIR (first time)
   → Builds project
   → Runs executable

3. User runs ./build_all.sh
   → Builds ALL projects to verify nothing broke
```

### Key Facts
- **SFML**: Built locally like node_modules, NOT in git
- **SFML_DIR.txt**: Machine-specific path, git-ignored, auto-generated
- **OS Detection**: Scripts detect Windows/macOS/Linux, use appropriate generators
- **Zero Config**: runApp.sh auto-configures on first run
- **Git Bash Compatible**: No sed/mkdir/find/head dependencies, uses bash builtins

### Common Issues
- **"CMake not found"**: Need to restart Git Bash after installation
- **Empty build folders**: Check build_all.sh output for actual errors
- **New projects**: Use create_sfml_project.sh, it copies template from "Random Walk"

### Script Naming
- `setup.sh` = Builds SFML only (one-time)
- `runApp.sh` = Configure + Build + Run project
- `build_all.sh` = Build all projects (CI-like verification)
- `create_sfml_project.sh` = New project generator

---

**Note:** This folder is for AI context only. Human users should read the main README.md.
