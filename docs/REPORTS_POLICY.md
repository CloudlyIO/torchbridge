# Reports and Validation Policy

> **Version**: v0.4.15 | **Status**: ✅ Policy | **Last Updated**: Jan 22, 2026

## ⚠️ NEVER COMMIT REPORTS TO THE REPOSITORY

All test reports, validation results, and benchmark outputs must be stored **locally only**.

## Policy

### What Should NOT Be Committed

❌ **Validation reports** - Test results, summaries, analysis
❌ **Benchmark results** - Performance measurements, comparison data
❌ **Test artifacts** - Raw test outputs (.txt, .json)
❌ **Cloud testing reports** - Platform-specific validation data
❌ **Historical reports** - Archived test results

### Where to Store Reports

✅ **Local directory**: `~/Documents/shahmod-reports-backup/`

**Structure**:
```
~/Documents/shahmod-reports-backup/
├── 2026-01-22/
│   ├── validation-reports/
│   │   ├── latest/
│   │   └── archive/
│   └── benchmarks-results/
├── 2026-01-15/
└── 2026-01-08/
```

**Organization**:
- Create a new dated folder for each test run
- Keep reports organized by type (validation, benchmarks, etc.)
- Maintain your own archival schedule

## Why This Policy?

1. **Repository Focus**: Keep repo focused on code and essential documentation
2. **Size Management**: Prevent repo bloat from accumulated test data
3. **Clarity**: Separate implementation from validation artifacts
4. **Flexibility**: Test reports change frequently, code should be stable

## What IS Allowed in Repo

✅ **Documentation** - User guides, technical docs, architecture
✅ **Code** - Source code, tests, examples
✅ **Configuration** - Setup scripts, CI/CD configs
✅ **Essential metadata** - README, CHANGELOG, etc.

## Enforcement

### .gitignore Rules

The repository `.gitignore` blocks:
```gitignore
# All validation reports
docs/validation-reports/
**/validation-reports/

# All test reports
**/*_report.md
**/*_REPORT.md
**/*_validation*.md

# Test artifacts
**/test-artifacts/
**/*_results.json
```

### Pre-Commit Hook

A pre-commit hook prevents accidental commits of reports:
- Location: `.git/hooks/pre-commit`
- Checks for report files before allowing commit
- Provides helpful error messages with correct location

### Manual Check

Before committing:
```bash
# Check for reports in staging area
git diff --cached --name-only | grep -i "report\|validation\|summary"

# If any matches, unstage them
git reset HEAD <file>
```

## Sharing Test Results

If you need to share test results with team:

1. **Documentation**: Use Git for setup guides and procedures
2. **Results**: Share via email, Slack, cloud storage (not Git)
3. **Summaries**: Create brief summary in planning docs if needed (no raw data)

## Examples

### ✅ GOOD - What to Commit
```
docs/
├── getting-started/
├── guides/
├── cloud-deployment/
│   ├── validated-guide.md    ← Setup procedures
│   └── instance-selection.md ← Configuration info
└── capabilities/
```

### ❌ BAD - What NOT to Commit
```
docs/
├── validation-reports/        ← NO! Save to ~/Documents
├── test-results/              ← NO! Save to ~/Documents
└── benchmarks-2026-01-22.md   ← NO! Save to ~/Documents
```

## Questions?

See the team lead or refer to project documentation standards.

---

**Remember**: Reports belong in `~/Documents/shahmod-reports-backup/`, not in Git!
