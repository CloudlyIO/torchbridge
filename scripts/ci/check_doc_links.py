#!/usr/bin/env python3
"""
Check for broken internal links in documentation.
"""

import re
import sys
from pathlib import Path


def check_doc_links():
    """Check all markdown files for broken internal links."""
    docs_dir = Path(__file__).parent.parent / "docs"
    broken_links = []

    # Find all markdown files
    md_files = list(docs_dir.rglob("*.md"))

    # Pattern to match markdown links
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+\.md)\)')

    for md_file in md_files:
        content = md_file.read_text()
        matches = link_pattern.findall(content)

        for text, link in matches:
            # Skip external links
            if link.startswith("http"):
                continue

            # Resolve relative path
            link_path = md_file.parent / link

            # Check if target exists
            if not link_path.exists():
                # Try from docs root
                alt_path = docs_dir / link
                if not alt_path.exists():
                    broken_links.append({
                        "file": str(md_file.relative_to(docs_dir.parent)),
                        "link": link,
                        "text": text
                    })

    return broken_links


def main():
    """Run link check."""
    print("🔍 Checking documentation links...")

    broken = check_doc_links()

    if broken:
        print(f"\n❌ Found {len(broken)} broken links:\n")
        for item in broken:
            print(f"  {item['file']}")
            print(f"    → [{item['text']}]({item['link']}) NOT FOUND")
        return 1
    else:
        print("✅ All documentation links valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())
