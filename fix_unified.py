#!/usr/bin/env python3
"""Fix the unified file by removing MCP server code"""

with open('montecarlo_unified_complete.py', 'r') as f:
    lines = f.readlines()

# Keep everything before MCP server (lines 0-4781)
part1 = lines[:4781]

# Skip MCP server section (4782-6057) and keep Slack handlers (6058+)
part2 = lines[6057:]

# Combine
fixed_content = part1 + part2

# Write fixed version
with open('montecarlo_unified_fixed.py', 'w') as f:
    f.writelines(fixed_content)

print(f"Created montecarlo_unified_fixed.py")
print(f"Original: {len(lines)} lines")
print(f"Fixed: {len(fixed_content)} lines")
print(f"Removed: {len(lines) - len(fixed_content)} lines of MCP server code")