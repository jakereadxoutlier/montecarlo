#!/usr/bin/env python3
"""
Script to merge stockflow.py and standalone_slack_app.py into montecarlo_unified.py
"""

# Read stockflow.py and extract all functions except MCP server stuff
with open('stockflow.py', 'r') as f:
    stockflow_lines = f.readlines()

# Find where the actual functions start (after imports and config)
functions_start = 0
for i, line in enumerate(stockflow_lines):
    if 'def fetch_alpha_vantage_data' in line or 'async def fetch_alpha_vantage_data' in line:
        functions_start = i
        break

# Find where MCP server stuff starts (we'll stop before this)
mcp_start = len(stockflow_lines)
for i, line in enumerate(stockflow_lines):
    if 'class Server:' in line or 'async def main():' in line:
        mcp_start = i
        break

# Extract the functions section
functions_section = ''.join(stockflow_lines[functions_start:mcp_start])

# Remove any Slack-specific message handlers that conflict
functions_section = functions_section.replace('async def handle_message_events', 'async def OLD_handle_message_events')
functions_section = functions_section.replace('async def start_slack_app', 'async def OLD_start_slack_app')
functions_section = functions_section.replace('async def stop_slack_app', 'async def OLD_stop_slack_app')

print(f"Extracted {mcp_start - functions_start} lines of trading functions from stockflow.py")
print(f"Functions section starts at line {functions_start}")
print(f"MCP section starts at line {mcp_start}")

# Save the extracted functions to a temporary file
with open('extracted_functions.py', 'w') as f:
    f.write(functions_section)

print("Trading functions saved to extracted_functions.py")
print("Next step: Insert these into montecarlo_unified.py")