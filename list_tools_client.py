#!/usr/bin/env python3
import json
import asyncio
import subprocess
import sys

async def list_tools():
    # Start the MCP server process
    process = await asyncio.create_subprocess_exec(
        sys.executable, "stockflow.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        # Send initialize request first
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }

        init_json = json.dumps(init_request) + '\n'
        process.stdin.write(init_json.encode())
        await process.stdin.drain()

        # Read initialization response
        init_response = await process.stdout.readline()
        print("Init response:", init_response.decode().strip())

        # Send list tools request
        list_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }

        request_json = json.dumps(list_request) + '\n'
        process.stdin.write(request_json.encode())
        await process.stdin.drain()

        # Read the response
        response_line = await process.stdout.readline()
        if response_line:
            response = json.loads(response_line.decode().strip())
            print("\nAvailable Tools:")
            print(json.dumps(response, indent=2))

        # Now call the options chain tool
        options_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_options_chain_v2",
                "arguments": {
                    "symbol": "TSLA",
                    "include_greeks": True
                }
            }
        }

        options_json = json.dumps(options_request) + '\n'
        process.stdin.write(options_json.encode())
        await process.stdin.drain()
        process.stdin.close()

        # Read the options response
        options_response = await process.stdout.readline()
        if options_response:
            response = json.loads(options_response.decode().strip())
            print("\nTSLA Options Chain Response:")
            print(json.dumps(response, indent=2))

        # Wait for process to complete
        await process.wait()

    except Exception as e:
        print(f"Error: {e}")
        process.kill()
        await process.wait()

if __name__ == "__main__":
    asyncio.run(list_tools())