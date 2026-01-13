"""
MCP Client for JIRA Ticket Creation via FastMCP

Uses Streamable HTTP transport (SSE) for FastMCP.cloud
Endpoint: https://scary-beige-sturgeon.fastmcp.app/mcp
"""

import os
import httpx
import json
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class JiraTicketResult:
    """Result of JIRA ticket creation"""
    success: bool
    ticket_id: Optional[str] = None
    ticket_url: Optional[str] = None
    error: Optional[str] = None


class MCPClient:
    """
    Async client for FastMCP JIRA integration.

    Uses Streamable HTTP transport (SSE) for FastMCP.cloud hosted endpoints.
    """

    def __init__(self):
        self.endpoint = os.getenv(
            "MCP_JIRA_ENDPOINT",
            "https://scary-beige-sturgeon.fastmcp.app/mcp"
        )
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "KAN")
        self.jira_url = os.getenv("JIRA_URL", "https://dataai12102.atlassian.net")
        self.timeout = 60.0  # seconds
        self._session_id = None

    async def _send_mcp_request(
        self,
        method: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Send MCP request using Streamable HTTP transport.

        Args:
            method: The MCP method to call
            params: Parameters for the method

        Returns:
            Response data from the MCP server
        """
        import uuid
        request_id = str(uuid.uuid4())

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        # Add session ID if we have one
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        print(f"[MCP] Sending request: method={method}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers=headers
            )

            print(f"[MCP] Response status: {response.status_code}")
            print(f"[MCP] Response headers: {dict(response.headers)}")

            # Check for session ID in response headers
            if "mcp-session-id" in response.headers:
                self._session_id = response.headers["mcp-session-id"]
                print(f"[MCP] Got session ID: {self._session_id}")

            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            # Handle SSE response
            if "text/event-stream" in content_type:
                return await self._parse_sse_response(response.text)
            else:
                # Regular JSON response
                return response.json()

    async def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response"""
        result = {}
        for line in text.split("\n"):
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        result = json.loads(data)
                    except json.JSONDecodeError:
                        pass
        return result

    async def initialize(self) -> bool:
        """
        Initialize MCP session.

        Returns:
            True if initialization succeeded
        """
        try:
            response = await self._send_mcp_request(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "it-chatbot",
                        "version": "1.0.0"
                    }
                }
            )
            print(f"[MCP] Initialize response: {response}")

            # Send initialized notification
            await self._send_mcp_request(
                method="notifications/initialized",
                params={}
            )

            return True
        except Exception as e:
            print(f"[MCP] Initialization failed: {e}")
            return False

    async def list_tools(self) -> list:
        """List available tools from the MCP server"""
        try:
            response = await self._send_mcp_request(
                method="tools/list",
                params={}
            )
            print(f"[MCP] Tools list response: {response}")
            return response.get("result", {}).get("tools", [])
        except Exception as e:
            print(f"[MCP] List tools failed: {e}")
            return []

    async def create_jira_issue(
        self,
        summary: str,
        description: str,
        issue_type: str = "Task",
        project_key: Optional[str] = None
    ) -> JiraTicketResult:
        """
        Create a JIRA issue via MCP.

        Args:
            summary: Ticket summary (max 255 chars)
            description: Full issue description
            issue_type: JIRA issue type (Task, Bug, Story)
            project_key: Override default project key

        Returns:
            JiraTicketResult with ticket ID and URL if successful
        """
        try:
            # Initialize session first
            print("[MCP] Starting JIRA ticket creation...")

            if not await self.initialize():
                return JiraTicketResult(
                    success=False,
                    error="Failed to initialize MCP session"
                )

            # List available tools to find the correct tool name
            tools = await self.list_tools()
            print(f"[MCP] Available tools: {[t.get('name') for t in tools]}")

            # Find JIRA create issue tool
            jira_tool = None
            for tool in tools:
                tool_name = tool.get("name", "").lower()
                if "create" in tool_name and ("issue" in tool_name or "jira" in tool_name):
                    jira_tool = tool
                    break

            if not jira_tool:
                # Try common tool names
                jira_tool = {"name": "create_issue"}

            tool_name = jira_tool.get("name", "create_issue")
            print(f"[MCP] Using tool: {tool_name}")

            # Prepare arguments based on tool schema
            arguments = {
                "project_key": project_key or self.project_key,
                "summary": summary[:255],
                "description": description,
                "issue_type": issue_type
            }

            # Call the tool
            response = await self._send_mcp_request(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                }
            )

            print(f"[MCP] Tool call response: {response}")

            # Parse response
            if "result" in response:
                result = response["result"]

                # Handle content array format
                content = result.get("content", [])
                if content and isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            ticket_id = self._extract_ticket_id(text)
                            if ticket_id:
                                return JiraTicketResult(
                                    success=True,
                                    ticket_id=ticket_id,
                                    ticket_url=self._build_ticket_url(ticket_id)
                                )
                            # If no ticket ID found but we got text, return it as success
                            if "created" in text.lower() or "success" in text.lower():
                                return JiraTicketResult(
                                    success=True,
                                    ticket_id="Created",
                                    ticket_url=text
                                )

                # Handle direct result format
                if isinstance(result, dict):
                    ticket_id = (
                        result.get("key") or
                        result.get("id") or
                        result.get("ticket_id") or
                        self._extract_ticket_id(str(result))
                    )
                    if ticket_id:
                        return JiraTicketResult(
                            success=True,
                            ticket_id=ticket_id,
                            ticket_url=self._build_ticket_url(ticket_id)
                        )

                # If we got a result but couldn't parse ticket ID
                return JiraTicketResult(
                    success=True,
                    ticket_id="Created",
                    ticket_url=str(result)
                )

            if "error" in response:
                error_msg = response["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                return JiraTicketResult(
                    success=False,
                    error=str(error_msg)
                )

            return JiraTicketResult(
                success=False,
                error=f"Unexpected response: {response}"
            )

        except httpx.HTTPStatusError as e:
            print(f"[MCP] HTTP error: {e.response.status_code} - {e.response.text}")
            return JiraTicketResult(
                success=False,
                error=f"HTTP error: {e.response.status_code}"
            )
        except httpx.RequestError as e:
            print(f"[MCP] Request error: {e}")
            return JiraTicketResult(
                success=False,
                error=f"Connection error: {str(e)}"
            )
        except Exception as e:
            print(f"[MCP] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return JiraTicketResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )

    def _extract_ticket_id(self, response_text: str) -> Optional[str]:
        """Extract ticket ID from MCP response text"""
        # Match patterns like ITSUPPORT-123, IT-456, PROJ-789, etc.
        match = re.search(r'([A-Z]+-\d+)', response_text)
        return match.group(1) if match else None

    def _build_ticket_url(self, ticket_id: Optional[str]) -> Optional[str]:
        """Build JIRA ticket URL from ticket ID"""
        if not ticket_id:
            return None
        return f"{self.jira_url}/browse/{ticket_id}"


# Singleton instance
mcp_client = MCPClient()
