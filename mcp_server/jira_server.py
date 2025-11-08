#!/usr/bin/env python3
"""
Jira MCP Server - Connect Claude Desktop to Jira

Provides tools for:
- Creating issues
- Updating issues
- Adding comments
- Searching issues
- Getting issue details
- Transitioning issues (workflow states)
- Adding/removing labels
- Assigning issues
"""

import os
import asyncio
import json
from typing import Any, Dict, List, Optional
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Jira API client
try:
    from jira import JIRA
    JIRA_AVAILABLE = True
except ImportError:
    JIRA_AVAILABLE = False
    print("Warning: jira library not installed. Install with: pip install jira")

# Initialize MCP server
app = Server("jira")

# Global Jira client (initialized on first use)
jira_client: Optional[JIRA] = None


def get_jira_client() -> JIRA:
    """Get or create Jira client with credentials from environment."""
    global jira_client

    if jira_client is not None:
        return jira_client

    if not JIRA_AVAILABLE:
        raise RuntimeError("jira library not installed. Run: pip install jira")

    # Get credentials from environment
    jira_url = os.getenv("JIRA_URL")
    jira_email = os.getenv("JIRA_EMAIL")
    jira_api_token = os.getenv("JIRA_API_TOKEN")

    if not all([jira_url, jira_email, jira_api_token]):
        raise ValueError(
            "Missing Jira credentials. Set environment variables:\n"
            "- JIRA_URL (e.g., https://yourcompany.atlassian.net)\n"
            "- JIRA_EMAIL (your Atlassian account email)\n"
            "- JIRA_API_TOKEN (generate at https://id.atlassian.com/manage-profile/security/api-tokens)"
        )

    # Create Jira client
    jira_client = JIRA(
        server=jira_url,
        basic_auth=(jira_email, jira_api_token)
    )

    return jira_client


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available Jira tools."""
    return [
        types.Tool(
            name="create_issue",
            description="Create a new Jira issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project key (e.g., 'PROJ', 'DEV')"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Issue title/summary"
                    },
                    "description": {
                        "type": "string",
                        "description": "Issue description (supports Jira markdown)"
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "Issue type (e.g., 'Bug', 'Task', 'Story')",
                        "default": "Task"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority (e.g., 'High', 'Medium', 'Low')",
                        "default": "Medium"
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to add to the issue"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Assignee username or email"
                    }
                },
                "required": ["project", "summary", "description"]
            }
        ),
        types.Tool(
            name="get_issue",
            description="Get details of a Jira issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Issue key (e.g., 'PROJ-123')"
                    }
                },
                "required": ["issue_key"]
            }
        ),
        types.Tool(
            name="update_issue",
            description="Update a Jira issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Issue key (e.g., 'PROJ-123')"
                    },
                    "summary": {
                        "type": "string",
                        "description": "New summary/title"
                    },
                    "description": {
                        "type": "string",
                        "description": "New description"
                    },
                    "priority": {
                        "type": "string",
                        "description": "New priority"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "New assignee username or email"
                    }
                },
                "required": ["issue_key"]
            }
        ),
        types.Tool(
            name="add_comment",
            description="Add a comment to a Jira issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Issue key (e.g., 'PROJ-123')"
                    },
                    "comment": {
                        "type": "string",
                        "description": "Comment text (supports Jira markdown)"
                    }
                },
                "required": ["issue_key", "comment"]
            }
        ),
        types.Tool(
            name="search_issues",
            description="Search for Jira issues using JQL",
            inputSchema={
                "type": "object",
                "properties": {
                    "jql": {
                        "type": "string",
                        "description": "JQL query (e.g., 'project = PROJ AND status = Open')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 50
                    }
                },
                "required": ["jql"]
            }
        ),
        types.Tool(
            name="transition_issue",
            description="Transition an issue to a different status (e.g., 'In Progress', 'Done')",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Issue key (e.g., 'PROJ-123')"
                    },
                    "transition": {
                        "type": "string",
                        "description": "Transition name (e.g., 'Start Progress', 'Done', 'In Review')"
                    }
                },
                "required": ["issue_key", "transition"]
            }
        ),
        types.Tool(
            name="add_labels",
            description="Add labels to a Jira issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Issue key (e.g., 'PROJ-123')"
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to add"
                    }
                },
                "required": ["issue_key", "labels"]
            }
        ),
        types.Tool(
            name="list_projects",
            description="List all accessible Jira projects",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="get_transitions",
            description="Get available transitions for an issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Issue key (e.g., 'PROJ-123')"
                    }
                },
                "required": ["issue_key"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[types.TextContent]:
    """Handle tool calls from Claude."""
    try:
        if name == "create_issue":
            return await tool_create_issue(arguments)
        elif name == "get_issue":
            return await tool_get_issue(arguments)
        elif name == "update_issue":
            return await tool_update_issue(arguments)
        elif name == "add_comment":
            return await tool_add_comment(arguments)
        elif name == "search_issues":
            return await tool_search_issues(arguments)
        elif name == "transition_issue":
            return await tool_transition_issue(arguments)
        elif name == "add_labels":
            return await tool_add_labels(arguments)
        elif name == "list_projects":
            return await tool_list_projects(arguments)
        elif name == "get_transitions":
            return await tool_get_transitions(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def tool_create_issue(args: Dict[str, Any]) -> list[types.TextContent]:
    """Create a new Jira issue."""
    jira = get_jira_client()

    issue_dict = {
        'project': {'key': args['project']},
        'summary': args['summary'],
        'description': args['description'],
        'issuetype': {'name': args.get('issue_type', 'Task')},
    }

    # Add optional fields
    if 'priority' in args:
        issue_dict['priority'] = {'name': args['priority']}

    if 'labels' in args:
        issue_dict['labels'] = args['labels']

    if 'assignee' in args:
        issue_dict['assignee'] = {'name': args['assignee']}

    # Create the issue
    new_issue = jira.create_issue(fields=issue_dict)

    result = f"""✅ Issue Created Successfully

**Issue Key:** {new_issue.key}
**URL:** {jira.client_info()}/browse/{new_issue.key}

**Summary:** {new_issue.fields.summary}
**Type:** {new_issue.fields.issuetype.name}
**Status:** {new_issue.fields.status.name}
**Priority:** {new_issue.fields.priority.name if hasattr(new_issue.fields, 'priority') else 'N/A'}
"""

    if hasattr(new_issue.fields, 'assignee') and new_issue.fields.assignee:
        result += f"**Assignee:** {new_issue.fields.assignee.displayName}\n"

    if hasattr(new_issue.fields, 'labels') and new_issue.fields.labels:
        result += f"**Labels:** {', '.join(new_issue.fields.labels)}\n"

    return [types.TextContent(type="text", text=result)]


async def tool_get_issue(args: Dict[str, Any]) -> list[types.TextContent]:
    """Get details of a Jira issue."""
    jira = get_jira_client()
    issue = jira.issue(args['issue_key'])

    result = f"""📋 Issue Details: {issue.key}

**Summary:** {issue.fields.summary}
**Type:** {issue.fields.issuetype.name}
**Status:** {issue.fields.status.name}
**Priority:** {issue.fields.priority.name if hasattr(issue.fields, 'priority') else 'N/A'}
**Project:** {issue.fields.project.name} ({issue.fields.project.key})
"""

    if hasattr(issue.fields, 'assignee') and issue.fields.assignee:
        result += f"**Assignee:** {issue.fields.assignee.displayName}\n"
    else:
        result += "**Assignee:** Unassigned\n"

    if hasattr(issue.fields, 'reporter') and issue.fields.reporter:
        result += f"**Reporter:** {issue.fields.reporter.displayName}\n"

    if hasattr(issue.fields, 'labels') and issue.fields.labels:
        result += f"**Labels:** {', '.join(issue.fields.labels)}\n"

    if hasattr(issue.fields, 'created'):
        result += f"**Created:** {issue.fields.created}\n"

    if hasattr(issue.fields, 'updated'):
        result += f"**Updated:** {issue.fields.updated}\n"

    result += f"\n**Description:**\n{issue.fields.description or '(No description)'}\n"

    # Add comments if any
    if hasattr(issue.fields, 'comment') and issue.fields.comment.comments:
        result += f"\n**Comments ({len(issue.fields.comment.comments)}):**\n"
        for comment in issue.fields.comment.comments[-5:]:  # Last 5 comments
            result += f"- {comment.author.displayName} ({comment.created}): {comment.body[:100]}...\n"

    return [types.TextContent(type="text", text=result)]


async def tool_update_issue(args: Dict[str, Any]) -> list[types.TextContent]:
    """Update a Jira issue."""
    jira = get_jira_client()
    issue = jira.issue(args['issue_key'])

    update_fields = {}

    if 'summary' in args:
        update_fields['summary'] = args['summary']

    if 'description' in args:
        update_fields['description'] = args['description']

    if 'priority' in args:
        update_fields['priority'] = {'name': args['priority']}

    if 'assignee' in args:
        update_fields['assignee'] = {'name': args['assignee']}

    issue.update(fields=update_fields)

    result = f"""✅ Issue Updated: {issue.key}

Updated fields:
"""
    for field, value in update_fields.items():
        if isinstance(value, dict):
            value = value.get('name', str(value))
        result += f"- **{field.capitalize()}:** {value}\n"

    return [types.TextContent(type="text", text=result)]


async def tool_add_comment(args: Dict[str, Any]) -> list[types.TextContent]:
    """Add a comment to a Jira issue."""
    jira = get_jira_client()
    comment = jira.add_comment(args['issue_key'], args['comment'])

    result = f"""💬 Comment Added to {args['issue_key']}

**Comment:** {comment.body}
**Author:** {comment.author.displayName}
**Created:** {comment.created}
"""

    return [types.TextContent(type="text", text=result)]


async def tool_search_issues(args: Dict[str, Any]) -> list[types.TextContent]:
    """Search for Jira issues using JQL."""
    jira = get_jira_client()
    max_results = args.get('max_results', 50)

    issues = jira.search_issues(args['jql'], maxResults=max_results)

    result = f"""🔍 Search Results ({len(issues)} issues found)

**JQL:** {args['jql']}

"""

    for issue in issues:
        assignee = issue.fields.assignee.displayName if hasattr(issue.fields, 'assignee') and issue.fields.assignee else 'Unassigned'
        result += f"""**{issue.key}** - {issue.fields.summary}
  Type: {issue.fields.issuetype.name} | Status: {issue.fields.status.name} | Assignee: {assignee}

"""

    return [types.TextContent(type="text", text=result)]


async def tool_transition_issue(args: Dict[str, Any]) -> list[types.TextContent]:
    """Transition an issue to a different status."""
    jira = get_jira_client()
    issue = jira.issue(args['issue_key'])

    # Find the transition ID by name
    transitions = jira.transitions(issue)
    transition_id = None

    for trans in transitions:
        if trans['name'].lower() == args['transition'].lower():
            transition_id = trans['id']
            break

    if not transition_id:
        available = ', '.join([t['name'] for t in transitions])
        return [types.TextContent(
            type="text",
            text=f"❌ Transition '{args['transition']}' not found.\n\nAvailable transitions: {available}"
        )]

    # Perform the transition
    jira.transition_issue(issue, transition_id)

    # Refresh issue to get new status
    issue = jira.issue(args['issue_key'])

    result = f"""✅ Issue Transitioned: {issue.key}

**New Status:** {issue.fields.status.name}
**Transition:** {args['transition']}
"""

    return [types.TextContent(type="text", text=result)]


async def tool_add_labels(args: Dict[str, Any]) -> list[types.TextContent]:
    """Add labels to a Jira issue."""
    jira = get_jira_client()
    issue = jira.issue(args['issue_key'])

    # Get existing labels
    existing_labels = issue.fields.labels or []
    new_labels = list(set(existing_labels + args['labels']))

    issue.update(fields={'labels': new_labels})

    result = f"""🏷️ Labels Updated: {issue.key}

**Labels:** {', '.join(new_labels)}
**Added:** {', '.join(args['labels'])}
"""

    return [types.TextContent(type="text", text=result)]


async def tool_list_projects(args: Dict[str, Any]) -> list[types.TextContent]:
    """List all accessible Jira projects."""
    jira = get_jira_client()
    projects = jira.projects()

    result = f"""📁 Accessible Projects ({len(projects)})

"""

    for project in projects:
        result += f"**{project.key}** - {project.name}\n"
        if hasattr(project, 'lead'):
            result += f"  Lead: {project.lead.displayName}\n"
        result += "\n"

    return [types.TextContent(type="text", text=result)]


async def tool_get_transitions(args: Dict[str, Any]) -> list[types.TextContent]:
    """Get available transitions for an issue."""
    jira = get_jira_client()
    issue = jira.issue(args['issue_key'])
    transitions = jira.transitions(issue)

    result = f"""🔄 Available Transitions for {issue.key}

**Current Status:** {issue.fields.status.name}

**Available Transitions:**
"""

    for trans in transitions:
        result += f"- {trans['name']}\n"

    return [types.TextContent(type="text", text=result)]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="jira",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())