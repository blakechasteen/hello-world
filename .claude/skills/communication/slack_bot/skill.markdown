# Skill: Slack Bot

## Metadata

- **Name**: `slack_bot`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `communication`
- **Tags**: `slack, messaging, notifications, collaboration, chat`

## Description

**Short Description**:
Slack bot integration for proactive communication, notifications, and team collaboration.

**Detailed Description**:
The Slack Bot skill provides comprehensive Slack workspace integration capabilities. Send messages, post to channels, create threaded conversations, upload files, react to messages with emojis, and update user status. Supports rich formatting (markdown, blocks), mentions (@user, @channel), scheduled messages, and Slack workflow integration. Ideal for automated notifications, incident alerts, deployment updates, and conversational workflows.

## Required Capabilities

Check all capabilities this skill requires:

- [ ] File system access (read)
- [ ] File system access (write)
- [x] Code execution (bash)
- [x] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [x] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `slack_sdk` (official Slack Python SDK)
- Alternative: `slack-bolt` (for interactive apps)
- Slack API token (Bot User OAuth Token)
- Slack App with appropriate OAuth scopes

**HoloLoom Integration**: Integrates with notification pipelines, alerting systems, deployment workflows, and team communication automation.

## Input Schema

```json
{
  "operation": "string - send_message|post_to_channel|create_thread|upload_file|react|update_status",
  "parameters": {
    "token": "string (required) - Slack Bot OAuth token",
    "channel": "string (required for post_to_channel, upload_file) - Channel ID or name",
    "user": "string (required for send_message) - User ID",
    "text": "string (required for send_message, post_to_channel) - Message text",
    "thread_ts": "string (optional for create_thread) - Parent message timestamp",
    "file_path": "string (required for upload_file) - File to upload",
    "emoji": "string (required for react) - Emoji name (e.g., 'thumbsup', 'rocket')",
    "message_ts": "string (required for react) - Message timestamp to react to",
    "status_text": "string (required for update_status) - Status text",
    "status_emoji": "string (optional for update_status) - Status emoji",
    "blocks": "array (optional) - Slack Block Kit elements for rich formatting",
    "attachments": "array (optional) - Message attachments (legacy)",
    "unfurl_links": "boolean (optional) - Unfurl links (default: true)",
    "unfurl_media": "boolean (optional) - Unfurl media (default: true)"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Operation-specific result",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "channel": "string - Channel ID",
    "message_ts": "string - Message timestamp (unique ID)",
    "thread_ts": "string - Thread timestamp (for threaded messages)",
    "file_id": "string - Uploaded file ID (for upload_file)",
    "permalink": "string - Permanent link to message",
    "success": "boolean - Operation success"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Send Direct Message

**Input**:
```json
{
  "operation": "send_message",
  "parameters": {
    "token": "xoxb-your-bot-token",
    "user": "U123456789",
    "text": "Hello! Your deployment to production has completed successfully.",
    "blocks": [
      {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Deployment Status* :white_check_mark:\n\nEnvironment: `production`\nVersion: `v2.1.0`\nDuration: 3m 45s"}
      }
    ]
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "send_message",
    "channel": "D987654321",
    "message_ts": "1732450800.123456",
    "permalink": "https://workspace.slack.com/archives/D987654321/p1732450800123456",
    "success": true
  },
  "message": "Direct message sent to U123456789",
  "execution_time_ms": 320
}
```

**Explanation**: Sends a formatted direct message to a user with deployment status. Uses Slack Block Kit for rich formatting.

### Example 2: Post to Channel

**Input**:
```json
{
  "operation": "post_to_channel",
  "parameters": {
    "token": "xoxb-your-bot-token",
    "channel": "#deployments",
    "text": "@channel Production deployment complete!",
    "blocks": [
      {
        "type": "header",
        "text": {"type": "plain_text", "text": "Production Deployment"}
      },
      {
        "type": "section",
        "fields": [
          {"type": "mrkdwn", "text": "*Version:*\nv2.1.0"},
          {"type": "mrkdwn", "text": "*Status:*\n:white_check_mark: Success"}
        ]
      }
    ]
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "post_to_channel",
    "channel": "C123456789",
    "message_ts": "1732450900.234567",
    "permalink": "https://workspace.slack.com/archives/C123456789/p1732450900234567",
    "success": true
  },
  "message": "Message posted to #deployments",
  "execution_time_ms": 280
}
```

**Explanation**: Posts announcement to a public channel with @channel mention. Uses structured blocks for professional formatting.

### Example 3: Create Thread Reply

**Input**:
```json
{
  "operation": "create_thread",
  "parameters": {
    "token": "xoxb-your-bot-token",
    "channel": "C123456789",
    "thread_ts": "1732450900.234567",
    "text": "Deployment logs available here: https://logs.example.com/deploy-2025-11-24"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "create_thread",
    "channel": "C123456789",
    "message_ts": "1732451000.345678",
    "thread_ts": "1732450900.234567",
    "permalink": "https://workspace.slack.com/archives/C123456789/p1732451000345678?thread_ts=1732450900.234567",
    "success": true
  },
  "message": "Thread reply created",
  "execution_time_ms": 250
}
```

**Explanation**: Replies in a thread to keep conversations organized. Links to external logs for additional context.

### Example 4: Upload File

**Input**:
```json
{
  "operation": "upload_file",
  "parameters": {
    "token": "xoxb-your-bot-token",
    "channel": "#reports",
    "file_path": "reports/weekly_metrics.pdf",
    "title": "Weekly Metrics Report",
    "initial_comment": "Here's this week's performance report. Key highlights in the summary section."
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "upload_file",
    "channel": "C234567890",
    "file_id": "F12345ABCDE",
    "file_name": "weekly_metrics.pdf",
    "file_size_bytes": 145600,
    "permalink": "https://workspace.slack.com/files/U123456789/F12345ABCDE/weekly_metrics.pdf",
    "success": true
  },
  "message": "File uploaded: weekly_metrics.pdf",
  "execution_time_ms": 1850
}
```

**Explanation**: Uploads PDF report to a channel with descriptive comment. Useful for automated report distribution.

### Example 5: React to Message

**Input**:
```json
{
  "operation": "react",
  "parameters": {
    "token": "xoxb-your-bot-token",
    "channel": "C123456789",
    "message_ts": "1732450900.234567",
    "emoji": "rocket"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "react",
    "channel": "C123456789",
    "message_ts": "1732450900.234567",
    "emoji": "rocket",
    "success": true
  },
  "message": "Reacted with :rocket: to message",
  "execution_time_ms": 180
}
```

**Explanation**: Adds emoji reaction to acknowledge or celebrate messages. Simple visual feedback for team communication.

## Testing Checklist

- [x] **Functionality**: All 6 operations execute correctly
- [x] **Error Handling**: Graceful handling of invalid tokens, missing channels, API rate limits
- [x] **Security**: No token logging, secure credential handling
- [x] **Performance**: Operations complete within expected time (<2s)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Slack SDK documented
- [x] **Edge Cases**: Handles rate limits, large files, long messages
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom notification and alerting systems

## Security Considerations

**Potential Risks**:
- **Token Exposure**: Slack tokens in logs -> Never log tokens, use environment variables
- **Unauthorized Access**: Bot tokens have workspace-wide access -> Use principle of least privilege (minimal OAuth scopes)
- **Message Injection**: User input in messages -> Sanitize and validate all user-provided content

**Data Privacy**:
- [x] Does not log message content or tokens
- [x] Does not cache sensitive data
- [x] Does not send data to unauthorized endpoints

**Sandboxing**:
- [x] Operates within defined OAuth scopes
- [x] Does not access private channels without permission
- [x] Respects Slack workspace policies

## Performance Characteristics

- **Expected Latency**: 200-2000ms (0.2-2 seconds depending on operation)
- **Token Usage**: 50-500 tokens per execution
- **Resource Requirements**: Network connectivity, valid Slack token
- **Scalability**: Subject to Slack API rate limits (1+ request per second for messages)

**Operation-Specific Latencies**:
- `send_message`: 200-500ms (single API call)
- `post_to_channel`: 200-500ms (single API call)
- `create_thread`: 200-500ms (single API call)
- `upload_file`: 1000-5000ms (depends on file size)
- `react`: 150-300ms (lightweight operation)
- `update_status`: 200-400ms (single API call)

## License

MIT License

## Related Documentation

- **Slack API Docs**: [api.slack.com](https://api.slack.com)
- **Slack Block Kit**: [api.slack.com/block-kit](https://api.slack.com/block-kit)
- **slack_sdk Python**: [slack.dev/python-slack-sdk](https://slack.dev/python-slack-sdk)
- **HoloLoom Communication Skills**: [../README.md](../README.md)
