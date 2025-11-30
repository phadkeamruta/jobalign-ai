# OpenAI Rate Limit Error - Troubleshooting Guide

## What is `openai.RateLimitError`?

The `RateLimitError` occurs when you exceed OpenAI's API rate limits. This can happen due to:

1. **Too many requests** - Sending too many API calls in a short time period
2. **Quota exceeded** - You've reached your monthly token/request limit
3. **Account restrictions** - Your API key has limited quota or is restricted

## Solutions

### ✅ Solution 1: Automatic Retry Logic (Implemented)
The code now includes exponential backoff retry logic that:
- Automatically retries up to 3 times
- Waits 1, 2, and 4 seconds between retries
- Shows progress messages

**No action needed** - this is handled automatically.

### ✅ Solution 2: Upgrade Your OpenAI Account
Visit [https://platform.openai.com/account/billing/overview](https://platform.openai.com/account/billing/overview) to:
- Check your current usage
- Increase usage limits
- Add a payment method for higher quotas

### ✅ Solution 3: Use Batch Processing
Process resumes with delays between requests:

```python
from time import sleep

resumes = [resume1, resume2, resume3]
agent = ResumeParserAgent()

for resume in resumes:
    try:
        result = agent.parse_resume(resume)
        print(f"✅ Processed: {result.get('name')}")
        sleep(2)  # Wait 2 seconds between requests
    except RateLimitError:
        print("❌ Rate limit exceeded")
        break
```

### ✅ Solution 4: Monitor Your Usage
Check real-time usage:
```python
# View your OpenAI account dashboard
# https://platform.openai.com/account/usage/overview
```

### ✅ Solution 5: Switch to a Lower-Cost Model
If using `gpt-4.1`, consider using a cheaper model:

```python
# Instead of:
agent = ResumeParserAgent(model="gpt-4.1")

# Try:
agent = ResumeParserAgent(model="gpt-3.5-turbo")
```

## Error Messages Explained

| Error | Cause | Solution |
|-------|-------|----------|
| `RateLimitError: 429` | Too many requests in short time | Wait and retry |
| `RateLimitError: Quota exceeded` | Monthly quota used | Upgrade account |
| `RateLimitError: Account invalid` | API key issue | Check API key |

## Setting API Key

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-your-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"

# Or in Python before running:
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

## Best Practices

1. **Use appropriate delays** between consecutive API calls
2. **Batch similar requests** together
3. **Cache results** when possible
4. **Monitor usage** regularly
5. **Use cheaper models** for high-volume processing
6. **Implement error handling** with retries

## Support

- OpenAI Docs: https://platform.openai.com/docs
- API Status: https://status.openai.com
- Support: https://help.openai.com
