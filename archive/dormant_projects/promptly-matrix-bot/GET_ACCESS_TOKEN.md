# How to Get Your Matrix Access Token from Element

If password login isn't working, you can use your existing Element session's access token instead.

## Steps to Get Access Token from Element Web

1. **Log into Element Web**: https://app.element.io

2. **Open Developer Tools**: Press `F12` or `Ctrl+Shift+I`

3. **Go to Application Tab** (in Chrome/Edge) or **Storage Tab** (in Firefox)

4. **Navigate to Local Storage** → `https://app.element.io`

5. **Find the access token**:
   - Look for a key that contains `mx_access_token` or similar
   - Or search for keys starting with `syt_` (Synapse access token format)

6. **Copy the token value** (it will be a long string)

## Alternative Method: Use Element Developer Console

1. Log into Element Web

2. Open Developer Tools Console (`F12` → Console tab)

3. Run this JavaScript:
   ```javascript
   JSON.parse(localStorage.getItem('mx_access_token'))
   ```

4. Copy the displayed token

## Use the Token with the Bot

Once you have the access token, you can use it instead of password:

### Option 1: Environment Variable
```bash
export MATRIX_ACCESS_TOKEN='your_access_token_here'
export MATRIX_USER_ID='@promptlybot:matrix.org'
python run_bot.py
```

### Option 2: Update config
Add to `config/matrix_config.json`:
```json
{
  "homeserver": "https://matrix.org",
  "user": "@promptlybot:matrix.org",
  "access_token": "your_access_token_here",
  "auto_join_rooms": true
}
```

**Note**: Access tokens are sensitive! Don't share them or commit them to git.