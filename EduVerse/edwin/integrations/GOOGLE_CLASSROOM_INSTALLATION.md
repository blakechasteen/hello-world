# Google Classroom Installation Guide

**Author**: Agent D
**Date**: 2025-11-15
**Version**: 1.0

Complete step-by-step guide to integrate EdWIN with Google Classroom.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Create Google Cloud Project](#create-google-cloud-project)
3. [Enable APIs](#enable-apis)
4. [Create OAuth2 Credentials](#create-oauth2-credentials)
5. [Configure OAuth Consent Screen](#configure-oauth-consent-screen)
6. [Configure EdWIN](#configure-edwin)
7. [Setup Push Notifications (Optional)](#setup-push-notifications-optional)
8. [Test Connection](#test-connection)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Google Workspace for Education account
- Teacher with Google Classroom access
- EdWIN server running
- Google Cloud Console access

---

## Create Google Cloud Project

### Step 1: Access Google Cloud Console

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Sign in with your Google Workspace admin account

### Step 2: Create New Project

1. Click project dropdown (top left)
2. Click **New Project**
3. **Project Name**: `EdWIN Integration`
4. **Organization**: Your school domain
5. Click **Create**

### Step 3: Note Project Details

- **Project ID**: `edwin-integration-123456` (will be auto-generated)
- **Project Number**: `123456789012` (found in Project Info)

---

## Enable APIs

### Step 1: Enable Google Classroom API

1. In Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for "Google Classroom API"
3. Click **Enable**

### Step 2: Enable Google Drive API (for reports)

1. In API Library, search for "Google Drive API"
2. Click **Enable**

### Step 3: Enable Cloud Pub/Sub API (for webhooks)

1. In API Library, search for "Cloud Pub/Sub API"
2. Click **Enable**

---

## Create OAuth2 Credentials

### Step 1: Create OAuth 2.0 Client ID

1. Go to **APIs & Services** → **Credentials**
2. Click **+ Create Credentials** → **OAuth client ID**
3. If prompted, configure consent screen first (see next section)

### Step 2: Configure Application Type

**Application Type**: Web application

**Name**: EdWIN Google Classroom Integration

**Authorized JavaScript origins**:
```
http://localhost:8000
https://yourserver.com
```

**Authorized redirect URIs**:
```
http://localhost:8000/integrations/google/callback
https://yourserver.com/integrations/google/callback
```

### Step 3: Save Credentials

1. Click **Create**
2. **Download JSON** (or copy Client ID and Client Secret)
3. Store securely (you'll need these for EdWIN configuration)

Example credentials:
```json
{
  "web": {
    "client_id": "123456789012-abc123def456.apps.googleusercontent.com",
    "client_secret": "GOCSPX-abc123def456ghi789",
    "redirect_uris": ["http://localhost:8000/integrations/google/callback"]
  }
}
```

---

## Configure OAuth Consent Screen

### Step 1: Choose User Type

1. Go to **APIs & Services** → **OAuth consent screen**
2. **User Type**: Internal (for Workspace) or External (for testing)
3. Click **Create**

### Step 2: App Information

**App name**: EdWIN AI Tutor

**User support email**: your-email@yourschool.edu

**App logo**: (optional) Upload EdWIN logo

**Application home page**: https://yourserver.com

**Application privacy policy link**: https://yourserver.com/privacy

**Application terms of service link**: https://yourserver.com/terms

**Authorized domains**:
```
yourserver.com
yourschool.edu
```

**Developer contact information**: your-email@yourschool.edu

### Step 3: Configure Scopes

Click **Add or Remove Scopes** and add:

**Required Scopes**:
```
https://www.googleapis.com/auth/classroom.courses.readonly
https://www.googleapis.com/auth/classroom.rosters.readonly
https://www.googleapis.com/auth/classroom.coursework.students
https://www.googleapis.com/auth/classroom.student-submissions.students.readonly
https://www.googleapis.com/auth/classroom.profile.emails
```

**Optional Scopes** (for additional features):
```
https://www.googleapis.com/auth/classroom.announcements
https://www.googleapis.com/auth/drive.file
```

### Step 4: Test Users (if External)

If using External user type during development:
1. Add test teacher emails
2. Only these users can authorize the app until verification

### Step 5: Publish App (Production)

For production:
1. Submit for verification
2. Google will review (takes 1-2 weeks)
3. Once approved, all users can authorize

---

## Configure EdWIN

### Step 1: Install Dependencies

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### Step 2: Update Configuration

Edit `config/google_classroom.yaml`:

```yaml
google_classroom:
  client_id: "YOUR_CLIENT_ID.apps.googleusercontent.com"
  client_secret: "YOUR_CLIENT_SECRET"
  redirect_uri: "http://localhost:8000/integrations/google/callback"

  scopes:
    - "https://www.googleapis.com/auth/classroom.courses.readonly"
    - "https://www.googleapis.com/auth/classroom.rosters.readonly"
    - "https://www.googleapis.com/auth/classroom.coursework.students"
    - "https://www.googleapis.com/auth/classroom.student-submissions.students.readonly"
    - "https://www.googleapis.com/auth/classroom.profile.emails"
```

### Step 3: Start EdWIN Server

```bash
python -m uvicorn EduVerse.edwin.api:app --reload --port 8000
```

---

## Setup Push Notifications (Optional)

Push notifications enable real-time updates when students submit work.

### Step 1: Create Pub/Sub Topic

```bash
gcloud pubsub topics create classroom-notifications
```

Or in Cloud Console:
1. Go to **Pub/Sub** → **Topics**
2. Click **Create Topic**
3. **Topic ID**: `classroom-notifications`

### Step 2: Create Subscription

```bash
gcloud pubsub subscriptions create edwin-classroom-sub \
  --topic=classroom-notifications \
  --push-endpoint=https://yourserver.com/integrations/webhooks/google
```

Or in Cloud Console:
1. Click on topic → **Create Subscription**
2. **Subscription ID**: `edwin-classroom-sub`
3. **Delivery Type**: Push
4. **Endpoint URL**: `https://yourserver.com/integrations/webhooks/google`

### Step 3: Register Feed

In your application code:

```python
from googleapiclient.discovery import build

service = build('classroom', 'v1', credentials=credentials)

feed = {
    'feed': {
        'feedType': 'COURSE_WORK_CHANGES',
        'courseWorkChangesInfo': {
            'courseId': course_id
        }
    },
    'cloudPubsubTopic': {
        'topicName': 'projects/YOUR_PROJECT_ID/topics/classroom-notifications'
    }
}

result = service.registrations().create(body=feed).execute()
```

### Step 4: Update EdWIN Config

Edit `config/google_classroom.yaml`:

```yaml
push_notifications:
  enabled: true
  topic: "projects/YOUR_PROJECT_ID/topics/classroom-notifications"
  subscription: "projects/YOUR_PROJECT_ID/subscriptions/edwin-classroom-sub"
  endpoint: "https://yourserver.com/integrations/webhooks/google"
```

---

## Test Connection

### Step 1: Admin Dashboard Test

1. Open EdWIN Admin Dashboard: `http://localhost:8000/static/lms_admin.html`
2. Enter Google Client ID and Client Secret
3. Click "Connect Google Classroom"
4. Complete OAuth flow:
   - Choose teacher Google account
   - Grant permissions
   - Redirect back to EdWIN
5. Verify "Connected" status

### Step 2: Roster Sync Test

1. Click "Load Courses"
2. Verify Google Classroom courses appear
3. Select a course
4. Click "Sync Roster"
5. Verify students imported

### Step 3: Assignment Test

1. In Google Classroom, create a test assignment
2. In EdWIN, map assignment to objectives
3. Have a student complete the objectives
4. Verify grade syncs to Google Classroom

### Step 4: Materials Test

1. In EdWIN, create a lesson/report
2. Share to Google Classroom
3. Verify appears in Classroom stream
4. Verify accessible to students

---

## Troubleshooting

### Issue: OAuth fails with "access_denied"

**Solution**:
- Verify app is published (or user is test user)
- Check that user has Google Workspace for Education account
- Ensure all scopes are configured in consent screen

### Issue: "Insufficient Permission" error

**Solution**:
- Verify all required scopes are added to consent screen
- Re-authorize (revoke and re-grant permissions)
- Check that APIs are enabled in Cloud Console

### Issue: Can't see courses

**Solution**:
- Ensure user is a teacher (not just student)
- Verify `classroom.courses.readonly` scope is granted
- Check that Google Classroom API is enabled

### Issue: Grade passback fails

**Solution**:
- Verify `classroom.coursework.students` scope is granted
- Ensure assignment exists and hasn't been deleted
- Check that student is enrolled in course
- Verify coursework has `maxPoints` set

### Issue: Push notifications not received

**Solution**:
- Verify Cloud Pub/Sub API is enabled
- Check that subscription endpoint is accessible (HTTPS required)
- Ensure feed registration succeeded
- Test with Pub/Sub emulator first

### Issue: "redirect_uri_mismatch" error

**Solution**:
- Verify redirect URI exactly matches OAuth credentials
- Check for trailing slashes (must match exactly)
- Ensure HTTP vs HTTPS matches

---

## Production Checklist

Before deploying to production:

- [ ] Use HTTPS for all endpoints
- [ ] Publish OAuth consent screen (submit for verification)
- [ ] Store credentials in environment variables (not config files)
- [ ] Enable error logging (Cloud Logging)
- [ ] Set up monitoring (Cloud Monitoring)
- [ ] Test with multiple teacher accounts
- [ ] Verify grade sync with real assignments
- [ ] Train teachers on setup wizard
- [ ] Document support procedures
- [ ] Set up incident response plan

---

## Security Best Practices

1. **Token Storage**: Store tokens encrypted (EdWIN uses Fernet encryption)
2. **Token Rotation**: Refresh tokens expire after 6 months - handle gracefully
3. **Scope Minimization**: Only request scopes you actually use
4. **Audit Logging**: Log all API calls for compliance
5. **User Consent**: Always explain why you need each scope
6. **Data Retention**: Follow your district's data retention policies

---

## Google Classroom API Quotas

**Free Tier Limits**:
- 500 queries per 100 seconds per project
- 1500 queries per 100 seconds per user

**Best Practices**:
- Batch requests when possible
- Implement exponential backoff on rate limit errors
- Cache roster data (sync hourly, not real-time)
- Use push notifications instead of polling

---

## Support

For help:
- EdWIN Documentation: `/docs`
- Google Classroom API Documentation: https://developers.google.com/classroom
- Google Cloud Console Help: https://console.cloud.google.com/support

---

## Appendix: Common Google Classroom API Calls

**Get Courses**:
```python
service.courses().list().execute()
```

**Get Students**:
```python
service.courses().students().list(courseId=course_id).execute()
```

**Get Course Work**:
```python
service.courses().courseWork().list(courseId=course_id).execute()
```

**Submit Grade**:
```python
service.courses().courseWork().studentSubmissions().patch(
    courseId=course_id,
    courseWorkId=coursework_id,
    id=submission_id,
    updateMask='assignedGrade',
    body={'assignedGrade': grade}
).execute()
```

**Create Announcement**:
```python
service.courses().announcements().create(
    courseId=course_id,
    body={
        'text': 'Check out this EdWIN lesson!',
        'materials': [{'link': {'url': lesson_url}}]
    }
).execute()
```
