# Canvas LMS Installation Guide

**Author**: Agent D
**Date**: 2025-11-15
**Version**: 1.0

Complete step-by-step guide to integrate EdWIN with Canvas LMS.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Create Developer Key](#create-developer-key)
3. [Configure LTI 1.3](#configure-lti-13)
4. [Setup OAuth2](#setup-oauth2)
5. [Configure EdWIN](#configure-edwin)
6. [Setup Webhooks](#setup-webhooks)
7. [Test Connection](#test-connection)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Canvas LMS admin account
- EdWIN server running
- HTTPS domain for production (required for LTI 1.3)

---

## Create Developer Key

### Step 1: Access Developer Keys

1. Log into Canvas as an admin
2. Navigate to **Admin** → **Developer Keys**
3. Click **+ Developer Key** → **+ LTI Key**

### Step 2: Configure Key Settings

**Key Name**: `EdWIN AI Tutor`

**Redirect URIs**:
```
http://localhost:8000/integrations/canvas/callback
https://yourserver.com/integrations/canvas/callback
```

**Method**: Manual Entry

**Title**: EdWIN AI Tutor

**Description**: K-12 adaptive learning platform with AI tutoring

**Target Link URI**: `https://yourserver.com/edwin/launch`

**OpenID Connect Initiation Url**: `https://yourserver.com/edwin/lti/login`

**JWK Method**: Public JWK URL

**Public JWK URL**: `https://yourserver.com/edwin/lti/jwks`

### Step 3: Configure LTI Advantage Services

Enable the following services:

- ✅ **Can create and view assignment data in the gradebook** (Assignment and Grade Services)
- ✅ **Can view assignment data in the gradebook** (Assignment and Grade Services)
- ✅ **Can view submission data for assignments** (Assignment and Grade Services)
- ✅ **Can retrieve user data associated with the context** (Names and Role Provisioning)

### Step 4: Additional Settings

**Privacy Level**: Public (to get user email and name)

**Placements**:
- ✅ Course Navigation
- ✅ Assignment Selection

**Custom Fields** (optional):
```
course_id=$Canvas.course.id
user_id=$Canvas.user.id
assignment_id=$Canvas.assignment.id
```

### Step 5: Save and Enable

1. Click **Save**
2. Find your new key in the list
3. Click **ON** to enable it
4. **Copy the Client ID** (you'll need this later)

---

## Configure LTI 1.3

### Step 1: Get LTI Configuration

From the developer key details, copy:
- **Client ID**: `1234567890`
- **Deployment ID**: Find in Course Settings → Apps
- **Platform**: Your Canvas URL

### Step 2: Generate JWK Key Pair

EdWIN needs a public/private key pair for LTI 1.3.

```bash
# Generate private key
openssl genrsa -out edwin_lti_private.pem 2048

# Generate public key
openssl rsa -in edwin_lti_private.pem -outform PEM -pubout -out edwin_lti_public.pem

# Convert to JWK format (use online tool or library)
```

### Step 3: Configure EdWIN LTI

Edit `config/canvas.yaml`:

```yaml
lti:
  deployment_id: "YOUR_DEPLOYMENT_ID"
  platform: "https://yourschool.instructure.com"
  client_id: "YOUR_LTI_CLIENT_ID"
  auth_login_url: "https://yourschool.instructure.com/api/lti/authorize_redirect"
  auth_token_url: "https://yourschool.instructure.com/login/oauth2/token"
  key_set_url: "https://yourschool.instructure.com/api/lti/security/jwks"
  private_key_path: "./edwin_lti_private.pem"
  public_key_path: "./edwin_lti_public.pem"
```

---

## Setup OAuth2

### Step 1: Create OAuth2 Credentials

Canvas uses the same developer key for OAuth2.

From your developer key:
- **Client ID**: Already created
- **Client Secret**: Click "Show Key" to reveal

### Step 2: Configure Canvas OAuth2

Edit `config/canvas.yaml`:

```yaml
canvas:
  base_url: "https://yourschool.instructure.com"
  client_id: "YOUR_CLIENT_ID"
  client_secret: "YOUR_CLIENT_SECRET"
  redirect_uri: "http://localhost:8000/integrations/canvas/callback"
```

### Step 3: Required Scopes

EdWIN requires these OAuth2 scopes (already included in developer key):

```yaml
scopes:
  - "url:GET|/api/v1/courses"
  - "url:GET|/api/v1/courses/:course_id/enrollments"
  - "url:GET|/api/v1/courses/:course_id/assignments"
  - "url:POST|/api/v1/courses/:course_id/assignments"
  - "url:PUT|/api/v1/courses/:course_id/assignments/:id/submissions/:user_id"
```

---

## Configure EdWIN

### Step 1: Install Dependencies

```bash
pip install pylti1p3 cryptography aiohttp pyyaml
```

### Step 2: Update Configuration

Edit `config/canvas.yaml` with your Canvas details.

### Step 3: Start EdWIN Server

```bash
# Development
python -m uvicorn EduVerse.edwin.api:app --reload --port 8000

# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker EduVerse.edwin.api:app
```

---

## Setup Webhooks

### Step 1: Enable Live Events

1. In Canvas Admin → **Settings**
2. Find **Live Events** section
3. Enable Live Events

### Step 2: Configure Webhook Endpoint

**Endpoint URL**: `https://yourserver.com/integrations/webhooks/canvas`

**Events to Subscribe**:
- `enrollment_created`
- `enrollment_updated`
- `enrollment_deleted`
- `assignment_created`
- `assignment_updated`
- `grade_change`
- `submission_created`

### Step 3: Generate Webhook Secret

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add to `config/canvas.yaml`:

```yaml
webhooks:
  enabled: true
  secret: "YOUR_GENERATED_SECRET"
```

---

## Test Connection

### Step 1: Admin Dashboard Test

1. Open EdWIN Admin Dashboard: `http://localhost:8000/static/lms_admin.html`
2. Enter Canvas URL, Client ID, Client Secret
3. Click "Connect Canvas"
4. Complete OAuth flow
5. Verify "Connected" status

### Step 2: Roster Sync Test

1. Click "Load Courses"
2. Select a course
3. Click "Sync Roster"
4. Verify students imported

### Step 3: Assignment Test

1. In Canvas, create a test assignment
2. In EdWIN, map assignment to objectives
3. Have a student complete the objectives
4. Verify grade syncs to Canvas gradebook

### Step 4: LTI Launch Test

1. In Canvas course, click "EdWIN AI Tutor" in navigation
2. Verify single sign-on works
3. Verify student launches into EdWIN

---

## Troubleshooting

### Issue: OAuth fails with "invalid_client"

**Solution**: Double-check Client ID and Client Secret. Make sure developer key is enabled.

### Issue: LTI launch fails with 401

**Solution**:
- Verify deployment ID is correct
- Check that JWK public key is accessible at `/edwin/lti/jwks`
- Ensure private key file path is correct

### Issue: Grade passback doesn't work

**Solution**:
- Verify Assignment and Grade Services is enabled in developer key
- Check that assignment has `submission_types: ['external_tool']`
- Ensure student has submitted to assignment

### Issue: Roster sync returns empty

**Solution**:
- Verify Names and Role Provisioning is enabled
- Check OAuth scopes include enrollment endpoints
- Ensure you're using correct course ID

### Issue: Webhooks not received

**Solution**:
- Verify Live Events is enabled in Canvas
- Check webhook endpoint is accessible (use ngrok for local testing)
- Ensure webhook secret matches configuration
- Check Canvas webhook logs in Admin → Settings → Live Events

---

## Production Checklist

Before deploying to production:

- [ ] Use HTTPS for all endpoints
- [ ] Generate strong OAuth2 client secret
- [ ] Generate strong webhook secret
- [ ] Store secrets in environment variables (not config files)
- [ ] Enable webhook signature verification
- [ ] Set up error monitoring (Sentry, etc.)
- [ ] Configure backup/archive for dropped students
- [ ] Test grade sync with real assignments
- [ ] Train teachers on assignment mapping
- [ ] Document incident response procedures

---

## Support

For help:
- EdWIN Documentation: `/docs`
- Canvas LTI Documentation: https://canvas.instructure.com/doc/api/file.lti_dev_key_config.html
- Canvas API Documentation: https://canvas.instructure.com/doc/api/

---

## Appendix: Common Canvas API Endpoints

**Get Courses**:
```
GET /api/v1/courses
```

**Get Course Enrollments**:
```
GET /api/v1/courses/:course_id/enrollments
```

**Get Course Assignments**:
```
GET /api/v1/courses/:course_id/assignments
```

**Submit Grade**:
```
PUT /api/v1/courses/:course_id/assignments/:assignment_id/submissions/:user_id
```

**Get User Profile**:
```
GET /api/v1/users/self/profile
```
