"""
Bad code with security issues - focuses on SQL injection, XSS, and unsafe practices.

Created: 2025-11-22
Category: Security vulnerabilities
Expected issues: 8-10
"""


def get_user_by_id(user_id):
    """
    ISSUE 1: SQL Injection via string concatenation
    """
    query = "SELECT * FROM users WHERE id = " + user_id
    # Execute query with unsanitized input
    return database.execute(query)


def search_users(search_term):
    """
    ISSUE 2: SQL Injection via string formatting
    """
    query = f"SELECT * FROM users WHERE name LIKE '%{search_term}%'"
    return database.execute(query)


def update_user_profile(user_id, profile_data):
    """
    ISSUE 3: SQL Injection with multiple parameters
    """
    query = "UPDATE users SET "
    for key, value in profile_data.items():
        query += f"{key} = '{value}', "  # No parameterization!
    query = query.rstrip(", ") + f" WHERE id = {user_id}"
    return database.execute(query)


def render_user_bio(user_id):
    """
    ISSUE 4: XSS vulnerability - unescaped HTML
    """
    user = get_user(user_id)
    html = f"<div class='bio'>{user.bio}</div>"  # No escaping!
    return html


def create_comment(comment_text):
    """
    ISSUE 5: XSS - innerHTML equivalent
    """
    return f"<p id='comment'>{comment_text}</p>"  # Direct HTML injection


def execute_command(user_input):
    """
    ISSUE 6: Command injection
    """
    import subprocess
    # User input directly in shell command!
    result = subprocess.run(f"ls {user_input}", shell=True)
    return result


def load_config(config_path):
    """
    ISSUE 7: Path traversal vulnerability
    """
    import os
    # No validation - user could pass "../../etc/passwd"
    full_path = os.path.join("/configs", config_path)
    with open(full_path) as f:
        return f.read()


def upload_file(filename, content):
    """
    ISSUE 8: Arbitrary file write
    """
    # No validation of filename - could overwrite system files!
    with open(f"/uploads/{filename}", "w") as f:
        f.write(content)


def deserialize_user_data(json_string):
    """
    ISSUE 9: Unsafe deserialization (Python pickle)
    """
    import pickle
    # pickle can execute arbitrary code!
    return pickle.loads(json_string)


def verify_api_request(api_key):
    """
    ISSUE 10: Hardcoded API key check
    """
    if api_key == "hardcoded_secret_key_12345":  # CRITICAL: Don't do this!
        return True
    return False


def connect_to_database():
    """
    ISSUE 11: Hardcoded database credentials
    """
    host = "db.example.com"
    username = "admin"
    password = "password123"  # CRITICAL!
    db_connection(host, username, password)


def log_sensitive_data(user_email, user_password):
    """
    ISSUE 12: Logging sensitive information
    """
    logger.info(f"User login: {user_email}, password: {user_password}")
    # Passwords should never be logged!


def create_reset_token(user_id):
    """
    ISSUE 13: Weak token generation
    """
    import random
    import string
    # Predictable token - not cryptographically secure!
    token = ''.join(random.choice(string.ascii_letters) for _ in range(10))
    return token


def validate_email(email):
    """
    ISSUE 14: Insufficient input validation
    """
    # Too simple, doesn't actually validate email format properly
    if "@" in email:
        return True
    return False


def process_user_json(json_data):
    """
    ISSUE 15: eval() with user input (EXTREMELY DANGEROUS)
    """
    result = eval(json_data)  # CRITICAL: eval() can execute arbitrary code!
    return result
