// Run this in Element Web's console (F12 -> Console)
// This will find your Matrix access token

console.log("=== Searching for Matrix Access Token ===\n");

// Method 1: Check all localStorage keys
console.log("All localStorage keys:");
for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key.includes('mx_') || key.includes('matrix') || key.includes('access')) {
        console.log(`  - ${key}`);
    }
}

console.log("\n=== Trying different token locations ===\n");

// Method 2: Try different common keys
const possibleKeys = [
    'mx_access_token',
    'mx_access_token_object',
    'mx_hs_url',
    'mx_user_id',
    'mx_device_id'
];

possibleKeys.forEach(key => {
    const value = localStorage.getItem(key);
    if (value) {
        console.log(`${key}:`);
        try {
            const parsed = JSON.parse(value);
            console.log(parsed);
        } catch {
            console.log(value);
        }
        console.log('');
    }
});

// Method 3: Search for anything starting with "syt_" (Synapse token prefix)
console.log("\n=== Searching for tokens (syt_ prefix) ===\n");
for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    const value = localStorage.getItem(key);
    if (value && value.includes('syt_')) {
        console.log(`Found token in key: ${key}`);
        console.log(`Token: ${value.substring(0, 50)}...`);
    }
}

console.log("\n=== Instructions ===");
console.log("Copy the access token value and add it to config/matrix_config.json");
console.log('Format: "access_token": "syt_YOUR_TOKEN_HERE"');
