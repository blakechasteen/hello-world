"use strict";
// Large TypeScript File for Testing Context Optimization
// This file has 1500+ lines to demonstrate automatic context reduction
Object.defineProperty(exports, "__esModule", { value: true });
exports.Timer = exports.Counter = exports.TestRunner = exports.Logger = exports.FileUploader = exports.WebSocketClient = exports.Router = exports.Store = exports.EventEmitter = exports.Cache = exports.ApiClient = exports.Database = exports.AuthService = exports.UserService = void 0;
exports.expect = expect;
exports.debounce = debounce;
exports.throttle = throttle;
exports.deepClone = deepClone;
exports.isEqual = isEqual;
exports.formatDate = formatDate;
exports.chunk = chunk;
exports.flatten = flatten;
exports.unique = unique;
exports.groupBy = groupBy;
exports.sortBy = sortBy;
exports.capitalize = capitalize;
exports.camelCase = camelCase;
exports.snakeCase = snakeCase;
exports.kebabCase = kebabCase;
exports.truncate = truncate;
exports.slugify = slugify;
exports.clamp = clamp;
exports.randomInt = randomInt;
exports.randomFloat = randomFloat;
exports.round = round;
exports.percentage = percentage;
exports.calculateImportantValue = calculateImportantValue;
class UserService {
    constructor() {
        this.users = new Map();
    }
    async createUser(data) {
        const user = {
            id: this.generateId(),
            ...data,
            createdAt: new Date(),
            updatedAt: new Date()
        };
        this.users.set(user.id, user);
        return user;
    }
    async getUser(id) {
        return this.users.get(id);
    }
    async updateUser(id, updates) {
        const user = this.users.get(id);
        if (!user)
            return undefined;
        const updated = {
            ...user,
            ...updates,
            updatedAt: new Date()
        };
        this.users.set(id, updated);
        return updated;
    }
    async deleteUser(id) {
        return this.users.delete(id);
    }
    generateId() {
        return Math.random().toString(36).substr(2, 9);
    }
}
exports.UserService = UserService;
class AuthService {
    constructor() {
        this.tokens = new Map();
        this.TOKEN_EXPIRY_HOURS = 24;
    }
    async login(credentials) {
        // Simulate authentication
        const userId = await this.validateCredentials(credentials);
        return this.createToken(userId);
    }
    async register(data) {
        // Simulate registration
        const userId = await this.createAccount(data);
        return this.createToken(userId);
    }
    async logout(token) {
        this.tokens.delete(token);
    }
    async validateToken(token) {
        const authToken = this.tokens.get(token);
        if (!authToken)
            return undefined;
        if (authToken.expiresAt < new Date()) {
            this.tokens.delete(token);
            return undefined;
        }
        return authToken;
    }
    async validateCredentials(credentials) {
        // Simulate credential validation
        return 'user-123';
    }
    async createAccount(data) {
        // Simulate account creation
        return 'user-' + Math.random().toString(36).substr(2, 9);
    }
    createToken(userId) {
        const token = Math.random().toString(36).substr(2, 16);
        const expiresAt = new Date();
        expiresAt.setHours(expiresAt.getHours() + this.TOKEN_EXPIRY_HOURS);
        const authToken = { token, expiresAt, userId };
        this.tokens.set(token, authToken);
        return authToken;
    }
}
exports.AuthService = AuthService;
class Validator {
    constructor() {
        this.rules = new Map();
    }
    addRule(field, rule) {
        const existing = this.rules.get(field) || [];
        this.rules.set(field, [...existing, rule]);
        return this;
    }
    validate(data) {
        const errors = [];
        for (const [field, rules] of this.rules.entries()) {
            const value = data[field];
            for (const rule of rules) {
                if (!rule(value)) {
                    errors.push({
                        field: String(field),
                        message: `Validation failed for ${String(field)}`
                    });
                }
            }
        }
        return errors;
    }
}
// Common validation rules
const required = (value) => value !== undefined && value !== null && value !== '';
const email = (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
const minLength = (min) => (value) => value.length >= min;
const maxLength = (max) => (value) => value.length <= max;
class Database {
    constructor() {
        this.data = new Map();
    }
    async insert(id, item) {
        this.data.set(id, item);
        return item;
    }
    async findById(id) {
        return this.data.get(id);
    }
    async findAll(options = {}) {
        const { page = 1, pageSize = 10, sortBy, sortOrder = 'asc' } = options;
        let items = Array.from(this.data.values());
        // Apply filters
        if (options.filters) {
            items = items.filter(item => {
                return Object.entries(options.filters).every(([key, value]) => {
                    return item[key] === value;
                });
            });
        }
        // Apply sorting
        if (sortBy) {
            items.sort((a, b) => {
                const aVal = a[sortBy];
                const bVal = b[sortBy];
                const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
                return sortOrder === 'asc' ? comparison : -comparison;
            });
        }
        // Apply pagination
        const start = (page - 1) * pageSize;
        const paginatedItems = items.slice(start, start + pageSize);
        return {
            data: paginatedItems,
            total: items.length,
            page,
            pageSize
        };
    }
    async update(id, updates) {
        const item = this.data.get(id);
        if (!item)
            return undefined;
        const updated = { ...item, ...updates };
        this.data.set(id, updated);
        return updated;
    }
    async delete(id) {
        return this.data.delete(id);
    }
}
exports.Database = Database;
class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    async request(endpoint, options = {}) {
        const { method = 'GET', headers = {}, body } = options;
        try {
            const url = `${this.baseUrl}${endpoint}`;
            const response = await fetch(url, {
                method,
                headers: {
                    'Content-Type': 'application/json',
                    ...headers
                },
                body: body ? JSON.stringify(body) : undefined
            });
            if (!response.ok) {
                return {
                    success: false,
                    error: `HTTP ${response.status}: ${response.statusText}`
                };
            }
            const data = await response.json();
            return {
                success: true,
                data
            };
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error'
            };
        }
    }
    async get(endpoint, headers) {
        return this.request(endpoint, { method: 'GET', headers });
    }
    async post(endpoint, body, headers) {
        return this.request(endpoint, { method: 'POST', body, headers });
    }
    async put(endpoint, body, headers) {
        return this.request(endpoint, { method: 'PUT', body, headers });
    }
    async delete(endpoint, headers) {
        return this.request(endpoint, { method: 'DELETE', headers });
    }
}
exports.ApiClient = ApiClient;
class Cache {
    constructor(defaultTTL = 3600000) {
        this.store = new Map();
        this.defaultTTL = defaultTTL;
    }
    set(key, value, ttl) {
        const expiresAt = Date.now() + (ttl || this.defaultTTL);
        this.store.set(key, { value, expiresAt });
    }
    get(key) {
        const entry = this.store.get(key);
        if (!entry)
            return undefined;
        if (Date.now() > entry.expiresAt) {
            this.store.delete(key);
            return undefined;
        }
        return entry.value;
    }
    has(key) {
        const value = this.get(key);
        return value !== undefined;
    }
    delete(key) {
        return this.store.delete(key);
    }
    clear() {
        this.store.clear();
    }
    cleanup() {
        const now = Date.now();
        let removed = 0;
        for (const [key, entry] of this.store.entries()) {
            if (now > entry.expiresAt) {
                this.store.delete(key);
                removed++;
            }
        }
        return removed;
    }
}
exports.Cache = Cache;
class EventEmitter {
    constructor() {
        this.listeners = new Map();
    }
    on(event, handler) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(handler);
        // Return unsubscribe function
        return () => this.off(event, handler);
    }
    off(event, handler) {
        const handlers = this.listeners.get(event);
        if (handlers) {
            handlers.delete(handler);
        }
    }
    emit(event, data) {
        const handlers = this.listeners.get(event);
        if (handlers) {
            handlers.forEach(handler => handler(data));
        }
    }
    removeAllListeners(event) {
        if (event) {
            this.listeners.delete(event);
        }
        else {
            this.listeners.clear();
        }
    }
}
exports.EventEmitter = EventEmitter;
class Store {
    constructor(initialState) {
        this.subscribers = new Set();
        this.state = initialState;
    }
    getState() {
        return this.state;
    }
    setState(updater) {
        if (typeof updater === 'function') {
            this.state = updater(this.state);
        }
        else {
            this.state = { ...this.state, ...updater };
        }
        this.notify();
    }
    subscribe(subscriber) {
        this.subscribers.add(subscriber);
        subscriber(this.state); // Call immediately with current state
        return () => {
            this.subscribers.delete(subscriber);
        };
    }
    notify() {
        this.subscribers.forEach(subscriber => subscriber(this.state));
    }
}
exports.Store = Store;
class Router {
    constructor() {
        this.routes = new Map();
    }
    get(path, handler) {
        this.addRoute('GET', path, handler);
    }
    post(path, handler) {
        this.addRoute('POST', path, handler);
    }
    put(path, handler) {
        this.addRoute('PUT', path, handler);
    }
    delete(path, handler) {
        this.addRoute('DELETE', path, handler);
    }
    addRoute(method, path, handler) {
        if (!this.routes.has(method)) {
            this.routes.set(method, new Map());
        }
        this.routes.get(method).set(path, handler);
    }
    async handle(req, res) {
        const methodRoutes = this.routes.get(req.method);
        if (!methodRoutes) {
            res.status(404).json({ error: 'Method not found' });
            return;
        }
        const handler = methodRoutes.get(req.url);
        if (!handler) {
            res.status(404).json({ error: 'Route not found' });
            return;
        }
        try {
            await handler(req, res);
        }
        catch (error) {
            res.status(500).json({
                error: error instanceof Error ? error.message : 'Internal server error'
            });
        }
    }
}
exports.Router = Router;
class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.messageHandlers = new Set();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }
    connect() {
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.url);
                this.ws.onopen = () => {
                    this.reconnectAttempts = 0;
                    resolve();
                };
                this.ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    this.messageHandlers.forEach(handler => handler(message));
                };
                this.ws.onclose = () => {
                    this.handleDisconnect();
                };
                this.ws.onerror = (error) => {
                    reject(error);
                };
            }
            catch (error) {
                reject(error);
            }
        });
    }
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }
    onMessage(handler) {
        this.messageHandlers.add(handler);
        return () => this.messageHandlers.delete(handler);
    }
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
    async handleDisconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            await this.sleep(this.reconnectDelay * this.reconnectAttempts);
            await this.connect();
        }
    }
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.WebSocketClient = WebSocketClient;
class FileUploader {
    constructor() {
        this.defaultMaxSize = 10 * 1024 * 1024; // 10MB
        this.defaultAllowedTypes = [
            'image/jpeg',
            'image/png',
            'image/gif',
            'application/pdf'
        ];
    }
    async upload(file, options = {}) {
        const { maxSize = this.defaultMaxSize, allowedTypes = this.defaultAllowedTypes, onProgress } = options;
        // Validate file size
        if (file.size > maxSize) {
            return {
                success: false,
                error: `File size exceeds maximum of ${maxSize} bytes`
            };
        }
        // Validate file type
        if (!allowedTypes.includes(file.type)) {
            return {
                success: false,
                error: `File type ${file.type} is not allowed`
            };
        }
        // Simulate upload with progress
        return new Promise((resolve) => {
            let progress = 0;
            const interval = setInterval(() => {
                progress += 10;
                if (onProgress) {
                    onProgress(progress);
                }
                if (progress >= 100) {
                    clearInterval(interval);
                    resolve({
                        success: true,
                        url: `https://example.com/uploads/${file.name}`
                    });
                }
            }, 100);
        });
    }
}
exports.FileUploader = FileUploader;
// ============================================================================
// Logging System
// ============================================================================
var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["DEBUG"] = 0] = "DEBUG";
    LogLevel[LogLevel["INFO"] = 1] = "INFO";
    LogLevel[LogLevel["WARN"] = 2] = "WARN";
    LogLevel[LogLevel["ERROR"] = 3] = "ERROR";
})(LogLevel || (LogLevel = {}));
class Logger {
    constructor() {
        this.logs = [];
        this.minLevel = LogLevel.INFO;
    }
    setLevel(level) {
        this.minLevel = level;
    }
    debug(message, context) {
        this.log(LogLevel.DEBUG, message, context);
    }
    info(message, context) {
        this.log(LogLevel.INFO, message, context);
    }
    warn(message, context) {
        this.log(LogLevel.WARN, message, context);
    }
    error(message, context) {
        this.log(LogLevel.ERROR, message, context);
    }
    log(level, message, context) {
        if (level < this.minLevel)
            return;
        const entry = {
            level,
            message,
            timestamp: new Date(),
            context
        };
        this.logs.push(entry);
        this.output(entry);
    }
    output(entry) {
        const levelName = LogLevel[entry.level];
        const timestamp = entry.timestamp.toISOString();
        const contextStr = entry.context ? JSON.stringify(entry.context) : '';
        console.log(`[${timestamp}] ${levelName}: ${entry.message} ${contextStr}`);
    }
    getLogs(level) {
        if (level !== undefined) {
            return this.logs.filter(log => log.level === level);
        }
        return [...this.logs];
    }
    clear() {
        this.logs = [];
    }
}
exports.Logger = Logger;
class TestRunner {
    constructor() {
        this.tests = [];
        this.results = [];
    }
    test(name, fn) {
        this.tests.push({ name, fn });
    }
    async run() {
        this.results = [];
        for (const test of this.tests) {
            const start = Date.now();
            try {
                await test.fn();
                this.results.push({
                    name: test.name,
                    passed: true,
                    duration: Date.now() - start
                });
            }
            catch (error) {
                this.results.push({
                    name: test.name,
                    passed: false,
                    error: error,
                    duration: Date.now() - start
                });
            }
        }
        return this.results;
    }
    getResults() {
        return this.results;
    }
}
exports.TestRunner = TestRunner;
// Assertion helpers
function expect(actual) {
    return {
        toBe(expected) {
            if (actual !== expected) {
                throw new Error(`Expected ${actual} to be ${expected}`);
            }
        },
        toEqual(expected) {
            if (JSON.stringify(actual) !== JSON.stringify(expected)) {
                throw new Error(`Expected ${actual} to equal ${expected}`);
            }
        },
        toBeTruthy() {
            if (!actual) {
                throw new Error(`Expected ${actual} to be truthy`);
            }
        },
        toBeFalsy() {
            if (actual) {
                throw new Error(`Expected ${actual} to be falsy`);
            }
        }
    };
}
// ============================================================================
// Utility Functions (Line 750+)
// ============================================================================
function debounce(fn, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}
function throttle(fn, limit) {
    let inThrottle;
    return function (...args) {
        if (!inThrottle) {
            fn.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object')
        return obj;
    if (obj instanceof Date)
        return new Date(obj.getTime());
    if (obj instanceof Array)
        return obj.map(item => deepClone(item));
    const cloned = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}
function isEqual(a, b) {
    if (a === b)
        return true;
    if (a == null || b == null)
        return false;
    if (typeof a !== 'object' || typeof b !== 'object')
        return false;
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    if (keysA.length !== keysB.length)
        return false;
    for (const key of keysA) {
        if (!keysB.includes(key))
            return false;
        if (!isEqual(a[key], b[key]))
            return false;
    }
    return true;
}
// ============================================================================
// THIS IS LINE 850 - THE IMPORTANT FUNCTION YOU SHOULD TEST WITH
// ============================================================================
/**
 * THIS IS THE FUNCTION TO SELECT FOR TESTING!
 * When you select this function and run any Squad command,
 * the context optimization should kick in automatically.
 */
function calculateImportantValue(a, b, c) {
    // This function is poorly typed - perfect for testing refactoring!
    const result = a + b * c;
    const adjusted = result / 2;
    return adjusted;
}
// ============================================================================
// More code continues below (Line 860+)
// ============================================================================
function formatDate(date, format = 'YYYY-MM-DD') {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return format
        .replace('YYYY', String(year))
        .replace('MM', month)
        .replace('DD', day)
        .replace('HH', hours)
        .replace('mm', minutes)
        .replace('ss', seconds);
}
function parseQuery(queryString) {
    const params = new URLSearchParams(queryString);
    const result = {};
    params.forEach((value, key) => {
        result[key] = value;
    });
    return result;
}
function buildQuery(params) {
    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
            searchParams.append(key, String(value));
        }
    }
    return searchParams.toString();
}
function generateRandomString(length) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
// ============================================================================
// Array Utilities (Line 920+)
// ============================================================================
function chunk(array, size) {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
        chunks.push(array.slice(i, i + size));
    }
    return chunks;
}
function flatten(array) {
    return array.reduce((acc, item) => {
        return acc.concat(Array.isArray(item) ? flatten(item) : item);
    }, []);
}
function unique(array) {
    return Array.from(new Set(array));
}
function groupBy(array, key) {
    return array.reduce((acc, item) => {
        const groupKey = String(item[key]);
        if (!acc[groupKey]) {
            acc[groupKey] = [];
        }
        acc[groupKey].push(item);
        return acc;
    }, {});
}
function sortBy(array, key, order = 'asc') {
    return [...array].sort((a, b) => {
        const aVal = a[key];
        const bVal = b[key];
        const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        return order === 'asc' ? comparison : -comparison;
    });
}
// ============================================================================
// String Utilities (Line 970+)
// ============================================================================
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}
function camelCase(str) {
    return str.replace(/[-_](.)/g, (_, char) => char.toUpperCase());
}
function snakeCase(str) {
    return str.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '');
}
function kebabCase(str) {
    return str.replace(/([A-Z])/g, '-$1').toLowerCase().replace(/^-/, '');
}
function truncate(str, length, suffix = '...') {
    if (str.length <= length)
        return str;
    return str.slice(0, length - suffix.length) + suffix;
}
function slugify(str) {
    return str
        .toLowerCase()
        .replace(/[^\w\s-]/g, '')
        .replace(/[\s_-]+/g, '-')
        .replace(/^-+|-+$/g, '');
}
// ============================================================================
// Number Utilities (Line 1010+)
// ============================================================================
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}
function randomFloat(min, max) {
    return Math.random() * (max - min) + min;
}
function round(value, decimals = 0) {
    const factor = Math.pow(10, decimals);
    return Math.round(value * factor) / factor;
}
function percentage(value, total) {
    return (value / total) * 100;
}
// ============================================================================
// Continuation of large file (Line 1040+)
// ============================================================================
class Counter {
    constructor() {
        this.count = 0;
    }
    increment() {
        return ++this.count;
    }
    decrement() {
        return --this.count;
    }
    reset() {
        this.count = 0;
    }
    value() {
        return this.count;
    }
}
exports.Counter = Counter;
class Timer {
    constructor() {
        this.start = 0;
        this.elapsed = 0;
        this.running = false;
    }
    begin() {
        this.start = Date.now();
        this.running = true;
    }
    stop() {
        if (this.running) {
            this.elapsed += Date.now() - this.start;
            this.running = false;
        }
        return this.elapsed;
    }
    reset() {
        this.start = 0;
        this.elapsed = 0;
        this.running = false;
    }
    getElapsed() {
        if (this.running) {
            return this.elapsed + (Date.now() - this.start);
        }
        return this.elapsed;
    }
}
exports.Timer = Timer;
// More filler content to reach 1500+ lines
const dummyData1 = { value: 1 };
const dummyData2 = { value: 2 };
const dummyData3 = { value: 3 };
const dummyData4 = { value: 4 };
const dummyData5 = { value: 5 };
const dummyData6 = { value: 6 };
const dummyData7 = { value: 7 };
const dummyData8 = { value: 8 };
const dummyData9 = { value: 9 };
const dummyData10 = { value: 10 };
// Line 1110+
function placeholder1() { return 1; }
function placeholder2() { return 2; }
function placeholder3() { return 3; }
function placeholder4() { return 4; }
function placeholder5() { return 5; }
function placeholder6() { return 6; }
function placeholder7() { return 7; }
function placeholder8() { return 8; }
function placeholder9() { return 9; }
function placeholder10() { return 10; }
function placeholder11() { return 11; }
function placeholder12() { return 12; }
function placeholder13() { return 13; }
function placeholder14() { return 14; }
function placeholder15() { return 15; }
function placeholder16() { return 16; }
function placeholder17() { return 17; }
function placeholder18() { return 18; }
function placeholder19() { return 19; }
function placeholder20() { return 20; }
// Line 1135+
const config1 = { key: 'value1' };
const config2 = { key: 'value2' };
const config3 = { key: 'value3' };
const config4 = { key: 'value4' };
const config5 = { key: 'value5' };
const config6 = { key: 'value6' };
const config7 = { key: 'value7' };
const config8 = { key: 'value8' };
const config9 = { key: 'value9' };
const config10 = { key: 'value10' };
const config11 = { key: 'value11' };
const config12 = { key: 'value12' };
const config13 = { key: 'value13' };
const config14 = { key: 'value14' };
const config15 = { key: 'value15' };
const config16 = { key: 'value16' };
const config17 = { key: 'value17' };
const config18 = { key: 'value18' };
const config19 = { key: 'value19' };
const config20 = { key: 'value20' };
// Line 1210+
class DummyClass1 {
    constructor() {
        this.prop = 1;
    }
}
class DummyClass2 {
    constructor() {
        this.prop = 2;
    }
}
class DummyClass3 {
    constructor() {
        this.prop = 3;
    }
}
class DummyClass4 {
    constructor() {
        this.prop = 4;
    }
}
class DummyClass5 {
    constructor() {
        this.prop = 5;
    }
}
class DummyClass6 {
    constructor() {
        this.prop = 6;
    }
}
class DummyClass7 {
    constructor() {
        this.prop = 7;
    }
}
class DummyClass8 {
    constructor() {
        this.prop = 8;
    }
}
class DummyClass9 {
    constructor() {
        this.prop = 9;
    }
}
class DummyClass10 {
    constructor() {
        this.prop = 10;
    }
}
class DummyClass11 {
    constructor() {
        this.prop = 11;
    }
}
class DummyClass12 {
    constructor() {
        this.prop = 12;
    }
}
class DummyClass13 {
    constructor() {
        this.prop = 13;
    }
}
class DummyClass14 {
    constructor() {
        this.prop = 14;
    }
}
class DummyClass15 {
    constructor() {
        this.prop = 15;
    }
}
class DummyClass16 {
    constructor() {
        this.prop = 16;
    }
}
class DummyClass17 {
    constructor() {
        this.prop = 17;
    }
}
class DummyClass18 {
    constructor() {
        this.prop = 18;
    }
}
class DummyClass19 {
    constructor() {
        this.prop = 19;
    }
}
class DummyClass20 {
    constructor() {
        this.prop = 20;
    }
}
// Line 1235+
var DummyEnum1;
(function (DummyEnum1) {
    DummyEnum1[DummyEnum1["A"] = 0] = "A";
    DummyEnum1[DummyEnum1["B"] = 1] = "B";
    DummyEnum1[DummyEnum1["C"] = 2] = "C";
})(DummyEnum1 || (DummyEnum1 = {}));
var DummyEnum2;
(function (DummyEnum2) {
    DummyEnum2[DummyEnum2["A"] = 0] = "A";
    DummyEnum2[DummyEnum2["B"] = 1] = "B";
    DummyEnum2[DummyEnum2["C"] = 2] = "C";
})(DummyEnum2 || (DummyEnum2 = {}));
var DummyEnum3;
(function (DummyEnum3) {
    DummyEnum3[DummyEnum3["A"] = 0] = "A";
    DummyEnum3[DummyEnum3["B"] = 1] = "B";
    DummyEnum3[DummyEnum3["C"] = 2] = "C";
})(DummyEnum3 || (DummyEnum3 = {}));
var DummyEnum4;
(function (DummyEnum4) {
    DummyEnum4[DummyEnum4["A"] = 0] = "A";
    DummyEnum4[DummyEnum4["B"] = 1] = "B";
    DummyEnum4[DummyEnum4["C"] = 2] = "C";
})(DummyEnum4 || (DummyEnum4 = {}));
var DummyEnum5;
(function (DummyEnum5) {
    DummyEnum5[DummyEnum5["A"] = 0] = "A";
    DummyEnum5[DummyEnum5["B"] = 1] = "B";
    DummyEnum5[DummyEnum5["C"] = 2] = "C";
})(DummyEnum5 || (DummyEnum5 = {}));
var DummyEnum6;
(function (DummyEnum6) {
    DummyEnum6[DummyEnum6["A"] = 0] = "A";
    DummyEnum6[DummyEnum6["B"] = 1] = "B";
    DummyEnum6[DummyEnum6["C"] = 2] = "C";
})(DummyEnum6 || (DummyEnum6 = {}));
var DummyEnum7;
(function (DummyEnum7) {
    DummyEnum7[DummyEnum7["A"] = 0] = "A";
    DummyEnum7[DummyEnum7["B"] = 1] = "B";
    DummyEnum7[DummyEnum7["C"] = 2] = "C";
})(DummyEnum7 || (DummyEnum7 = {}));
var DummyEnum8;
(function (DummyEnum8) {
    DummyEnum8[DummyEnum8["A"] = 0] = "A";
    DummyEnum8[DummyEnum8["B"] = 1] = "B";
    DummyEnum8[DummyEnum8["C"] = 2] = "C";
})(DummyEnum8 || (DummyEnum8 = {}));
var DummyEnum9;
(function (DummyEnum9) {
    DummyEnum9[DummyEnum9["A"] = 0] = "A";
    DummyEnum9[DummyEnum9["B"] = 1] = "B";
    DummyEnum9[DummyEnum9["C"] = 2] = "C";
})(DummyEnum9 || (DummyEnum9 = {}));
var DummyEnum10;
(function (DummyEnum10) {
    DummyEnum10[DummyEnum10["A"] = 0] = "A";
    DummyEnum10[DummyEnum10["B"] = 1] = "B";
    DummyEnum10[DummyEnum10["C"] = 2] = "C";
})(DummyEnum10 || (DummyEnum10 = {}));
var DummyEnum11;
(function (DummyEnum11) {
    DummyEnum11[DummyEnum11["A"] = 0] = "A";
    DummyEnum11[DummyEnum11["B"] = 1] = "B";
    DummyEnum11[DummyEnum11["C"] = 2] = "C";
})(DummyEnum11 || (DummyEnum11 = {}));
var DummyEnum12;
(function (DummyEnum12) {
    DummyEnum12[DummyEnum12["A"] = 0] = "A";
    DummyEnum12[DummyEnum12["B"] = 1] = "B";
    DummyEnum12[DummyEnum12["C"] = 2] = "C";
})(DummyEnum12 || (DummyEnum12 = {}));
var DummyEnum13;
(function (DummyEnum13) {
    DummyEnum13[DummyEnum13["A"] = 0] = "A";
    DummyEnum13[DummyEnum13["B"] = 1] = "B";
    DummyEnum13[DummyEnum13["C"] = 2] = "C";
})(DummyEnum13 || (DummyEnum13 = {}));
var DummyEnum14;
(function (DummyEnum14) {
    DummyEnum14[DummyEnum14["A"] = 0] = "A";
    DummyEnum14[DummyEnum14["B"] = 1] = "B";
    DummyEnum14[DummyEnum14["C"] = 2] = "C";
})(DummyEnum14 || (DummyEnum14 = {}));
var DummyEnum15;
(function (DummyEnum15) {
    DummyEnum15[DummyEnum15["A"] = 0] = "A";
    DummyEnum15[DummyEnum15["B"] = 1] = "B";
    DummyEnum15[DummyEnum15["C"] = 2] = "C";
})(DummyEnum15 || (DummyEnum15 = {}));
var DummyEnum16;
(function (DummyEnum16) {
    DummyEnum16[DummyEnum16["A"] = 0] = "A";
    DummyEnum16[DummyEnum16["B"] = 1] = "B";
    DummyEnum16[DummyEnum16["C"] = 2] = "C";
})(DummyEnum16 || (DummyEnum16 = {}));
var DummyEnum17;
(function (DummyEnum17) {
    DummyEnum17[DummyEnum17["A"] = 0] = "A";
    DummyEnum17[DummyEnum17["B"] = 1] = "B";
    DummyEnum17[DummyEnum17["C"] = 2] = "C";
})(DummyEnum17 || (DummyEnum17 = {}));
var DummyEnum18;
(function (DummyEnum18) {
    DummyEnum18[DummyEnum18["A"] = 0] = "A";
    DummyEnum18[DummyEnum18["B"] = 1] = "B";
    DummyEnum18[DummyEnum18["C"] = 2] = "C";
})(DummyEnum18 || (DummyEnum18 = {}));
var DummyEnum19;
(function (DummyEnum19) {
    DummyEnum19[DummyEnum19["A"] = 0] = "A";
    DummyEnum19[DummyEnum19["B"] = 1] = "B";
    DummyEnum19[DummyEnum19["C"] = 2] = "C";
})(DummyEnum19 || (DummyEnum19 = {}));
var DummyEnum20;
(function (DummyEnum20) {
    DummyEnum20[DummyEnum20["A"] = 0] = "A";
    DummyEnum20[DummyEnum20["B"] = 1] = "B";
    DummyEnum20[DummyEnum20["C"] = 2] = "C";
})(DummyEnum20 || (DummyEnum20 = {}));
// Continue with more filler to reach 1500+ lines
// Line 1260+
const array1 = [1, 2, 3, 4, 5];
const array2 = [6, 7, 8, 9, 10];
const array3 = [11, 12, 13, 14, 15];
const array4 = [16, 17, 18, 19, 20];
const array5 = [21, 22, 23, 24, 25];
const array6 = [26, 27, 28, 29, 30];
const array7 = [31, 32, 33, 34, 35];
const array8 = [36, 37, 38, 39, 40];
const array9 = [41, 42, 43, 44, 45];
const array10 = [46, 47, 48, 49, 50];
// More repetitive content
const obj1 = { id: 1, name: 'Object 1' };
const obj2 = { id: 2, name: 'Object 2' };
const obj3 = { id: 3, name: 'Object 3' };
const obj4 = { id: 4, name: 'Object 4' };
const obj5 = { id: 5, name: 'Object 5' };
const obj6 = { id: 6, name: 'Object 6' };
const obj7 = { id: 7, name: 'Object 7' };
const obj8 = { id: 8, name: 'Object 8' };
const obj9 = { id: 9, name: 'Object 9' };
const obj10 = { id: 10, name: 'Object 10' };
// Even more filler content
function helper1(x) { return x * 1; }
function helper2(x) { return x * 2; }
function helper3(x) { return x * 3; }
function helper4(x) { return x * 4; }
function helper5(x) { return x * 5; }
function helper6(x) { return x * 6; }
function helper7(x) { return x * 7; }
function helper8(x) { return x * 8; }
function helper9(x) { return x * 9; }
function helper10(x) { return x * 10; }
// And more...
const str1 = 'String 1';
const str2 = 'String 2';
const str3 = 'String 3';
const str4 = 'String 4';
const str5 = 'String 5';
const str6 = 'String 6';
const str7 = 'String 7';
const str8 = 'String 8';
const str9 = 'String 9';
const str10 = 'String 10';
// Final section to ensure 1500+ lines
const finalData = {
    line1: 'data',
    line2: 'data',
    line3: 'data',
    line4: 'data',
    line5: 'data',
    line6: 'data',
    line7: 'data',
    line8: 'data',
    line9: 'data',
    line10: 'data',
    line11: 'data',
    line12: 'data',
    line13: 'data',
    line14: 'data',
    line15: 'data',
    line16: 'data',
    line17: 'data',
    line18: 'data',
    line19: 'data',
    line20: 'data',
};
//# sourceMappingURL=test_large_file.js.map