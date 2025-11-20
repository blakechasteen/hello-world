/**
 * API Client for Community Platform backend
 * Handles HTTP requests with axios and authentication
 */
import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  User,
  Post,
  Comment,
  Community,
  AuthResponse,
  LoginRequest,
  RegisterRequest,
  CreatePostRequest,
  CreateCommentRequest,
  VoteRequest,
  CreateCommunityRequest,
  UpdateUserRequest,
  PostsListParams,
  CommentsListParams,
  SearchParams,
} from '@/types';

const API_BASE_URL = '/api/v1';

class APIClient {
  private client: AxiosInstance;
  private accessToken: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for adding auth token
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for handling errors
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired - try to refresh
          try {
            await this.refreshToken();
            // Retry original request
            return this.client.request(error.config!);
          } catch {
            // Refresh failed - logout
            this.clearTokens();
            window.location.href = '/login';
          }
        }
        return Promise.reject(error);
      }
    );

    // Load tokens from localStorage
    const storedAccessToken = localStorage.getItem('accessToken');
    const storedRefreshToken = localStorage.getItem('refreshToken');
    if (storedAccessToken && storedRefreshToken) {
      this.setTokens(storedAccessToken, storedRefreshToken);
    }
  }

  setTokens(accessToken: string, refreshToken: string) {
    this.accessToken = accessToken;
    localStorage.setItem('accessToken', accessToken);
    localStorage.setItem('refreshToken', refreshToken);
  }

  clearTokens() {
    this.accessToken = null;
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
  }

  // Authentication
  async register(data: RegisterRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/register', data);
    this.setTokens(response.data.access_token, response.data.refresh_token);
    return response.data;
  }

  async login(data: LoginRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/login', data);
    this.setTokens(response.data.access_token, response.data.refresh_token);
    return response.data;
  }

  async logout(): Promise<void> {
    await this.client.post('/auth/logout');
    this.clearTokens();
  }

  async refreshToken(): Promise<void> {
    const refreshToken = localStorage.getItem('refreshToken');
    if (!refreshToken) throw new Error('No refresh token');

    const response = await this.client.post<{ access_token: string }>('/auth/refresh', {
      refresh_token: refreshToken,
    });
    this.accessToken = response.data.access_token;
    localStorage.setItem('accessToken', response.data.access_token);
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/me');
    return response.data;
  }

  // Users
  async getUser(username: string): Promise<User> {
    const response = await this.client.get<User>(`/users/${username}`);
    return response.data;
  }

  async updateUser(username: string, data: UpdateUserRequest): Promise<User> {
    const response = await this.client.patch<User>(`/users/${username}`, data);
    return response.data;
  }

  async getUserPosts(username: string, params?: PostsListParams): Promise<Post[]> {
    const response = await this.client.get<Post[]>(`/users/${username}/posts`, { params });
    return response.data;
  }

  async getUserComments(username: string, params?: PostsListParams): Promise<Comment[]> {
    const response = await this.client.get<Comment[]>(`/users/${username}/comments`, { params });
    return response.data;
  }

  // Communities
  async getCommunities(params?: PostsListParams): Promise<Community[]> {
    const response = await this.client.get<Community[]>('/communities', { params });
    return response.data;
  }

  async getCommunity(name: string): Promise<Community> {
    const response = await this.client.get<Community>(`/communities/${name}`);
    return response.data;
  }

  async createCommunity(data: CreateCommunityRequest): Promise<Community> {
    const response = await this.client.post<Community>('/communities', data);
    return response.data;
  }

  async joinCommunity(name: string): Promise<void> {
    await this.client.post(`/communities/${name}/join`);
  }

  async leaveCommunity(name: string): Promise<void> {
    await this.client.post(`/communities/${name}/leave`);
  }

  // Posts
  async getPosts(params?: PostsListParams): Promise<Post[]> {
    const response = await this.client.get<Post[]>('/posts', { params });
    return response.data;
  }

  async getPost(postId: string): Promise<Post> {
    const response = await this.client.get<Post>(`/posts/${postId}`);
    return response.data;
  }

  async createPost(data: CreatePostRequest): Promise<Post> {
    const response = await this.client.post<Post>('/posts', data);
    return response.data;
  }

  async updatePost(postId: string, data: Partial<CreatePostRequest>): Promise<Post> {
    const response = await this.client.patch<Post>(`/posts/${postId}`, data);
    return response.data;
  }

  async deletePost(postId: string): Promise<void> {
    await this.client.delete(`/posts/${postId}`);
  }

  async votePost(postId: string, voteType: 'upvote' | 'downvote'): Promise<Post> {
    const response = await this.client.post<Post>(`/posts/${postId}/vote`, { vote_type: voteType });
    return response.data;
  }

  // Comments
  async getPostComments(postId: string, params?: CommentsListParams): Promise<Comment[]> {
    const response = await this.client.get<Comment[]>(`/comments/posts/${postId}`, { params });
    return response.data;
  }

  async getComment(commentId: string): Promise<Comment> {
    const response = await this.client.get<Comment>(`/comments/${commentId}`);
    return response.data;
  }

  async createComment(postId: string, data: CreateCommentRequest): Promise<Comment> {
    const response = await this.client.post<Comment>(`/comments/posts/${postId}`, data);
    return response.data;
  }

  async updateComment(commentId: string, content: string): Promise<Comment> {
    const response = await this.client.patch<Comment>(`/comments/${commentId}`, { content });
    return response.data;
  }

  async deleteComment(commentId: string): Promise<void> {
    await this.client.delete(`/comments/${commentId}`);
  }

  async voteComment(commentId: string, voteType: 'upvote' | 'downvote'): Promise<Comment> {
    const response = await this.client.post<Comment>(`/comments/${commentId}/vote`, { vote_type: voteType });
    return response.data;
  }

  // Search
  async search(params: SearchParams): Promise<any[]> {
    const endpoint = params.type ? `/search/${params.type}` : '/search/posts';
    const response = await this.client.get<any[]>(endpoint, { params: { q: params.q, page: params.page, page_size: params.page_size } });
    return response.data;
  }
}

export const apiClient = new APIClient();
export default apiClient;
