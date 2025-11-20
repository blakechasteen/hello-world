/**
 * TypeScript types for Community Platform
 * Matches backend API models
 */

export interface User {
  user_id: string;
  username: string;
  email: string;
  display_name?: string;
  bio?: string;
  avatar_url?: string;
  reputation_score: number;
  trust_level: number;
  post_count: number;
  comment_count: number;
  status: 'active' | 'suspended' | 'banned';
  role: 'guest' | 'member' | 'moderator' | 'admin';
  created_at: string;
  updated_at: string;
}

export interface Community {
  community_id: string;
  name: string;
  display_name: string;
  description?: string;
  rules?: string;
  banner_url?: string;
  icon_url?: string;
  member_count: number;
  post_count: number;
  visibility: 'public' | 'private' | 'restricted';
  status: 'active' | 'suspended' | 'archived';
  creator: User;
  created_at: string;
  updated_at: string;
}

export interface Post {
  post_id: string;
  title: string;
  content?: string;
  post_type: 'text' | 'link' | 'image' | 'video' | 'poll' | 'question';
  link_url?: string;
  media_urls?: string[];
  tags?: string[];
  upvote_count: number;
  downvote_count: number;
  vote_score: number;
  comment_count: number;
  view_count: number;
  hot_score: number;
  status: 'published' | 'draft' | 'removed' | 'deleted';
  pinned: boolean;
  locked: boolean;
  author: User;
  community: Community;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Comment {
  comment_id: string;
  post_id: string;
  author: User;
  parent_id?: string;
  content: string;
  path: string;
  depth: number;
  upvote_count: number;
  downvote_count: number;
  vote_score: number;
  children_count: number;
  status: 'published' | 'pending_moderation' | 'removed' | 'deleted';
  edited: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  display_name?: string;
}

export interface CreatePostRequest {
  title: string;
  content?: string;
  post_type: 'text' | 'link' | 'image' | 'video' | 'poll' | 'question';
  community_name: string;
  link_url?: string;
  media_urls?: string[];
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface CreateCommentRequest {
  content: string;
  parent_id?: string;
}

export interface VoteRequest {
  vote_type: 'upvote' | 'downvote';
}

export interface CreateCommunityRequest {
  name: string;
  display_name: string;
  description?: string;
  rules?: string;
  visibility?: 'public' | 'private' | 'restricted';
}

export interface UpdateUserRequest {
  display_name?: string;
  bio?: string;
  avatar_url?: string;
}

export interface PaginationParams {
  page?: number;
  page_size?: number;
}

export interface PostsListParams extends PaginationParams {
  community_name?: string;
  sort_by?: 'new' | 'hot' | 'top' | 'controversial';
}

export interface CommentsListParams extends PaginationParams {
  sort_by?: 'best' | 'new' | 'top' | 'controversial';
}

export interface SearchParams extends PaginationParams {
  q: string;
  type?: 'posts' | 'users' | 'communities';
}

// WebSocket Events
export interface WSMessage {
  conversation_id: string;
  sender_id: string;
  content: string;
  timestamp: string;
}

export interface WSUserTyping {
  user_id: string;
  conversation_id: string;
}

export interface WSPostUpdate {
  post_id: string;
  update_type: 'vote' | 'comment' | 'edit';
}

// Redux State
export interface RootState {
  auth: AuthState;
  posts: PostsState;
  communities: CommunitiesState;
  ui: UIState;
}

export interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
}

export interface PostsState {
  posts: Post[];
  currentPost: Post | null;
  loading: boolean;
  error: string | null;
  hasMore: boolean;
  page: number;
}

export interface CommunitiesState {
  communities: Community[];
  currentCommunity: Community | null;
  loading: boolean;
  error: string | null;
}

export interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  notifications: Notification[];
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
  timestamp: string;
}
