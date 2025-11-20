/**
 * Redux slice for posts state
 */
import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { PostsState, Post, CreatePostRequest, PostsListParams } from '@/types';
import { apiClient } from '@/api/client';

const initialState: PostsState = {
  posts: [],
  currentPost: null,
  loading: false,
  error: null,
  hasMore: true,
  page: 1,
};

// Async thunks
export const fetchPosts = createAsyncThunk(
  'posts/fetchPosts',
  async (params: PostsListParams = {}, { rejectWithValue }) => {
    try {
      const posts = await apiClient.getPosts(params);
      return { posts, page: params.page || 1 };
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to fetch posts');
    }
  }
);

export const fetchPost = createAsyncThunk(
  'posts/fetchPost',
  async (postId: string, { rejectWithValue }) => {
    try {
      const post = await apiClient.getPost(postId);
      return post;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to fetch post');
    }
  }
);

export const createPost = createAsyncThunk(
  'posts/createPost',
  async (data: CreatePostRequest, { rejectWithValue }) => {
    try {
      const post = await apiClient.createPost(data);
      return post;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to create post');
    }
  }
);

export const votePost = createAsyncThunk(
  'posts/votePost',
  async ({ postId, voteType }: { postId: string; voteType: 'upvote' | 'downvote' }, { rejectWithValue }) => {
    try {
      const post = await apiClient.votePost(postId, voteType);
      return post;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to vote');
    }
  }
);

export const deletePost = createAsyncThunk(
  'posts/deletePost',
  async (postId: string, { rejectWithValue }) => {
    try {
      await apiClient.deletePost(postId);
      return postId;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.detail || 'Failed to delete post');
    }
  }
);

const postsSlice = createSlice({
  name: 'posts',
  initialState,
  reducers: {
    clearPosts: (state) => {
      state.posts = [];
      state.page = 1;
      state.hasMore = true;
    },
    clearCurrentPost: (state) => {
      state.currentPost = null;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch posts
    builder
      .addCase(fetchPosts.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchPosts.fulfilled, (state, action) => {
        state.loading = false;
        const { posts, page } = action.payload;
        if (page === 1) {
          state.posts = posts;
        } else {
          state.posts = [...state.posts, ...posts];
        }
        state.page = page;
        state.hasMore = posts.length > 0;
      })
      .addCase(fetchPosts.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Fetch single post
    builder
      .addCase(fetchPost.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchPost.fulfilled, (state, action) => {
        state.loading = false;
        state.currentPost = action.payload;
      })
      .addCase(fetchPost.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Create post
    builder
      .addCase(createPost.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createPost.fulfilled, (state, action) => {
        state.loading = false;
        state.posts.unshift(action.payload);
      })
      .addCase(createPost.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Vote post
    builder
      .addCase(votePost.fulfilled, (state, action) => {
        const updatedPost = action.payload;
        // Update in posts list
        const index = state.posts.findIndex(p => p.post_id === updatedPost.post_id);
        if (index !== -1) {
          state.posts[index] = updatedPost;
        }
        // Update current post if it matches
        if (state.currentPost?.post_id === updatedPost.post_id) {
          state.currentPost = updatedPost;
        }
      });

    // Delete post
    builder
      .addCase(deletePost.fulfilled, (state, action) => {
        state.posts = state.posts.filter(p => p.post_id !== action.payload);
        if (state.currentPost?.post_id === action.payload) {
          state.currentPost = null;
        }
      });
  },
});

export const { clearPosts, clearCurrentPost, clearError } = postsSlice.actions;
export default postsSlice.reducer;
