/**
 * O2 Mobile App
 *
 * Platform Anarchism on iOS & Android
 *
 * Features:
 * - Matrix messaging
 * - HoloLoom queries
 * - Democratic governance (proposals, voting)
 * - Memory management
 * - Real-time updates
 */

import React, {useState, useEffect} from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  Alert,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';

// O2 API Configuration
const O2_API_URL = 'http://localhost:8080';

// ============================================
// Types
// ============================================

interface Message {
  id: string;
  sender: string;
  content: string;
  timestamp: Date;
}

interface Proposal {
  proposal_id: number;
  title: string;
  status: string;
  created_at: string;
}

// ============================================
// Main App Component
// ============================================

function App(): JSX.Element {
  const [token, setToken] = useState<string | null>(null);
  const [userId, setUserId] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [activeTab, setActiveTab] = useState<'chat' | 'governance' | 'memory'>(
    'chat',
  );

  // Load token on mount
  useEffect(() => {
    loadToken();
  }, []);

  // Load proposals when governance tab active
  useEffect(() => {
    if (activeTab === 'governance' && token) {
      loadProposals();
    }
  }, [activeTab, token]);

  // ============================================
  // Authentication
  // ============================================

  const loadToken = async () => {
    const savedToken = await AsyncStorage.getItem('o2_token');
    if (savedToken) {
      setToken(savedToken);
    }
  };

  const login = async () => {
    try {
      const response = await axios.post(`${O2_API_URL}/auth/login`, {
        user_id: userId,
        password: password,
      });

      const {token: newToken} = response.data;
      await AsyncStorage.setItem('o2_token', newToken);
      setToken(newToken);
      Alert.alert('Success', 'Logged in to O2 Platform!');
    } catch (error) {
      Alert.alert('Error', 'Login failed. Please check your credentials.');
    }
  };

  const logout = async () => {
    await AsyncStorage.removeItem('o2_token');
    setToken(null);
    setUserId('');
    setPassword('');
    Alert.alert('Success', 'Logged out');
  };

  // ============================================
  // Query (Chat)
  // ============================================

  const sendQuery = async () => {
    if (!query.trim()) {
      return;
    }

    try {
      const response = await axios.post(
        `${O2_API_URL}/query`,
        {
          query: query,
          mode: 'direct',
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      setResponse(response.data.response);
      setQuery('');
    } catch (error) {
      Alert.alert('Error', 'Query failed');
    }
  };

  // ============================================
  // Governance
  // ============================================

  const loadProposals = async () => {
    try {
      const response = await axios.get(`${O2_API_URL}/proposals`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      setProposals(response.data.proposals || []);
    } catch (error) {
      console.error('Failed to load proposals:', error);
    }
  };

  const vote = async (proposalId: number, voteValue: string) => {
    try {
      await axios.post(
        `${O2_API_URL}/proposals/${proposalId}/vote`,
        {
          proposal_id: proposalId,
          vote: voteValue,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      Alert.alert('Success', `Voted ${voteValue} on proposal #${proposalId}`);
      loadProposals(); // Reload to see updated vote counts
    } catch (error) {
      Alert.alert('Error', 'Vote failed');
    }
  };

  // ============================================
  // Render
  // ============================================

  if (!token) {
    // Login Screen
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" />
        <View style={styles.loginContainer}>
          <Text style={styles.title}>O2 Platform</Text>
          <Text style={styles.subtitle}>
            Platform Anarchism meets Agentic Intelligence
          </Text>

          <TextInput
            style={styles.input}
            placeholder="Matrix User ID (@user:server.org)"
            value={userId}
            onChangeText={setUserId}
            autoCapitalize="none"
          />

          <TextInput
            style={styles.input}
            placeholder="Password"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />

          <TouchableOpacity style={styles.button} onPress={login}>
            <Text style={styles.buttonText}>Login</Text>
          </TouchableOpacity>

          <Text style={styles.infoText}>
            Don't have an account? Set up your own O2 instance!
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  // Main App Screen
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>O2 Platform</Text>
        <TouchableOpacity onPress={logout}>
          <Text style={styles.logoutButton}>Logout</Text>
        </TouchableOpacity>
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'chat' && styles.activeTab]}
          onPress={() => setActiveTab('chat')}>
          <Text
            style={[
              styles.tabText,
              activeTab === 'chat' && styles.activeTabText,
            ]}>
            Chat
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tab, activeTab === 'governance' && styles.activeTab]}
          onPress={() => setActiveTab('governance')}>
          <Text
            style={[
              styles.tabText,
              activeTab === 'governance' && styles.activeTabText,
            ]}>
            Governance
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.tab, activeTab === 'memory' && styles.activeTab]}
          onPress={() => setActiveTab('memory')}>
          <Text
            style={[
              styles.tabText,
              activeTab === 'memory' && styles.activeTabText,
            ]}>
            Memory
          </Text>
        </TouchableOpacity>
      </View>

      {/* Content */}
      <ScrollView style={styles.content}>
        {activeTab === 'chat' && (
          <View style={styles.chatContainer}>
            <Text style={styles.sectionTitle}>Ask HoloLoom</Text>

            {response && (
              <View style={styles.responseContainer}>
                <Text style={styles.responseText}>{response}</Text>
              </View>
            )}

            <TextInput
              style={styles.queryInput}
              placeholder="Ask anything..."
              value={query}
              onChangeText={setQuery}
              multiline
            />

            <TouchableOpacity style={styles.button} onPress={sendQuery}>
              <Text style={styles.buttonText}>Send Query</Text>
            </TouchableOpacity>
          </View>
        )}

        {activeTab === 'governance' && (
          <View style={styles.governanceContainer}>
            <Text style={styles.sectionTitle}>Active Proposals</Text>

            {proposals.length === 0 && (
              <Text style={styles.emptyText}>No active proposals</Text>
            )}

            {proposals.map(proposal => (
              <View key={proposal.proposal_id} style={styles.proposalCard}>
                <Text style={styles.proposalTitle}>
                  #{proposal.proposal_id}: {proposal.title}
                </Text>
                <Text style={styles.proposalStatus}>
                  Status: {proposal.status}
                </Text>

                <View style={styles.voteButtons}>
                  <TouchableOpacity
                    style={[styles.voteButton, styles.voteYes]}
                    onPress={() => vote(proposal.proposal_id, 'yes')}>
                    <Text style={styles.voteButtonText}>✅ Yes</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.voteButton, styles.voteNo]}
                    onPress={() => vote(proposal.proposal_id, 'no')}>
                    <Text style={styles.voteButtonText}>❌ No</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.voteButton, styles.voteAbstain]}
                    onPress={() => vote(proposal.proposal_id, 'abstain')}>
                    <Text style={styles.voteButtonText}>🤔 Abstain</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </View>
        )}

        {activeTab === 'memory' && (
          <View style={styles.memoryContainer}>
            <Text style={styles.sectionTitle}>Your Memory</Text>
            <Text style={styles.infoText}>
              Your knowledge graph is private and encrypted.
            </Text>

            <TouchableOpacity style={styles.button}>
              <Text style={styles.buttonText}>Export Data</Text>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.button, styles.secondaryButton]}>
              <Text style={styles.buttonText}>Shared Memories</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

// ============================================
// Styles
// ============================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loginContainer: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  logoutButton: {
    color: '#007AFF',
    fontSize: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginBottom: 32,
  },
  tabs: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  tab: {
    flex: 1,
    paddingVertical: 16,
    alignItems: 'center',
  },
  activeTab: {
    borderBottomWidth: 2,
    borderBottomColor: '#007AFF',
  },
  tabText: {
    fontSize: 16,
    color: '#666',
  },
  activeTabText: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  input: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
    fontSize: 16,
  },
  queryInput: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
    fontSize: 16,
    minHeight: 100,
    textAlignVertical: 'top',
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 16,
  },
  secondaryButton: {
    backgroundColor: '#5AC8FA',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  infoText: {
    textAlign: 'center',
    color: '#666',
    marginTop: 16,
  },
  emptyText: {
    textAlign: 'center',
    color: '#999',
    fontSize: 16,
    marginTop: 32,
  },
  chatContainer: {
    padding: 16,
  },
  responseContainer: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
  },
  responseText: {
    fontSize: 16,
    lineHeight: 24,
  },
  governanceContainer: {
    padding: 16,
  },
  proposalCard: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
  },
  proposalTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  proposalStatus: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
  },
  voteButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  voteButton: {
    flex: 1,
    padding: 12,
    borderRadius: 6,
    alignItems: 'center',
    marginHorizontal: 4,
  },
  voteYes: {
    backgroundColor: '#34C759',
  },
  voteNo: {
    backgroundColor: '#FF3B30',
  },
  voteAbstain: {
    backgroundColor: '#FF9500',
  },
  voteButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  memoryContainer: {
    padding: 16,
  },
});

export default App;
