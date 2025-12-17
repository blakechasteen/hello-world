'use client';

import { useState } from 'react';
import {
  Button,
  Card,
  CardHeader,
  CardBody,
  Input,
  ConfidenceIndicator,
  SafetyBadge,
  ReasoningBadge,
  LoadingSpinner,
  WeavingIndicator,
  Avatar,
  Badge,
} from '@hololoom/design-system';
import { useHoloLoom, useConversation, useStats } from '@hololoom/api-client';
import { Navigation } from '@/components/Navigation';
import { ChatInterface } from '@/components/ChatInterface';
import { QuickActions } from '@/components/QuickActions';

export default function HomePage() {
  const { isConnected, health } = useHoloLoom();
  const { data: stats } = useStats();
  const [showChat, setShowChat] = useState(false);

  return (
    <div className="page-container">
      {/* Navigation */}
      <Navigation />

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Cosmic background effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-cosmic-nebula/5 via-transparent to-cosmic-aurora/5" />
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cosmic-nebula/10 rounded-full blur-3xl animate-cosmic-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cosmic-aurora/10 rounded-full blur-3xl animate-cosmic-pulse" style={{ animationDelay: '1.5s' }} />

        <div className="relative main-content py-20 text-center">
          {/* Status indicator */}
          <div className="flex items-center justify-center gap-2 mb-6">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-confidence-high animate-pulse' : 'bg-safety-danger'}`} />
            <span className="text-sm text-fg-secondary">
              {isConnected ? 'Connected to HoloLoom' : 'Connecting...'}
            </span>
            {health && (
              <SafetyBadge
                level={health.status === 'healthy' ? 'safe' : health.status === 'degraded' ? 'caution' : 'danger'}
                size="sm"
                showLabel={false}
              />
            )}
          </div>

          {/* Main heading */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6">
            <span className="text-gradient-cosmic">HoloLoom</span>
          </h1>
          <p className="text-xl sm:text-2xl text-fg-secondary max-w-2xl mx-auto mb-8">
            Intelligent knowledge weaving with multi-scale memory,
            recursive learning, and transparent AI reasoning.
          </p>

          {/* Quick stats */}
          {stats && (
            <div className="flex flex-wrap justify-center gap-6 mb-10">
              <div className="text-center">
                <div className="text-3xl font-bold text-cosmic-nebula">{stats.memory.totalNodes.toLocaleString()}</div>
                <div className="text-sm text-fg-tertiary">Memories</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-cosmic-aurora">{(stats.cacheHitRate * 100).toFixed(0)}%</div>
                <div className="text-sm text-fg-tertiary">Cache Hit</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-cosmic-starlight">{stats.avgLatencyMs.toFixed(0)}ms</div>
                <div className="text-sm text-fg-tertiary">Avg Latency</div>
              </div>
            </div>
          )}

          {/* CTA buttons */}
          <div className="flex flex-wrap justify-center gap-4">
            <Button
              size="lg"
              variant="primary"
              onClick={() => setShowChat(true)}
            >
              Start Conversation
            </Button>
            <Button
              size="lg"
              variant="ghost"
              onClick={() => window.location.href = '/memory'}
            >
              Explore Memory
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="main-content py-16">
        <h2 className="text-2xl font-bold text-center mb-10">Reasoning Modes</h2>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <Card variant="elevated" className="cosmic-glow">
            <CardHeader>
              <div className="flex items-center gap-3">
                <ReasoningBadge mode="direct" size="lg" />
                <h3 className="font-semibold">Direct</h3>
              </div>
            </CardHeader>
            <CardBody>
              <p className="text-fg-secondary text-sm">
                Single-pass answers for simple factual queries. ~150ms latency.
              </p>
            </CardBody>
          </Card>

          <Card variant="elevated" className="cosmic-glow">
            <CardHeader>
              <div className="flex items-center gap-3">
                <ReasoningBadge mode="verify" size="lg" />
                <h3 className="font-semibold">Verify</h3>
              </div>
            </CardHeader>
            <CardBody>
              <p className="text-fg-secondary text-sm">
                Answer with verification step. Checks for contradictions. ~600ms.
              </p>
            </CardBody>
          </Card>

          <Card variant="elevated" className="cosmic-glow">
            <CardHeader>
              <div className="flex items-center gap-3">
                <ReasoningBadge mode="research" size="lg" />
                <h3 className="font-semibold">Research</h3>
              </div>
            </CardHeader>
            <CardBody>
              <p className="text-fg-secondary text-sm">
                Multi-query exploration for complex topics. Deep analysis. ~900ms.
              </p>
            </CardBody>
          </Card>

          <Card variant="elevated" className="cosmic-glow">
            <CardHeader>
              <div className="flex items-center gap-3">
                <ReasoningBadge mode="plan_execute" size="lg" />
                <h3 className="font-semibold">Plan & Execute</h3>
              </div>
            </CardHeader>
            <CardBody>
              <p className="text-fg-secondary text-sm">
                Goal decomposition into steps. For multi-step tasks. ~750ms.
              </p>
            </CardBody>
          </Card>
        </div>
      </section>

      {/* AI Transparency Section */}
      <section className="bg-bg-secondary py-16">
        <div className="main-content">
          <h2 className="text-2xl font-bold text-center mb-4">AI Transparency</h2>
          <p className="text-fg-secondary text-center max-w-2xl mx-auto mb-10">
            Every decision is transparent. See confidence levels, safety assessments,
            and complete audit trails for all AI reasoning.
          </p>

          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader>
                <h3 className="font-semibold">Confidence Levels</h3>
              </CardHeader>
              <CardBody className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-fg-secondary">High confidence</span>
                  <ConfidenceIndicator value={0.92} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-fg-secondary">Medium confidence</span>
                  <ConfidenceIndicator value={0.65} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-fg-secondary">Low confidence</span>
                  <ConfidenceIndicator value={0.35} />
                </div>
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">Safety Levels</h3>
              </CardHeader>
              <CardBody className="space-y-3">
                <div className="flex items-center gap-3">
                  <SafetyBadge level="safe" size="sm" />
                  <span className="text-sm text-fg-secondary">Action approved</span>
                </div>
                <div className="flex items-center gap-3">
                  <SafetyBadge level="caution" size="sm" />
                  <span className="text-sm text-fg-secondary">Review recommended</span>
                </div>
                <div className="flex items-center gap-3">
                  <SafetyBadge level="warning" size="sm" />
                  <span className="text-sm text-fg-secondary">Human approval needed</span>
                </div>
                <div className="flex items-center gap-3">
                  <SafetyBadge level="danger" size="sm" />
                  <span className="text-sm text-fg-secondary">Action blocked</span>
                </div>
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">Weaving Progress</h3>
              </CardHeader>
              <CardBody>
                <div className="space-y-4">
                  <WeavingIndicator step={4} showLabels />
                  <p className="text-sm text-fg-tertiary mt-4">
                    9-step weaving cycle with real-time progress tracking.
                  </p>
                </div>
              </CardBody>
            </Card>
          </div>
        </div>
      </section>

      {/* Quick Actions */}
      <section className="main-content py-16">
        <QuickActions />
      </section>

      {/* Chat Modal */}
      {showChat && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-bg-primary/80 backdrop-blur-sm">
          <div className="w-full max-w-2xl h-[80vh] m-4">
            <Card className="h-full flex flex-col">
              <CardHeader className="flex items-center justify-between border-b border-border-primary">
                <div className="flex items-center gap-3">
                  <Avatar variant="ai" size="sm" />
                  <h2 className="font-semibold">HoloLoom Assistant</h2>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowChat(false)}
                >
                  Close
                </Button>
              </CardHeader>
              <CardBody className="flex-1 overflow-hidden p-0">
                <ChatInterface />
              </CardBody>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
