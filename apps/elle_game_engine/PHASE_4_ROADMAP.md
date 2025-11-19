# BigPlay Engine - Phase 4 Roadmap

**Version**: 1.1.0 - 2.0.0
**Status**: Planning
**Timeline**: 8-12 weeks
**Complexity**: High

---

## 🎯 Overview

Phase 4 represents a major evolution of BigPlay Engine, transforming it from a single-player narrative system into a **multiplayer-capable, autonomous NPC platform** with visual authoring tools.

### Objectives

1. **Multiplayer Support** - Enable shared worlds with real-time synchronization
2. **Advanced NPC Autonomy** - NPCs with goals, schedules, and emergent behavior
3. **Visual Workflow Builder** - No-code tools for quest and NPC design

### Success Criteria

- ✅ 100+ concurrent players in shared world
- ✅ NPCs autonomously pursue goals without player interaction
- ✅ Non-technical users can create quests/NPCs visually
- ✅ Zero breaking changes to v1.0 API
- ✅ <100ms additional latency for multiplayer features

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      GAME CLIENTS (v1.0)                         │
│         Unity     Godot     Unreal     Web     Mobile            │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/JSON + WebSocket
┌────────────────────────┴────────────────────────────────────────┐
│                    BIGPLAY ENGINE v1.0                           │
│  (Existing: Emotions, Quests, Voice, Sessions, Fine-tuning)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    PHASE 4 EXTENSIONS                            │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   Multiplayer    │  │   NPC Autonomy   │  │   Workflow   │  │
│  │                  │  │                  │  │   Builder    │  │
│  │  Redis Pub/Sub   │  │  GOAP Planner    │  │  React UI    │  │
│  │  WebSocket       │  │  Daily Routines  │  │  Drag/Drop   │  │
│  │  Session Sync    │  │  Behavior Trees  │  │  JSON Export │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                SHARED INFRASTRUCTURE                      │  │
│  │   Redis (world state)  │  PostgreSQL (persistence)       │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Multiplayer Support

**Goal**: Enable 100+ players to share a persistent world with real-time NPC updates.

### 1.1 Architecture Design

**Tech Stack**:
- **Redis**: Shared world state (pub/sub, sorted sets for leaderboards)
- **WebSocket**: Real-time bidirectional communication
- **PostgreSQL**: Persistent world state backup
- **FastAPI WebSocket**: Native WebSocket support

**Data Flow**:
```
Player A → WebSocket → BigPlay → Redis (publish) → WebSocket → Player B
                             ↓
                      PostgreSQL (persist)
```

### 1.2 Core Components

#### 1.2.1 Shared World State Manager (`multiplayer/world_state.py`)

**Responsibilities**:
- Synchronize world flags across all players
- Track active players in world
- Broadcast NPC state changes

**Key Classes**:
```python
class SharedWorldState:
    """Redis-backed shared world state"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def set_flag(self, flag: str, value: bool):
        """Set world flag, broadcast to all players"""
        await self.redis.set(f"world:flag:{flag}", value)
        await self.redis.publish("world:updates", json.dumps({
            "type": "flag_changed",
            "flag": flag,
            "value": value
        }))

    async def get_active_players(self) -> List[str]:
        """Get list of active player IDs"""
        return await self.redis.smembers("world:active_players")

    async def broadcast_npc_state(self, npc_id: str, state: dict):
        """Broadcast NPC emotional/quest state change"""
        await self.redis.publish("npc:updates", json.dumps({
            "npc_id": npc_id,
            "state": state
        }))
```

**Estimated Lines**: 350-450

#### 1.2.2 WebSocket Server (`multiplayer/websocket_server.py`)

**Responsibilities**:
- Handle WebSocket connections
- Subscribe to Redis pub/sub channels
- Route updates to connected clients

**Key Classes**:
```python
class MultiplayerWebSocketManager:
    """Manages WebSocket connections for multiplayer"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, player_id: str):
        """Register new player connection"""
        await websocket.accept()
        self.connections[player_id] = websocket
        await self.redis.sadd("world:active_players", player_id)

    async def disconnect(self, player_id: str):
        """Remove player connection"""
        self.connections.pop(player_id, None)
        await self.redis.srem("world:active_players", player_id)

    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        """Broadcast message to all connected players"""
        for player_id, ws in self.connections.items():
            if player_id != exclude:
                await ws.send_json(message)

    async def listen_redis(self):
        """Listen to Redis pub/sub and forward to WebSocket clients"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("world:updates", "npc:updates")

        async for message in pubsub.listen():
            if message["type"] == "message":
                await self.broadcast(json.loads(message["data"]))
```

**Estimated Lines**: 400-500

#### 1.2.3 Session Coordination (`multiplayer/session_coordinator.py`)

**Responsibilities**:
- Prevent conflicting NPC interactions
- Coordinate multi-player quests
- Handle race conditions

**Key Classes**:
```python
class SessionCoordinator:
    """Coordinates multi-player sessions and prevents conflicts"""

    async def acquire_npc_lock(self, npc_id: str, player_id: str, timeout: int = 30):
        """Acquire exclusive lock on NPC for conversation"""
        lock_key = f"npc:lock:{npc_id}"
        acquired = await self.redis.set(lock_key, player_id, nx=True, ex=timeout)
        return acquired

    async def is_npc_available(self, npc_id: str) -> bool:
        """Check if NPC is available for interaction"""
        lock_key = f"npc:lock:{npc_id}"
        return not await self.redis.exists(lock_key)

    async def coordinate_quest(self, quest_id: str, player_ids: List[str]):
        """Create multi-player quest coordination"""
        quest_key = f"quest:multi:{quest_id}"
        await self.redis.sadd(quest_key, *player_ids)
        await self.redis.expire(quest_key, 3600)  # 1 hour TTL
```

**Estimated Lines**: 300-400

### 1.3 API Endpoints

**New Endpoints**:

1. **WebSocket Connection** (`WS /elle/game/multiplayer/ws/{player_id}`)
   - Establish real-time connection
   - Receive world/NPC updates

2. **Get World State** (`GET /elle/game/multiplayer/world`)
   - Current world flags
   - Active players
   - NPC availability

3. **Acquire NPC Lock** (`POST /elle/game/multiplayer/npc/{npc_id}/lock`)
   - Request exclusive conversation lock
   - Returns lock status

4. **Broadcast Event** (`POST /elle/game/multiplayer/broadcast`)
   - Server-triggered world events
   - Quest completions affecting all players

### 1.4 Testing Strategy

**Unit Tests** (`tests/test_multiplayer.py`):
- Redis pub/sub message routing
- WebSocket connection management
- NPC lock acquisition/release
- Race condition handling

**Integration Tests**:
- 10 concurrent players
- NPC state synchronization
- Quest coordination across players

**Load Tests**:
- 100 concurrent WebSocket connections
- 1000 messages/second broadcast rate

**Estimated Test Lines**: 500-600

### 1.5 Estimated Effort

| Component | Lines | Complexity | Time |
|-----------|-------|------------|------|
| SharedWorldState | 400 | Medium | 1 week |
| WebSocketManager | 450 | High | 1.5 weeks |
| SessionCoordinator | 350 | High | 1 week |
| API Endpoints | 300 | Medium | 3 days |
| Tests | 550 | Medium | 1 week |
| **Total** | **2,050** | **High** | **4-5 weeks** |

---

## Feature 2: Advanced NPC Autonomy

**Goal**: NPCs autonomously pursue goals, follow daily routines, and exhibit emergent behavior.

### 2.1 Architecture Design

**Approach**: Hybrid GOAP (Goal-Oriented Action Planning) + Behavior Trees

**Why Hybrid?**:
- **GOAP**: Long-term goal planning (e.g., "Become wealthy merchant")
- **Behavior Trees**: Short-term reactive behavior (e.g., "Greet nearby player")

**Data Flow**:
```
NPC Goals → GOAP Planner → Action Sequence → Behavior Tree → Execution
                                                    ↓
                                            Emotional State Update
```

### 2.2 Core Components

#### 2.2.1 GOAP Planner (`autonomy/goap_planner.py`)

**Responsibilities**:
- Define NPC goals (wealth, reputation, knowledge)
- Plan action sequences to achieve goals
- Re-plan when world state changes

**Key Classes**:
```python
class GOAPPlanner:
    """Goal-Oriented Action Planning for NPCs"""

    def __init__(self):
        self.actions: List[GOAPAction] = []

    def plan(self, current_state: dict, goal: dict) -> List[GOAPAction]:
        """A* search to find action sequence achieving goal"""
        # A* heuristic: distance from goal state
        # Returns: [GoToMarket, BuyGoods, SellGoods, ...]

    def register_action(self, action: GOAPAction):
        """Register available NPC action"""
        self.actions.append(action)

class GOAPAction:
    """Single GOAP action with preconditions and effects"""

    name: str
    cost: float
    preconditions: Dict[str, Any]  # Required world state
    effects: Dict[str, Any]        # Changes to world state

    async def execute(self, npc: NPCState) -> bool:
        """Execute action, return success"""
        pass

# Example actions:
class GoToLocationAction(GOAPAction):
    name = "go_to_location"
    cost = 1.0
    preconditions = {}
    effects = {"location": "target_location"}

class BuyGoodsAction(GOAPAction):
    name = "buy_goods"
    cost = 5.0
    preconditions = {"location": "market", "gold": 50}
    effects = {"gold": -50, "has_goods": True}
```

**Estimated Lines**: 500-600

#### 2.2.2 Daily Routines (`autonomy/daily_routines.py`)

**Responsibilities**:
- Define time-based NPC schedules
- Override routines when goals require
- Blend with player interactions

**Key Classes**:
```python
class DailyRoutine:
    """24-hour schedule for NPC"""

    schedule: List[ScheduleEntry]

    def get_current_activity(self, time_of_day: str) -> ScheduleEntry:
        """Get NPC's current scheduled activity"""
        # time_of_day: "morning" (6-12), "afternoon" (12-18), etc.
        pass

    def is_interruptible(self, time_of_day: str) -> bool:
        """Can player interrupt NPC right now?"""
        activity = self.get_current_activity(time_of_day)
        return activity.interruptible

@dataclass
class ScheduleEntry:
    time_range: Tuple[int, int]  # (start_hour, end_hour)
    location: str
    activity: str
    interruptible: bool = True
    priority: int = 0

# Example: Innkeeper routine
innkeeper_routine = DailyRoutine(schedule=[
    ScheduleEntry((6, 8), "inn", "preparing_breakfast", interruptible=False),
    ScheduleEntry((8, 12), "inn", "serving_customers", interruptible=True),
    ScheduleEntry((12, 14), "market", "buying_supplies", interruptible=True),
    ScheduleEntry((14, 18), "inn", "serving_customers", interruptible=True),
    ScheduleEntry((18, 22), "inn", "evening_rush", interruptible=False),
    ScheduleEntry((22, 6), "inn_bedroom", "sleeping", interruptible=False),
])
```

**Estimated Lines**: 300-400

#### 2.2.3 Behavior Trees (`autonomy/behavior_trees.py`)

**Responsibilities**:
- Reactive short-term decision making
- Blend planned actions with reactive responses
- Handle unexpected events (player interactions, world events)

**Key Classes**:
```python
class BehaviorTree:
    """Behavior tree for NPC decision making"""

    root: BehaviorNode

    def tick(self, npc: NPCState, world_state: dict) -> NodeStatus:
        """Execute one tick of behavior tree"""
        return self.root.tick(npc, world_state)

class BehaviorNode:
    """Base behavior tree node"""

    def tick(self, npc: NPCState, world_state: dict) -> NodeStatus:
        raise NotImplementedError

class SequenceNode(BehaviorNode):
    """Execute children in sequence, fail if any fails"""
    children: List[BehaviorNode]

class SelectorNode(BehaviorNode):
    """Try children until one succeeds"""
    children: List[BehaviorNode]

class ConditionNode(BehaviorNode):
    """Check condition, return success/failure"""
    condition: Callable[[NPCState, dict], bool]

class ActionNode(BehaviorNode):
    """Execute action (e.g., talk to player, move to location)"""
    action: Callable[[NPCState], NodeStatus]

# Example: Innkeeper behavior tree
innkeeper_tree = BehaviorTree(root=SelectorNode([
    SequenceNode([  # Handle player interaction
        ConditionNode(lambda npc, world: world["player_nearby"]),
        ConditionNode(lambda npc, world: npc.routine.is_interruptible(world["time"])),
        ActionNode(lambda npc: greet_player(npc))
    ]),
    SequenceNode([  # Follow GOAP plan
        ConditionNode(lambda npc, world: npc.has_active_plan()),
        ActionNode(lambda npc: execute_next_planned_action(npc))
    ]),
    ActionNode(lambda npc: follow_daily_routine(npc))  # Default: routine
]))
```

**Estimated Lines**: 400-500

#### 2.2.4 Emergent Behavior Engine (`autonomy/emergent_behavior.py`)

**Responsibilities**:
- Detect patterns in NPC interactions
- Generate unscripted responses to world events
- Create "water cooler" conversations between NPCs

**Key Classes**:
```python
class EmergentBehaviorEngine:
    """Generates emergent NPC behavior from world events"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.event_history: List[WorldEvent] = []

    async def detect_patterns(self, npc: NPCState) -> Optional[EmergentGoal]:
        """Detect patterns in NPC's experience, suggest new goals"""
        # Example: If NPC sold goods successfully 5 times, create "Expand business" goal
        pass

    async def generate_gossip(self, npc1: NPCState, npc2: NPCState) -> Optional[str]:
        """Generate gossip between NPCs based on recent events"""
        recent_events = self.event_history[-10:]
        # Use LLM to generate contextual gossip
        pass

@dataclass
class WorldEvent:
    timestamp: datetime
    event_type: str  # "quest_completed", "npc_died", "player_action"
    participants: List[str]
    description: str
```

**Estimated Lines**: 350-450

### 2.3 Integration with Existing Systems

**Emotional State Integration**:
- GOAP actions affect emotional state (BuyGoods → happy)
- Emotional state affects action costs (angry NPCs avoid cooperation)

**Quest Integration**:
- NPCs can autonomously create quests based on goals
- Example: Merchant with "wealth" goal → offers delivery quests

**Conversation Integration**:
- Behavior tree interrupts routine for player conversations
- GOAP plans pause during multi-NPC conversations

### 2.4 API Endpoints

**New Endpoints**:

1. **Get NPC Schedule** (`GET /elle/game/npc/{npc_id}/schedule`)
   - Returns 24-hour routine
   - Current activity
   - Interruptibility status

2. **Set NPC Goal** (`POST /elle/game/npc/{npc_id}/goal`)
   - Manually assign goal
   - Triggers GOAP re-planning

3. **Get NPC Plan** (`GET /elle/game/npc/{npc_id}/plan`)
   - Current GOAP action sequence
   - Progress toward goal

4. **Get Emergent Events** (`GET /elle/game/emergent-events`)
   - Recent emergent behaviors
   - NPC-generated goals

### 2.5 Testing Strategy

**Unit Tests**:
- GOAP planner A* algorithm
- Daily routine time calculations
- Behavior tree node execution
- Emergent pattern detection

**Integration Tests**:
- Full NPC autonomy loop (GOAP → routine → behavior tree)
- Goal achievement scenarios
- Routine interruption by player

**Estimated Test Lines**: 450-550

### 2.6 Estimated Effort

| Component | Lines | Complexity | Time |
|-----------|-------|------------|------|
| GOAP Planner | 550 | Very High | 2 weeks |
| Daily Routines | 350 | Medium | 1 week |
| Behavior Trees | 450 | High | 1.5 weeks |
| Emergent Behavior | 400 | High | 1.5 weeks |
| API Endpoints | 250 | Medium | 3 days |
| Tests | 500 | High | 1 week |
| **Total** | **2,500** | **Very High** | **7-8 weeks** |

---

## Feature 3: Visual Workflow Builder

**Goal**: No-code quest and NPC design for non-technical users.

### 3.1 Architecture Design

**Tech Stack**:
- **Frontend**: React 18 + TypeScript
- **Drag-and-Drop**: react-flow (node-based editor)
- **UI Components**: shadcn/ui (Tailwind CSS)
- **Backend**: FastAPI (JSON export/import)
- **State Management**: Zustand

**Architecture**:
```
React Frontend (drag/drop) → JSON Schema → FastAPI → BigPlay Engine
                                  ↓
                          Exported Files (.json)
```

### 3.2 Core Components

#### 3.2.1 Quest Builder (`workflow_builder/frontend/QuestBuilder.tsx`)

**Features**:
- Drag-and-drop quest nodes
- Visual objective chaining
- Reward configuration UI
- Condition/branching support

**Node Types**:
- **Start**: Quest initiation
- **Objective**: Task to complete (collect, talk, defeat)
- **Branch**: Conditional split
- **Reward**: XP, gold, items, flags
- **End**: Quest completion

**Example React Component**:
```typescript
interface QuestNode {
  id: string;
  type: 'start' | 'objective' | 'branch' | 'reward' | 'end';
  data: {
    title?: string;
    description?: string;
    target?: number;
    conditions?: Condition[];
  };
  position: { x: number; y: number };
}

function QuestBuilder() {
  const [nodes, setNodes] = useState<QuestNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  const onNodeAdd = (type: string) => {
    const newNode: QuestNode = {
      id: `node_${Date.now()}`,
      type: type as any,
      data: {},
      position: { x: 100, y: 100 }
    };
    setNodes([...nodes, newNode]);
  };

  const onExport = async () => {
    const questJSON = {
      title: "My Quest",
      objectives: nodes.filter(n => n.type === 'objective'),
      rewards: nodes.filter(n => n.type === 'reward'),
      flow: edges
    };

    await fetch('/elle/game/workflow/quest/import', {
      method: 'POST',
      body: JSON.stringify(questJSON)
    });
  };

  return (
    <ReactFlow nodes={nodes} edges={edges}>
      <Toolbar onAddNode={onNodeAdd} />
      <ExportButton onClick={onExport} />
    </ReactFlow>
  );
}
```

**Estimated Lines**: 800-1000 (TypeScript)

#### 3.2.2 NPC Personality Designer (`workflow_builder/frontend/NPCDesigner.tsx`)

**Features**:
- Visual emotion baseline configuration
- Voice profile selection
- Daily routine timeline editor
- Relationship matrix (likes/dislikes other NPCs)

**UI Components**:
- **Emotion Sliders**: Valence, arousal, dominance, trust
- **Voice Picker**: Dropdown with voice preview
- **Timeline Editor**: 24-hour schedule drag-and-drop
- **Relationship Graph**: Node graph of NPC connections

**Example Component**:
```typescript
interface NPCPersonality {
  id: string;
  name: string;
  role: string;
  emotion_baseline: {
    valence: number;
    arousal: number;
    dominance: number;
    trust: number;
  };
  voice_profile: string;
  daily_routine: ScheduleEntry[];
  relationships: { [npcId: string]: number };  // -1.0 to 1.0
}

function NPCDesigner() {
  const [personality, setPersonality] = useState<NPCPersonality>({...});

  return (
    <div className="grid grid-cols-2 gap-4">
      <EmotionSliders
        values={personality.emotion_baseline}
        onChange={(values) => setPersonality({...personality, emotion_baseline: values})}
      />
      <VoicePicker
        selected={personality.voice_profile}
        onChange={(voice) => setPersonality({...personality, voice_profile: voice})}
      />
      <TimelineEditor
        schedule={personality.daily_routine}
        onChange={(schedule) => setPersonality({...personality, daily_routine: schedule})}
      />
      <RelationshipGraph
        npcId={personality.id}
        relationships={personality.relationships}
      />
    </div>
  );
}
```

**Estimated Lines**: 600-800 (TypeScript)

#### 3.2.3 Conversation Tree Editor (`workflow_builder/frontend/ConversationEditor.tsx`)

**Features**:
- Visual dialogue tree
- Conditional branching (player flags, reputation)
- Emotion tags per line
- Voice preview

**Node Types**:
- **NPC Line**: What NPC says
- **Player Choice**: Multiple choice for player
- **Condition Check**: Branch based on game state
- **Flag Set**: Change world/player flags

**Example Component**:
```typescript
interface DialogueNode {
  id: string;
  type: 'npc_line' | 'player_choice' | 'condition' | 'flag';
  data: {
    text?: string;
    tone?: string;
    choices?: string[];
    condition?: string;
    flag?: { key: string; value: any };
  };
}

function ConversationEditor() {
  const [dialogueTree, setDialogueTree] = useState<DialogueNode[]>([]);

  return (
    <ReactFlow nodes={dialogueTree}>
      <NodePalette types={['npc_line', 'player_choice', 'condition', 'flag']} />
      <PreviewPanel tree={dialogueTree} />
    </ReactFlow>
  );
}
```

**Estimated Lines**: 700-900 (TypeScript)

#### 3.2.4 Backend API (`workflow_builder/backend/workflow_api.py`)

**Responsibilities**:
- Import/export JSON schemas
- Validate workflow definitions
- Convert visual workflows to BigPlay config

**Endpoints**:

1. **Import Quest** (`POST /elle/game/workflow/quest/import`)
   - Input: JSON from quest builder
   - Output: Quest ID in database

2. **Export Quest** (`GET /elle/game/workflow/quest/{quest_id}/export`)
   - Output: JSON for quest builder

3. **Import NPC** (`POST /elle/game/workflow/npc/import`)
   - Input: JSON from NPC designer
   - Output: NPC ID in database

4. **Import Conversation** (`POST /elle/game/workflow/conversation/import`)
   - Input: JSON from conversation editor
   - Output: Conversation template ID

**Example API**:
```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/elle/game/workflow")

class QuestWorkflow(BaseModel):
    title: str
    description: str
    objectives: List[dict]
    rewards: List[dict]
    flow: List[dict]  # Node connections

@router.post("/quest/import")
async def import_quest(workflow: QuestWorkflow):
    """Convert visual workflow to Quest object"""
    quest = Quest(
        id=str(uuid.uuid4()),
        title=workflow.title,
        description=workflow.description,
        objectives=[
            QuestObjective(**obj) for obj in workflow.objectives
        ],
        reward=QuestReward(**workflow.rewards[0])
    )

    # Save to database
    await quest_manager.save_quest(quest)

    return {"quest_id": quest.id}
```

**Estimated Lines**: 400-500 (Python)

### 3.3 Testing Strategy

**Frontend Tests** (Jest + React Testing Library):
- Node drag-and-drop
- Edge creation
- JSON export/import
- Validation errors

**Backend Tests**:
- Schema validation
- Quest creation from JSON
- NPC creation from JSON
- Malformed input handling

**E2E Tests** (Playwright):
- Full quest creation flow
- Export → Import → Run in game
- NPC designer → Conversation with NPC

**Estimated Test Lines**: 400-500

### 3.4 Estimated Effort

| Component | Lines | Complexity | Time |
|-----------|-------|------------|------|
| Quest Builder (React) | 900 | High | 2 weeks |
| NPC Designer (React) | 700 | Medium | 1.5 weeks |
| Conversation Editor (React) | 800 | High | 2 weeks |
| Backend API | 450 | Medium | 1 week |
| Tests | 450 | Medium | 1 week |
| **Total** | **3,300** | **High** | **7-8 weeks** |

---

## Implementation Sequence

### Recommended Order

**Phase 4A: Foundation (Weeks 1-4)**
1. **Week 1-2**: Multiplayer - Redis + WebSocket infrastructure
2. **Week 3-4**: Multiplayer - Session coordination + testing

**Phase 4B: Autonomy (Weeks 5-9)**
3. **Week 5-6**: NPC Autonomy - GOAP planner
4. **Week 7-8**: NPC Autonomy - Daily routines + behavior trees
5. **Week 9**: NPC Autonomy - Emergent behavior + testing

**Phase 4C: Visual Tools (Weeks 10-14)**
6. **Week 10-11**: Workflow Builder - Quest builder frontend
7. **Week 12-13**: Workflow Builder - NPC designer + conversation editor
8. **Week 14**: Workflow Builder - Backend API + E2E testing

**Phase 4D: Integration & Polish (Weeks 15-16)**
9. **Week 15**: Cross-feature integration testing
10. **Week 16**: Documentation, demos, bug fixes

### Why This Order?

1. **Multiplayer First**: Foundation for all other features (shared world state needed)
2. **Autonomy Second**: Builds on existing emotion/quest systems
3. **Visual Tools Last**: Requires understanding of final data structures

### Parallelization Opportunities

- **After Week 4**: Autonomy and Visual Tools can proceed in parallel
- **Week 10-13**: Frontend and backend developers can work independently

---

## Integration with Existing Systems (v1.0)

### Multiplayer + Emotions

**Challenge**: Synchronize NPC emotional states across players
**Solution**:
```python
# Emotional state stored in Redis
await redis.hset(f"npc:{npc_id}", "valence", 0.7)
await redis.publish("npc:updates", json.dumps({
    "npc_id": npc_id,
    "emotion": "happy"
}))
```

### Multiplayer + Quests

**Challenge**: Multi-player quest coordination
**Solution**:
```python
# Track quest progress per player in Redis
quest_key = f"quest:{quest_id}:players"
await redis.hincrby(quest_key, player_id, 1)  # Increment progress

# When all players complete
if await redis.hlen(quest_key) >= required_players:
    await complete_multiplayer_quest(quest_id)
```

### Autonomy + Fine-Tuning

**Challenge**: Train models on emergent NPC behavior
**Solution**:
```python
# Export emergent dialogues to fine-tuning dataset
if emergent_dialogue.quality == DatasetQuality.HIGH:
    await fine_tuning_exporter.export_conversation(
        game_state=npc.state,
        player_message="",  # Emergent, no player trigger
        npc_response=emergent_dialogue.text,
        npc_id=npc.id
    )
```

### Visual Tools + All Features

**Challenge**: Visual builder must support all v1.0 + Phase 4 features
**Solution**:
- Quest builder includes multiplayer coordination options
- NPC designer includes GOAP goal configuration
- Conversation editor supports emergent dialogue patterns

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Redis scalability** | Medium | High | Use Redis Cluster, implement sharding |
| **WebSocket connection limits** | High | High | Use connection pooling, horizontal scaling |
| **GOAP planning complexity** | High | Medium | Limit action space, cache plans |
| **Frontend bundle size** | Medium | Low | Code splitting, lazy loading |
| **Breaking changes to v1.0** | Low | Critical | Comprehensive integration tests |
| **LLM latency for emergent behavior** | Medium | Medium | Pre-generate common patterns, cache |

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **WebSocket latency** | <50ms | 95th percentile message delivery |
| **GOAP planning time** | <100ms | Average plan generation |
| **Workflow export** | <1s | Quest JSON export |
| **Multiplayer broadcast** | <200ms | Redis pub → all clients |
| **Emergent behavior generation** | <500ms | LLM call + storage |

---

## Documentation Plan

### New Documentation Files

1. **MULTIPLAYER_GUIDE.md** (500 lines)
   - Redis setup
   - WebSocket client examples (Unity, Godot, Unreal)
   - Multi-player quest patterns

2. **NPC_AUTONOMY_GUIDE.md** (600 lines)
   - GOAP concepts
   - Daily routine configuration
   - Emergent behavior examples

3. **WORKFLOW_BUILDER_GUIDE.md** (400 lines)
   - UI walkthrough
   - JSON schema reference
   - Export/import workflow

### Updated Documentation

- **README.md**: Add Phase 4 features overview
- **BIGPLAY_ENGINE.md**: Update platform comparison, roadmap
- **API_REFERENCE.md**: Document new endpoints

---

## Success Metrics

### Quantitative

- ✅ 100 concurrent players in shared world (load test)
- ✅ 95% WebSocket uptime
- ✅ NPCs autonomously complete 10+ goals without bugs
- ✅ 50+ quests created via visual builder by beta testers
- ✅ <5% performance regression vs v1.0

### Qualitative

- ✅ Non-technical users can create quests in <10 minutes
- ✅ NPCs feel "alive" and unpredictable
- ✅ Multiplayer interactions enhance single-player experience
- ✅ Developer feedback: "Phase 4 is a game-changer"

---

## Timeline Summary

| Phase | Duration | Features | Lines of Code |
|-------|----------|----------|---------------|
| **4A: Multiplayer** | 4 weeks | Redis, WebSocket, Coordination | ~2,050 |
| **4B: Autonomy** | 5 weeks | GOAP, Routines, Emergent | ~2,500 |
| **4C: Visual Tools** | 5 weeks | Quest/NPC/Conversation Builders | ~3,300 |
| **4D: Integration** | 2 weeks | Testing, Docs, Polish | ~1,000 |
| **Total** | **16 weeks** | **All Features** | **~8,850** |

---

## Next Steps

1. **Review Roadmap**: Stakeholder approval of timeline and features
2. **Proof of Concept**: Build minimal Redis + WebSocket demo (1 week)
3. **Tech Stack Finalization**: Confirm React vs alternatives for frontend
4. **Team Assignment**: Assign developers to each phase
5. **Kick-Off**: Begin Phase 4A (Multiplayer Foundation)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Status**: Ready for Review

---

*Ready to build the future of LLM-native multiplayer gaming!* 🚀
