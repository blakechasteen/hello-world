# BDR Outbound Sequence - Visual Architecture

## High-Level Flow

```mermaid
graph TD
    Start([New Prospect]) --> Research[Day 0: Research Phase]
    Research --> Email[Day 1: Email Campaign]
    Email --> D1{Day 3: Opened?}
    D1 -->|Yes| LinkedIn[LinkedIn Connection]
    D1 -->|No| Retry[Retry New Subject]
    LinkedIn --> D2{Day 5: Accepted?}
    Retry --> D2
    D2 -->|Yes| WarmCall[Warm Call Script]
    D2 -->|No| ColdCall[Cold Call Script]
    WarmCall --> D3{Day 7: Engaged?}
    ColdCall --> D3
    D3 -->|Yes| ValueAdd[Value-Add Content]
    D3 -->|No| MultiThread[Multi-Thread Strategy]
    ValueAdd --> D4{Day 10: Posted?}
    MultiThread --> D4
    D4 -->|Yes| Comment[Engage Content]
    D4 -->|No| Breakup[Day 12: Breakup Email]
    Comment --> Breakup
    Breakup --> Analytics[Generate Report]
    Analytics --> Learning[Background Learning]
    Learning -.->|Update Priors| Email

    style Start fill:#e1f5ff
    style Research fill:#fff4e1
    style Email fill:#e1ffe1
    style LinkedIn fill:#e1ffe1
    style WarmCall fill:#e1ffe1
    style ValueAdd fill:#e1ffe1
    style Comment fill:#e1ffe1
    style Breakup fill:#ffe1e1
    style Analytics fill:#f0e1ff
    style Learning fill:#f0e1ff
```

## Detailed Agent Pipeline

```mermaid
graph LR
    subgraph "Day 0: Research"
        A1[HoloLoom Query<br/>RESEARCH mode] --> A2[Matryoshka<br/>Embedder]
        A2 --> A3[Memory Store<br/>Knowledge Graph]
    end

    subgraph "Day 1: Email"
        B1[Thompson<br/>Sampler] --> B2[Context<br/>Retriever]
        B2 --> B3[HoloLoom Query<br/>DIRECT mode]
        B3 --> B4[Safety<br/>Guardrails]
        B4 --> B5[Response<br/>Generator]
    end

    subgraph "Day 3: Branch"
        C1{Conditional<br/>Branch} --> C2[LinkedIn Note<br/>VERIFY mode]
        C1 --> C3[Retry Email<br/>DIRECT mode]
    end

    subgraph "Day 5: Call"
        D1{Conditional<br/>Branch} --> D2[Warm Script<br/>PLAN_EXECUTE]
        D1 --> D3[Cold Script<br/>PLAN_EXECUTE]
        D2 --> D4[Memory Store<br/>Call Outcome]
        D3 --> D4
    end

    subgraph "Day 7: Follow-up"
        E1{Conditional<br/>Branch} --> E2[Context<br/>Retriever]
        E2 --> E3[Value Content<br/>DIRECT mode]
        E1 --> E4[Multi-Query<br/>Find Contacts]
        E4 --> E5[Thompson<br/>Sampler]
    end

    subgraph "Day 10-12: Close"
        F1{Conditional<br/>Branch} --> F2[Comment<br/>DIRECT mode]
        F1 --> F3[Breakup Email<br/>DIRECT mode]
        F3 --> F4[Memory Store<br/>Sequence Outcome]
        F2 --> F3
    end

    subgraph "Background"
        G1[Recursive<br/>Refiner] -.->|Every 5 min| B1
        F4 -.->|Feedback| G1
    end

    A3 --> B1
    B5 --> C1
    C2 --> D1
    C3 --> D1
    D4 --> E1
    E3 --> F1
    E5 --> F1

    style A1 fill:#fff4e1
    style B1 fill:#e1f0ff
    style B4 fill:#ffe1e1
    style C1 fill:#f0f0f0
    style D1 fill:#f0f0f0
    style E1 fill:#f0f0f0
    style F1 fill:#f0f0f0
    style G1 fill:#f0e1ff
```

## Thompson Sampling Learning Loop

```mermaid
sequenceDiagram
    participant TS as Thompson Sampler
    participant WF as Workflow
    participant P as Prospect
    participant MB as Memory/Backend
    participant BL as Background Learner

    Note over TS: Day 1: Select Variant
    TS->>TS: Sample from Beta(α, β)
    TS->>WF: Selected: "funding_news" variant
    WF->>P: Send email with subject
    P-->>WF: Opens email ✓
    WF->>MB: Log: variant="funding_news", opened=true

    Note over BL: Every 5 minutes
    BL->>MB: Query outcomes
    MB-->>BL: funding_news: 46 opens, 12 no-opens
    BL->>TS: Update priors: α=46, β=12

    Note over TS: Day 1 (Next Prospect)
    TS->>TS: E[funding_news] = 46/(46+12) = 0.79
    TS->>TS: E[tech_stack] = 38/(38+15) = 0.72
    TS->>TS: Allocate 70% to funding_news
    TS->>WF: Selected: "funding_news" (exploit)

    Note over TS: Day 1 (Another Prospect)
    TS->>TS: 30% exploration sample
    TS->>WF: Selected: "hiring_signal" (explore)
    WF->>P: Send email
    P-->>WF: No open ✗
    WF->>MB: Log: variant="hiring_signal", opened=false

    Note over BL: Update cycle
    BL->>MB: Query outcomes
    MB-->>BL: hiring_signal: 22 opens, 9 no-opens
    BL->>TS: Update priors: α=22, β=9
```

## Conditional Branching Decision Tree

```mermaid
graph TD
    Start([Prospect Enters Sequence]) --> Research[Day 0: Research]
    Research --> Email[Day 1: Send Email]

    Email --> D1{Day 3<br/>Opened?}
    D1 -->|Yes ✓| LI[Send LinkedIn<br/>Connection]
    D1 -->|No ✗| Retry[Try Different<br/>Subject]

    LI --> D2{Day 5<br/>Accepted?}
    Retry --> D2

    D2 -->|Yes ✓| Warm[Warm Call<br/>'I sent you note...']
    D2 -->|No ✗| Cold[Cold Call<br/>'Pattern interrupt']

    Warm --> CallOut[Make Call]
    Cold --> CallOut

    CallOut --> D3{Day 7<br/>Any Engage?}

    D3 -->|Yes ✓| Value[Send Value<br/>Content]
    D3 -->|No ✗| Multi[Find Other<br/>Contact]

    Value --> D4{Day 10<br/>Posted?}
    Multi --> NewContact[Start Fresh<br/>Sequence]

    D4 -->|Yes ✓| Comment[Thoughtful<br/>Comment]
    D4 -->|No ✗| Breakup[Day 12<br/>Breakup Email]

    Comment --> Breakup
    Breakup --> D5{Reply?}

    D5 -->|'Let's talk'| Meeting[Book Meeting ✓]
    D5 -->|'Not a fit'| Archive[Archive Forever]
    D5 -->|No reply| Nurture[Add to Nurture<br/>Campaign]

    NewContact -.->|Reset| Email

    style Start fill:#e1f5ff
    style Email fill:#e1ffe1
    style LI fill:#e1ffe1
    style Warm fill:#e1ffe1
    style Value fill:#e1ffe1
    style Comment fill:#e1ffe1
    style Meeting fill:#c8ffc8
    style Archive fill:#ffc8c8
    style Nurture fill:#fff4c8
    style D1 fill:#f0f0f0
    style D2 fill:#f0f0f0
    style D3 fill:#f0f0f0
    style D4 fill:#f0f0f0
    style D5 fill:#f0f0f0
```

## Agent Type Distribution

```mermaid
pie title "26 Agents by Type"
    "HoloLoom Query (Research/Direct/Verify/Plan)" : 9
    "Thompson Sampler (A/B Selection)" : 2
    "Conditional Branch (Decisions)" : 5
    "Memory Store (Persistence)" : 3
    "Context Retriever (Knowledge Graph)" : 2
    "Response Generator (Formatting)" : 2
    "Safety Guardrails (Compliance)" : 1
    "Matryoshka Embedder (Encoding)" : 1
    "Recursive Refiner (Learning)" : 1
```

## Data Flow: Single Prospect Journey

```mermaid
graph TB
    subgraph "Input"
        I1[Prospect Data<br/>Name, Company, Persona]
    end

    subgraph "Knowledge Graph"
        KG1[(Research<br/>News, Tech, Pain)]
        KG2[(Email<br/>Opens, Clicks)]
        KG3[(LinkedIn<br/>Accepted, Posts)]
        KG4[(Call<br/>Outcomes)]
        KG5[(Content<br/>Assets)]
    end

    subgraph "Thompson Sampling"
        TS1[Subject Line<br/>Priors α, β]
        TS2[Call Time<br/>Priors α, β]
        TS3[Multi-Thread<br/>Priors α, β]
    end

    subgraph "Output"
        O1[8 Touchpoints<br/>Emails, LinkedIn, Call]
        O2[Analytics Report<br/>Metrics, Insights]
        O3[Updated Priors<br/>For Next Prospect]
    end

    I1 --> KG1
    KG1 --> TS1
    TS1 --> O1
    O1 --> KG2
    KG2 --> KG3
    KG3 --> TS2
    TS2 --> KG4
    KG4 --> TS3
    TS3 --> KG5
    KG5 --> O2
    O2 --> O3
    O3 -.->|Feedback Loop| TS1

    style I1 fill:#e1f5ff
    style KG1 fill:#fff4e1
    style KG2 fill:#fff4e1
    style KG3 fill:#fff4e1
    style KG4 fill:#fff4e1
    style KG5 fill:#fff4e1
    style TS1 fill:#e1f0ff
    style TS2 fill:#e1f0ff
    style TS3 fill:#e1f0ff
    style O1 fill:#e1ffe1
    style O2 fill:#f0e1ff
    style O3 fill:#f0e1ff
```

## Performance Timeline

```mermaid
gantt
    title BDR Sequence Timeline (12 Days)
    dateFormat YYYY-MM-DD

    section Research
    Deep Research (RESEARCH mode, 5 queries)    :r1, 2025-11-05, 1d

    section Email Campaign
    Thompson Sampling (subject selection)       :e1, 2025-11-06, 1d
    Send Email (DIRECT mode)                    :e2, 2025-11-06, 1d

    section Day 3 Branch
    Check: Email Opened?                        :d1, 2025-11-08, 1d
    LinkedIn OR Retry                           :d2, 2025-11-08, 1d

    section Day 5 Call
    Check: LinkedIn Accepted?                   :c1, 2025-11-10, 1d
    Warm OR Cold Call (PLAN_EXECUTE)           :c2, 2025-11-10, 1d

    section Day 7 Follow-up
    Check: Any Engagement?                      :f1, 2025-11-12, 1d
    Value Content OR Multi-Thread              :f2, 2025-11-12, 1d

    section Day 10 Social
    Check: Posted Recently?                     :s1, 2025-11-15, 1d
    Engage Content OR Skip                     :s2, 2025-11-15, 1d

    section Day 12 Close
    Breakup Email                              :b1, 2025-11-17, 1d
    Generate Analytics Report                  :b2, 2025-11-17, 1d

    section Background
    Learning Loop (every 5 min)                :bg, 2025-11-05, 12d
```

## Cost Breakdown (per 100 Prospects)

```mermaid
graph LR
    subgraph "Compute Costs"
        C1[Research: 5s × 100<br/>= 500s @ $0.001/s<br/>= $0.50]
        C2[Email Gen: 1.5s × 100<br/>= 150s @ $0.001/s<br/>= $0.15]
        C3[Call Scripts: 4s × 100<br/>= 400s @ $0.001/s<br/>= $0.40]
        C4[Follow-ups: 1.5s × 300<br/>= 450s @ $0.001/s<br/>= $0.45]
    end

    subgraph "Infrastructure"
        I1[Neo4j + Qdrant<br/>Docker Containers<br/>$50/month]
        I2[FastAPI Server<br/>AWS t3.medium<br/>$30/month]
    end

    subgraph "Total"
        T1[Compute: $1.50/100<br/>Infrastructure: $80/month<br/><br/>Cost per Prospect: $0.015<br/>vs Manual: $25]
    end

    C1 --> T1
    C2 --> T1
    C3 --> T1
    C4 --> T1
    I1 --> T1
    I2 --> T1

    style C1 fill:#e1ffe1
    style C2 fill:#e1ffe1
    style C3 fill:#e1ffe1
    style C4 fill:#e1ffe1
    style I1 fill:#fff4e1
    style I2 fill:#fff4e1
    style T1 fill:#f0e1ff
```

## ROI Comparison

```mermaid
graph TD
    subgraph "Manual BDR (No HoloLoom)"
        M1[Time: 50 min/prospect]
        M2[Capacity: 200/month]
        M3[Cost: $25/prospect]
        M4[Meetings: 10/month]
        M5[Cost per Meeting: $500]
    end

    subgraph "Automated BDR (HoloLoom)"
        A1[Time: 21 min/prospect]
        A2[Capacity: 500/month]
        A3[Cost: $11/prospect]
        A4[Meetings: 25/month]
        A5[Cost per Meeting: $220]
    end

    subgraph "Improvement"
        I1[2.5x More Prospects<br/>500 vs 200]
        I2[2.5x More Meetings<br/>25 vs 10]
        I3[56% Lower Cost<br/>$220 vs $500]
        I4[Better Personalization<br/>Thompson Sampling]
    end

    M1 --> M2 --> M3 --> M4 --> M5
    A1 --> A2 --> A3 --> A4 --> A5
    M5 --> I3
    A5 --> I3
    M2 --> I1
    A2 --> I1
    M4 --> I2
    A4 --> I2
    A5 --> I4

    style M1 fill:#ffc8c8
    style M2 fill:#ffc8c8
    style M3 fill:#ffc8c8
    style M4 fill:#ffc8c8
    style M5 fill:#ffc8c8
    style A1 fill:#c8ffc8
    style A2 fill:#c8ffc8
    style A3 fill:#c8ffc8
    style A4 fill:#c8ffc8
    style A5 fill:#c8ffc8
    style I1 fill:#e1f0ff
    style I2 fill:#e1f0ff
    style I3 fill:#e1f0ff
    style I4 fill:#e1f0ff
```

---

## Legend

**Node Colors**:
- 🔵 Blue (`#e1f5ff`): Start/Input nodes
- 🟡 Yellow (`#fff4e1`): Research/Knowledge nodes
- 🟢 Green (`#e1ffe1`): Action/Execution nodes
- 🔴 Red (`#ffe1e1`): Safety/Compliance nodes
- 🟣 Purple (`#f0e1ff`): Learning/Analytics nodes
- ⚪ Gray (`#f0f0f0`): Decision/Conditional nodes

**Agent Types**:
- **HoloLoom Query**: Multi-modal reasoning (RESEARCH/DIRECT/VERIFY/PLAN_EXECUTE)
- **Thompson Sampler**: Bayesian A/B testing
- **Conditional Branch**: If/else logic
- **Memory Store**: Knowledge graph persistence
- **Safety Guardrails**: Compliance gating
- **Recursive Refiner**: Background learning loop

**Workflow Execution**:
- Solid lines (→): Synchronous execution
- Dashed lines (-.->): Asynchronous/background
- Decision diamonds (◇): Conditional branches

---

## Usage

These diagrams are rendered with Mermaid. To view:

1. **GitHub**: Automatically renders in `.md` files
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Copy/paste into https://mermaid.live/

**Export Options**:
- PNG: Use Mermaid Live Editor → Export as PNG
- SVG: Use `mmdc` CLI tool (Mermaid CLI)
- PDF: Use Markdown → PDF converter with Mermaid support