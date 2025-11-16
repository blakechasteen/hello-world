# HoloLoom LSP Screenshot & GIF Creation Guide

**Purpose**: Create visual assets for documentation, tutorials, and marketing
**Created**: 2025-11-16
**Target**: High-quality, readable screenshots showing all LSP features

---

## Quick Start

```bash
# Install screenshot tools
pip install asciinema pillow
apt-get install scrot maim ffmpeg

# Record VSCode interaction as GIF
asciinema rec -w 100 -h 30 demo.cast
agg demo.cast demo.gif

# Or use native tools
ffmpeg -f x11grab -i :0 -vf fps=10 -pix_fmt yuv420p demo.mp4
```

---

## Screenshots to Create

### 1. Code Completion Screenshot

**File**: `screenshots/01_completion.png`
**Dimensions**: 1920×1080 (crop to relevant area)
**Content**: PolicyEngine class with completion dropdown visible

#### Setup Steps
1. Open `demos/demo_lsp_features.py` in VS Code
2. Navigate to line with `policy = PolicyEngine(n_arms=8)`
3. Position cursor after `policy.`
4. Trigger completion: `Ctrl+Space`
5. Ensure completion dropdown shows at least 5 suggestions:
   ```
   select_action()
   update(action, reward)
   arm_rewards
   arm_counts
   ```

#### Screenshot Details
- **Font Size**: 14pt (readable)
- **Theme**: Dark mode (VS Code One Dark Pro)
- **Focus**: Completion dropdown should be centered
- **Context**: Show 2-3 lines above and below for context
- **Annotations** (optional):
  - Arrow pointing to top suggestion with label "⭐ Most Relevant"
  - Small note: "Results ranked by semantic relevance"

#### Capture Command
```bash
# Using VS Code screenshot feature
Ctrl+Shift+P > "Screenshot"
# Or using system tool:
scrot ~/screenshots/01_completion.png -d 2
```

#### Post-Processing
- Crop to 1200×400 for documentation
- Add subtle border (2px gray)
- Ensure text is crisp (no blur)
- Verify dropdown text is legible

---

### 2. Hover Information Screenshot

**File**: `screenshots/02_hover.png`
**Dimensions**: 1920×1080
**Content**: Hover tooltip with rich semantic context

#### Setup Steps
1. Open same file in VS Code
2. Hover cursor over `EXPLORATION_RATE = 0.1`
3. Wait for hover popover to appear (1-2 seconds)
4. Position to show full hover content
5. Hover should display:
   ```
   EXPLORATION_RATE: float = 0.1

   Semantic Context:
   - Parameter in bandit algorithms
   - Controls exploration rate
   - Typical range: [0.01, 0.2]
   ```

#### Screenshot Details
- **Popover Placement**: Right of variable (not overlapping code)
- **Background**: Translucent dark (VS Code hover style)
- **Text Color**: Light (contrasts with background)
- **Arrow Indicator**: Shows connection to variable
- **Context**: Show surrounding code for reference

#### Capture Command
```bash
# Move cursor to target, wait for hover, then capture
sleep 1 && scrot ~/screenshots/02_hover.png
```

#### Post-Processing
- Crop popover to reasonable size (600×300)
- Sharpen text for clarity
- Add subtle drop shadow
- Verify all text is readable

---

### 3. Go-to-Definition Screenshot

**File**: `screenshots/03_definition.png`
**Dimensions**: 1920×1080
**Content**: Split view showing jump from call to definition

#### Setup Steps
1. Open file with cursor on `create_memory_shard()` call
2. Show breadcrumb path at top: `demo_lsp_features.py > create_memory_shard`
3. Call: `shard = create_memory_shard(...)`
4. Definition is visible on screen below
5. Visual indicator shows the jump connection

#### Screenshot Details
- **Layout**: Show both call and definition in frame
- **Breadcrumb**: Visible at top showing: `[file] > [function]`
- **Highlighting**: Code on both sides highlighted similarly
- **Line Numbers**: Visible, showing line jump distance
- **Arrow/Visual**: Optional visual indicator of jump (diagram)

#### Capture Command
```bash
# Scroll to show both call and definition
code --goto demos/demo_lsp_features.py:142  # Or similar
sleep 1 && scrot ~/screenshots/03_definition.png
```

#### Alternative: Before/After Format
- **Left Panel**: "Before" - cursor on function call
- **Right Panel**: "After" - definition visible
- **Arrow**: Show jump path between them

#### Post-Processing
- Ensure both sections are visible and clear
- Highlight line numbers for context
- Add labels: "Function Call" and "Definition"
- Show jump distance (e.g., "+10 lines")

---

### 4. Symbol Search Screenshot

**File**: `screenshots/04_symbol_search.png`
**Dimensions**: 1920×1080
**Content**: Symbol search dialog with results

#### Setup Steps
1. Open file in VS Code
2. Trigger symbol search: `Ctrl+Shift+O` (document) or `Ctrl+T` (workspace)
3. Type search term: `"policy"`
4. Show search results dropdown:
   ```
   PolicyEngine (class) - line 23
   select_action (method) - line 35
   update (method) - line 48
   policy (variable) - line 142
   ```

#### Screenshot Details
- **Search Box**: Visible at top with query text
- **Results List**: Show top 5-7 results
- **Icons**: Show symbol types (class, method, variable)
- **Line Numbers**: Show where each symbol is defined
- **Highlighting**: Top result highlighted/selected
- **Keyboard Hint**: Show "↑↓ to navigate, Enter to select"

#### Capture Command
```bash
# Open symbol search and type
code demos/demo_lsp_features.py
# Ctrl+T (or Ctrl+Shift+O)
# Type "policy"
# Wait for results to populate
sleep 1 && scrot ~/screenshots/04_symbol_search.png
```

#### Post-Processing
- Crop dialog to show full results (800×600)
- Ensure search text is visible
- Verify line numbers readable
- Add annotation: "Results ranked by semantic relevance"

---

### 5. Multi-Editor Comparison

**File**: `screenshots/05_editors_side_by_side.png`
**Dimensions**: 3840×1080 (double width) or two separate 1920×1080
**Content**: Same file open in VS Code (left) and Neovim (right)

#### Setup Steps
1. Open file in VS Code (left half)
2. Open same file in terminal with Neovim (right half)
3. Position both windows side-by-side
4. Trigger same feature in each (e.g., completion)
5. Capture both showing identical results in different UI styles

#### Scenarios to Capture
- **Completion**: Same dropdown in VS Code and Neovim UI
- **Hover**: Hover tooltip in VS Code vs. Neovim popover
- **Symbol Search**: Dialog in VS Code vs. Telescope in Neovim
- **Diagnostics**: Error indicators in both editors

#### Screenshot Details
- **Alignment**: Code at same line in both editors
- **Feature**: Same LSP feature triggered in both
- **UI**: Different but equivalent presentations
- **Label**: Title at top: "LSP Features Work Across Editors"

#### Capture Command
```bash
# Arrange windows, then capture entire desktop
scrot ~/screenshots/05_editors_comparison.png

# Or use a compositing tool:
maim -i $(wmctrl -l | grep VS Code | awk '{print $1}') left.png
maim -i $(wmctrl -l | grep Neovim | awk '{print $1}') right.png
# Then paste side-by-side in Python/ImageMagick
```

#### Post-Processing
- Crop excess desktop/empty space
- Ensure both code sections are readable
- Use consistent font sizes (14pt)
- Add subtle divider line between editors
- Label each: "VS Code" and "Neovim"

---

### 6. Type Checking/Diagnostics

**File**: `screenshots/06_diagnostics.png`
**Dimensions**: 1920×1080
**Content**: Red squiggly underlines showing type errors

#### Setup Steps
1. Create file with intentional type errors
2. Trigger LSP diagnostic pass: `Ctrl+Shift+M` (VS Code)
3. Show errors in Problems panel:
   ```
   Error [1]: Type 'str' is not assignable to 'int'
   Error [2]: Missing required argument 'reward'
   Error [3]: Undefined name 'undefined_var'
   ```

#### Screenshot Details
- **Code**: Show lines with type errors
- **Squiggles**: Red underlines on problematic code
- **Error List**: Problems panel at bottom showing all errors
- **Icons**: Warning/error icons visible
- **Details**: Hover shows error explanation

#### Capture Command
```bash
# Open diagnostic file with errors
code demos/demo_lsp_features_with_errors.py
# Wait for diagnostics to run
sleep 2
# Trigger problems panel
# Ctrl+Shift+M
scrot ~/screenshots/06_diagnostics.png
```

#### Post-Processing
- Ensure errors are visible and clear
- Crop to show both code and problems panel
- Highlight error messages for readability
- Add annotation: "Type safety powered by semantic analysis"

---

### 7. Animation/GIF: Completion in Action

**File**: `animated/completion_demo.gif`
**Duration**: 10 seconds
**Frame Rate**: 10 fps (reduced for small file size)
**Content**: Type a few characters, see completions update in real-time

#### Recording Steps
1. Use `asciinema` for terminal or screen recording
2. Record typing: `policy.se`
3. Show completion dropdown updating
4. Select one option and insert
5. Total duration: ~8 seconds

#### asciinema Command
```bash
asciinema rec -w 120 -h 30 completion.cast
# Follow recording, type slowly and deliberately
# Ctrl+D to stop recording

# Convert to GIF
agg completion.cast completion.gif
```

#### Post-Processing
- Crop to relevant area (no extra terminal)
- Ensure typing is visible and clear
- Speed up/slow down for clarity
- Add subtitle: "Smart completion with semantic ranking"

---

### 8. Animation/GIF: Symbol Search

**File**: `animated/symbol_search.gif`
**Duration**: 8 seconds
**Frame Rate**: 10 fps
**Content**: Type search query, see results appear and update

#### Recording Steps
1. Open symbol search: `Ctrl+T`
2. Type slowly: "t-h-o-m-p"
3. Watch results filter in real-time
4. Show top results highlighted
5. Total duration: ~6 seconds

#### Recording Command
```bash
asciinema rec -w 100 -h 40 symbol_search.cast
# Ctrl+T
# Type "t h o m p" (with pauses)
# Ctrl+D

agg symbol_search.cast symbol_search.gif
```

#### Post-Processing
- Smooth animation (no stuttering)
- Clear text input visibility
- Results update shown clearly
- Add caption: "Semantic symbol search"

---

### 9. Diagnostic Animation: Multi-Editor Sync

**File**: `animated/multi_editor_sync.gif`
**Duration**: 12 seconds
**Frame Rate**: 10 fps
**Content**: Change code in one editor, see reflection in other

#### Recording Steps
1. Show VS Code on left, Neovim on right
2. Make a code change in VS Code
3. Show Neovim updates (LSP diagnostic)
4. Repeat for symbol addition
5. Show both editors stay in sync

#### Capture Command
```bash
# This requires split-screen recording
ffmpeg -f x11grab -i :0 -vf fps=10 -pix_fmt yuv420p multi_editor.mp4

# Or use asciinema on VS Code's integrated terminal
# while Neovim is visible
```

#### Post-Processing
- Smooth transitions between editors
- Clear focus on changes
- Timestamp shows real-time sync
- Add label: "Semantic changes propagate instantly"

---

## Tools & Setup

### Recommended Tools

| Task | Tool | Cost | Platform |
|------|------|------|----------|
| **Desktop Screenshot** | `scrot` or `maim` | Free | Linux |
| | `screencapture` | Free | macOS |
| | Snip & Sketch | Free | Windows |
| **Terminal Recording** | `asciinema` | Free | All |
| **Screen Recording** | OBS Studio | Free | All |
| | ffmpeg | Free | All |
| **GIF Creation** | `agg` (from asciinema) | Free | All |
| | ImageMagick `convert` | Free | All |
| **Image Editing** | GIMP | Free | All |
| | Photoshop | $$$ | macOS/Windows |
| **Annotation** | Krita | Free | All |

### Installation

#### Linux
```bash
# Screenshots
sudo apt-get install scrot maim

# Recording
sudo apt-get install ffmpeg obs-studio

# GIF creation
pip install asciinema agg pillow

# Image tools
sudo apt-get install imagemagick gimp
```

#### macOS
```bash
# Screenshots (built-in screencapture)
# Recording
brew install ffmpeg obs-studio
pip install asciinema

# GIF creation
brew install imagemagick
```

#### Windows
```bash
# Use Snip & Sketch (built-in)
# OBS Studio from website
# FFmpeg from chocolatey or direct download
choco install ffmpeg obs-studio
pip install asciinema
```

---

## Capture Techniques

### Technique 1: Simple Screenshot with scrot

```bash
# Immediate screenshot
scrot ~/screenshot.png

# Screenshot after 2-second delay (time to prepare)
scrot -d 2 ~/screenshot.png

# Screenshot of specific window
scrot -u ~/screenshot.png  # Active window

# Interactive selection
scrot -s ~/screenshot.png  # Draw selection box
```

### Technique 2: High-Quality Video Recording (OBS)

**OBS Settings for LSP Demos:**
```
Video:
  - Resolution: 1920×1080
  - FPS: 30 (or 60 for smooth motion)
  - Bitrate: 6000 kbps

Audio:
  - Bitrate: 128 kbps
  - Sample Rate: 48 kHz

Recording Format:
  - Container: MP4
  - Codec: H.264
  - Quality: High
```

**OBS Scene Setup:**
```
Scene 1: VS Code
  - Source: Window capture (VS Code)
  - Position: Center, full screen

Scene 2: Side-by-side editors
  - Source 1: Window capture (VS Code, left half)
  - Source 2: Window capture (Neovim, right half)

Scene 3: Terminal recording
  - Source: Terminal window
  - Font size: Large (readable)
```

### Technique 3: Terminal/TUI Recording with asciinema

```bash
# Basic recording
asciinema rec my_demo.cast

# Specify dimensions (width height)
asciinema rec -w 100 -h 30 my_demo.cast

# Limit idle time (skip long pauses)
asciinema rec --idle-time-limit 0.5 my_demo.cast

# Convert to GIF
agg my_demo.cast my_demo.gif

# Convert to MP4
asciinema convert my_demo.cast my_demo.mp4
```

### Technique 4: Crop & Annotate with ImageMagick

```bash
# Crop image
convert screenshot.png -crop 1200x400+360+300 cropped.png

# Add text annotation
convert screenshot.png -pointsize 24 -fill yellow \
  -annotate +50+50 "Code Completion" annotated.png

# Add border
convert screenshot.png -border 2 -bordercolor gray bordered.png

# Create side-by-side comparison
convert left.png right.png +append comparison.png

# Combine multiple images vertically
convert top.png bottom.png -append stacked.png
```

---

## Best Practices

### Font & Readability
- **Font Size**: Minimum 14pt for screenshots (readable at 1920×1080)
- **Font Family**: Monospace (Fira Code, Cascadia Code)
- **Contrast**: Ensure text contrasts with background
- **Line Height**: 1.6+ for readability
- **Test**: View at 50% zoom to check readability from distance

### Colors & Theme
- **Editor Theme**: Use consistent theme across all shots
- **Recommended**: Dark themes (One Dark Pro, Dracula)
- **Consistency**: Same theme in all editors shown
- **Contrast**: Ensure UI elements pop against background
- **Accessibility**: Test for color-blind friendly palette

### Composition
- **Focus**: Center important elements (completion, hover, etc.)
- **Context**: Show 2-3 lines above/below for reference
- **Whitespace**: Don't crop too tight; allow breathing room
- **Alignment**: Use grid (multiples of 10px) for cropping
- **Symmetry**: Crop to aspect ratios: 16:9, 4:3, or 1:1

### Performance Capture
- **Frame Rate**: 30fps minimum, 60fps for motion
- **Bitrate**: 6-8 Mbps for 1080p (good quality)
- **No Lag**: Test LSP latency before capturing
- **Smooth**: No stuttering or dropped frames
- **Audio**: Clear narration, no background noise

### Timing
- **Delays**: Leave 1-2 second pauses between actions
- **Hover**: Show hover tooltip for 2-3 seconds
- **Completion**: Let dropdown stay visible for 3+ seconds
- **Transition**: 0.5 second pause between different features
- **Total**: Each feature demo should be 30-60 seconds

---

## Directory Structure

```
screenshots/
├── 01_completion.png           # Code completion with dropdown
├── 02_hover.png               # Hover information tooltip
├── 03_definition.png          # Go-to-definition jump
├── 04_symbol_search.png       # Symbol search dialog
├── 05_editors_comparison.png  # VS Code vs Neovim
├── 06_diagnostics.png         # Type errors and diagnostics
│
animated/
├── completion_demo.gif        # Typing completion query
├── symbol_search.gif          # Symbol search results update
├── multi_editor_sync.gif      # Changes sync across editors
│
videos/
├── lsp_features_demo_raw.mp4  # Raw screen recording
├── lsp_features_demo_final.mp4 # With narration & editing
│
thumbnails/
├── youtube_thumbnail.png      # 1280×720 for YouTube
├── twitter_card.png           # 1200×675 for Twitter
├── social_square.png          # 1080×1080 for Instagram
```

---

## Publishing Guidelines

### Screenshot Placement

**In README.md:**
```markdown
## LSP Features in Action

### Code Completion
![Code completion dropdown]
(screenshots/01_completion.png)

Smart suggestions ranked by semantic relevance...

### Hover Information
![Hover tooltip]
(screenshots/02_hover.png)

Rich context from knowledge graphs...
```

**In Documentation:**
- Place screenshots in `/docs/images/` directory
- Use relative paths: `![alt](../images/01_completion.png)`
- Provide alt text for accessibility
- Keep file sizes < 500KB per image (use compression)

**In GitHub Issues/Discussions:**
- Drag and drop screenshots into comments
- Use Markdown image syntax
- Include captions explaining the feature

### Image Optimization

```bash
# Lossy compression (JPEG, good for photos)
convert screenshot.png -quality 85 optimized.jpg

# Lossless compression (PNG, best for UI)
convert screenshot.png -strip optimized.png
optipng -o2 optimized.png

# Batch optimization
for f in screenshots/*.png; do
  optipng -o2 "$f"
done

# Check file sizes
du -h screenshots/
```

### Social Media Sizing

| Platform | Optimal Size | Aspect Ratio |
|----------|--------------|--------------|
| **Twitter** | 1200×675 | 16:9 |
| **LinkedIn** | 1200×627 | 16:9 |
| **YouTube** | 1280×720 | 16:9 |
| **Instagram** | 1080×1080 | 1:1 |
| **Facebook** | 1200×628 | 16:9 |

```bash
# Resize for Twitter (1200×675)
convert screenshot.png -resize 1200x675 twitter_card.png

# Create square for Instagram
convert screenshot.png -gravity center -extent 1080x1080 instagram.png
```

---

## Checklist for Screenshot Capture

- [ ] **Environment Setup**
  - [ ] LSP server running and responsive
  - [ ] Editor configured with appropriate theme
  - [ ] Font size set to 14pt minimum
  - [ ] Demo files prepared and populated
  - [ ] Network latency acceptable (<50ms)

- [ ] **Completion Screenshot**
  - [ ] Cursor positioned after `.`
  - [ ] Completion dropdown visible
  - [ ] At least 5 suggestions shown
  - [ ] Top suggestion is most relevant
  - [ ] No errors or warnings visible

- [ ] **Hover Screenshot**
  - [ ] Cursor hovering over target symbol
  - [ ] Tooltip visible and complete
  - [ ] All text readable
  - [ ] No overlap with code
  - [ ] Rich context displayed

- [ ] **Definition Screenshot**
  - [ ] Call site visible
  - [ ] Definition visible on screen
  - [ ] Breadcrumb shows jump path
  - [ ] Line numbers visible
  - [ ] Both code sections clear

- [ ] **Symbol Search Screenshot**
  - [ ] Search dialog visible
  - [ ] Query text shown
  - [ ] Results populated
  - [ ] At least 5 results visible
  - [ ] Top result highlighted

- [ ] **Multi-Editor Comparison**
  - [ ] Both editors visible side-by-side
  - [ ] Same feature shown in both
  - [ ] Same code line visible
  - [ ] UI differences apparent
  - [ ] Both clearly labeled

- [ ] **Post-Processing**
  - [ ] Image cropped to relevant area
  - [ ] Text is sharp and readable
  - [ ] Colors are accurate
  - [ ] File size < 500KB
  - [ ] Aspect ratio correct

- [ ] **Animation (GIF) Quality**
  - [ ] No dropped frames
  - [ ] Smooth playback
  - [ ] Text readable at 100% size
  - [ ] Timing appropriate (not too fast/slow)
  - [ ] File size < 5MB for web

---

## Example Workflow

```bash
#!/bin/bash
# Complete screenshot capture workflow

DEMO_DIR="$HOME/projects/hololoom"
OUTPUT_DIR="$DEMO_DIR/demos/screenshots"

mkdir -p "$OUTPUT_DIR/raw"
mkdir -p "$OUTPUT_DIR/final"

# 1. Prepare environment
cd "$DEMO_DIR"
PYTHONPATH=. python -m HoloLoom.lsp.server &
LSP_PID=$!
sleep 2

# 2. Open editor
code demos/demo_lsp_features.py &
EDITOR_PID=$!
sleep 3

# 3. Capture completion (manual)
echo "Position cursor after 'policy.' and press Ctrl+Space"
echo "Press ENTER when ready to capture..."
read
scrot "$OUTPUT_DIR/raw/01_completion.png"

# 4. Capture hover
echo "Move cursor to 'EXPLORATION_RATE' and wait for hover..."
echo "Press ENTER when ready to capture..."
read
sleep 1
scrot "$OUTPUT_DIR/raw/02_hover.png"

# 5. Continue for other features...

# 6. Post-process
cd "$OUTPUT_DIR/raw"
for img in *.png; do
  optipng -o2 "$img"
  convert "$img" -crop 1200x400+360+300 "../final/$img"
done

# 7. Cleanup
kill $LSP_PID
kill $EDITOR_PID

echo "✓ Screenshots captured to $OUTPUT_DIR/final/"
```

---

## Quick Reference Commands

```bash
# Take screenshot after 2 second delay
scrot -d 2 screenshot.png

# Take screenshot of active window
scrot -u screenshot.png

# Interactive selection (draw box)
scrot -s screenshot.png

# Record desktop to video (10 seconds)
ffmpeg -f x11grab -i :0 -t 10 -pix_fmt yuv420p output.mp4

# Record with audio from microphone
ffmpeg -f x11grab -i :0 -f pulse -i default -t 10 output.mp4

# Convert video to GIF
ffmpeg -i video.mp4 -vf "fps=10,scale=1280:-1:flags=lanczos" output.gif

# Crop image to 1200×400
convert input.png -crop 1200x400+0+0 output.png

# Resize image to 1200×675 (Twitter)
convert input.png -resize 1200x675! output.png

# Add border to image
convert input.png -border 2 -bordercolor gray output.png

# Compare two images side-by-side
convert left.png right.png +append output.png
```

---

## Final Checklist

- [ ] All 7 main screenshots captured
- [ ] 3 GIF/animations recorded
- [ ] Images optimized (< 500KB each)
- [ ] All text readable at 100% zoom
- [ ] Consistent theme and fonts
- [ ] Proper aspect ratios maintained
- [ ] Social media sizes created
- [ ] Directory structure organized
- [ ] README updated with image references
- [ ] Backup of raw files created

Good luck with your visual assets! They'll make a huge difference in documentation quality and user engagement.
