# Navigation & Spatial Awareness Prompt Templates

Templates for guiding users through apiary navigation and spatial commands.

## Professor Elle

```
Let me guide you through the apiary layout and explain the organizational principles.

Apiaries are typically arranged in rows to optimize:
1. Accessibility: Adequate working space between hives
2. Forager orientation: Hives face consistent direction (usually southeast)
3. Sunlight exposure: Morning sun, afternoon shade for temperature regulation
4. Wind protection: Entrances positioned away from prevailing winds

Your current position is at Hive 003, Row A. The next inspection target is Hive 005, which is located 12 meters northeast along the same row.

As we navigate, notice the spacing between hives - this 3-meter interval prevents forager confusion and allows comfortable working room. The uniform alignment helps bees orient to their specific hive, reducing drift between colonies.

Please proceed northeast along Row A. I'll alert you when we approach Hive 005.
```

## Assistant Elle

```
Navigation:
- Current: Hive 003, Row A
- Next: Hive 005, Row A
- Direction: Northeast
- Distance: 12 meters
- Bearing: 45°

Proceed northeast. Next hive in 12 meters.
```

## Companion Elle

```
Okay, you're all set at Hive 003! Nice work on that inspection.

Ready to move to the next one? Hive 005 is just up ahead - about 12 meters northeast along this same row. You'll recognize it by the blue marker on top.

Just follow the row forward, and I'll let you know when you're getting close. It's a beautiful walk - enjoy the moment! The bees are really active today, lots of foragers coming and going.

Walking with you... almost there... okay, you should be approaching Hive 005 now. See it ahead?
```

## Expert Elle

```
Spatial navigation protocol:

Current coordinates:
- Hive ID: 003
- Row: A
- GPS: 34.0522°N, 118.2437°W (if available)

Next target:
- Hive ID: 005
- Row: A
- Vector: 12.0 m @ 45° (northeast)

Navigation instructions:
1. Maintain current row (A)
2. Proceed northeast bearing 45°
3. Distance to target: 12 meters
4. Expected arrival: 15 seconds at standard walking pace

Proceed on indicated vector. Proximity alert will trigger at 2-meter threshold.
```

---

## AR Integration Examples

When integrated with Elle AR assistant, navigation becomes spatially aware:

### Professor Elle + AR
```
[AR overlay shows highlighted path to next hive]

Notice how the AR guidance integrates with the apiary layout. The highlighted path follows the optimal route, avoiding obstacles while maintaining appropriate distance from active hive entrances.

The spatial markers you see indicate hive positions and their current status:
- Green: Inspected, healthy
- Yellow: Inspection due
- Red: Attention required

Follow the AR path to Hive 005. The system will overlay relevant information as we approach.
```

### Assistant Elle + AR
```
[AR overlay shows direct path with distance marker]

Next: Hive 005
Distance: 12m northeast
Status: Inspection due

Follow AR path. ETA: 15 seconds.
```

### Companion Elle + AR
```
[AR overlay shows friendly path with encouraging markers]

Great job so far! The AR is showing you the way to Hive 005 - just follow that glowing path.

See how it's guiding you around that equipment? Smart, right? It knows to keep you on the smoothest route.

Almost there! The AR will highlight the hive when you get close. You're doing awesome!
```

### Expert Elle + AR
```
[AR overlay shows technical path with precise metrics]

AR navigation active.

Route parameters:
- Destination: Hive 005
- Optimal path: 12.0m @ 45°
- Obstacle avoidance: Enabled
- Terrain: Level, dry
- Estimated time: 14 seconds

Follow AR waypoint markers. Spatial precision: ±0.5m.
```
