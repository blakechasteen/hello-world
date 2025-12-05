"""
Visual Calendar Generator

Creates HTML yearly calendar view with:
- Month-by-month grid
- Color-coded events by type
- Hover tooltips with event details
- Production timeline view

Created: 2025-11-16
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import calendar

from elle.calendar.yearly_calendar import YearlyCalendar, CalendarEvent, EventType


# Event type colors (following Tufte principles - subtle, meaningful)
EVENT_COLORS = {
    EventType.PLANTING: "#4CAF50",      # Green
    EventType.HARVEST: "#FF9800",       # Orange
    EventType.PRODUCTION: "#2196F3",    # Blue
    EventType.MARKET: "#9C27B0",        # Purple
    EventType.MAINTENANCE: "#795548",   # Brown
    EventType.REGULATORY: "#F44336",    # Red
    EventType.FINANCIAL: "#FFC107",     # Amber
    EventType.GENERAL: "#607D88",       # Blue Grey
}

EVENT_ICONS = {
    EventType.PLANTING: "🌱",
    EventType.HARVEST: "🌾",
    EventType.PRODUCTION: "⚗️",
    EventType.MARKET: "🛒",
    EventType.MAINTENANCE: "🔧",
    EventType.REGULATORY: "📋",
    EventType.FINANCIAL: "💰",
    EventType.GENERAL: "📌",
}


def render_yearly_calendar_html(yearly_calendar: YearlyCalendar) -> str:
    """Render full yearly calendar as HTML"""

    html_parts = []

    # Header
    html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Farm Calendar - {farm_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
            line-height: 1.4;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .subtitle {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .calendar-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        @media (max-width: 1200px) {{
            .calendar-grid {{ grid-template-columns: repeat(3, 1fr); }}
        }}
        @media (max-width: 900px) {{
            .calendar-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        @media (max-width: 600px) {{
            .calendar-grid {{ grid-template-columns: 1fr; }}
        }}
        .month-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .month-header {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .month-days {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 2px;
        }}
        .day-label {{
            text-align: center;
            font-size: 10px;
            font-weight: 600;
            color: #95a5a6;
            padding: 4px 0;
        }}
        .day-cell {{
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            border-radius: 4px;
            position: relative;
            cursor: pointer;
        }}
        .day-cell:hover {{
            background: #ecf0f1;
        }}
        .day-cell.today {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        .day-cell.has-event {{
            background: #e3f2fd;
            font-weight: 500;
        }}
        .day-cell.has-critical {{
            background: #ffebee;
            font-weight: 600;
        }}
        .event-dot {{
            position: absolute;
            bottom: 2px;
            left: 50%;
            transform: translateX(-50%);
            width: 4px;
            height: 4px;
            border-radius: 50%;
        }}
        .event-list {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .event-list h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .event-item {{
            padding: 12px;
            margin-bottom: 10px;
            border-left: 4px solid #bdc3c7;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .event-item.critical {{ border-left-color: #e74c3c; }}
        .event-item.high {{ border-left-color: #f39c12; }}
        .event-item.medium {{ border-left-color: #3498db; }}
        .event-name {{
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .event-meta {{
            font-size: 11px;
            color: #7f8c8d;
        }}
        .event-notes {{
            font-size: 12px;
            color: #555;
            margin-top: 6px;
        }}
        .legend {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .legend-title {{
            font-weight: 600;
            margin-bottom: 10px;
        }}
        .legend-items {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 {farm_name} - Yearly Calendar</h1>
        <div class="subtitle">{current_date}</div>
""".format(
        farm_name=yearly_calendar.farm_name,
        current_date=datetime.now().strftime("%B %d, %Y")
    ))

    # Legend
    html_parts.append("""
        <div class="legend">
            <div class="legend-title">Event Types</div>
            <div class="legend-items">
""")
    for event_type, color in EVENT_COLORS.items():
        icon = EVENT_ICONS.get(event_type, "")
        html_parts.append(f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {color};"></div>
                    <span>{icon} {event_type.value.title()}</span>
                </div>
""")
    html_parts.append("""
            </div>
        </div>
""")

    # Upcoming events
    upcoming = yearly_calendar.get_upcoming_events(30)
    if upcoming:
        html_parts.append("""
        <div class="event-list">
            <h2>📅 Upcoming Events (Next 30 Days)</h2>
""")
        for event in upcoming[:10]:
            priority_class = event.priority.value
            icon = EVENT_ICONS.get(event.type, "")
            html_parts.append(f"""
            <div class="event-item {priority_class}">
                <div class="event-name">{icon} {event.name}</div>
                <div class="event-meta">
                    {event.start_date.strftime("%B %d, %Y")} • {event.days_until} days away • {event.type.value.title()}
                </div>
                {f'<div class="event-notes">{event.notes}</div>' if event.notes else ''}
            </div>
""")
        html_parts.append("""
        </div>
""")

    # Active events
    active = yearly_calendar.get_active_events()
    if active:
        html_parts.append("""
        <div class="event-list">
            <h2>✅ Active Now</h2>
""")
        for event in active:
            icon = EVENT_ICONS.get(event.type, "")
            duration_str = f"{event.days_remaining} days remaining" if event.days_remaining else "Today only"
            html_parts.append(f"""
            <div class="event-item">
                <div class="event-name">{icon} {event.name}</div>
                <div class="event-meta">
                    {duration_str} • {event.type.value.title()}
                </div>
                {f'<div class="event-notes">{event.notes}</div>' if event.notes else ''}
            </div>
""")
        html_parts.append("""
        </div>
""")

    # Monthly calendar grid
    html_parts.append("""
        <div class="calendar-grid">
""")

    year = datetime.now().year
    today = datetime.now().date()

    for month_num in range(1, 13):
        month_name = calendar.month_name[month_num]
        cal = calendar.monthcalendar(year, month_num)

        html_parts.append(f"""
            <div class="month-card">
                <div class="month-header">{month_name} {year}</div>
                <div class="month-days">
""")

        # Day labels
        for day_name in ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa']:
            html_parts.append(f'                    <div class="day-label">{day_name}</div>\n')

        # Days
        month_events = yearly_calendar.get_events_by_month(month_num)

        for week in cal:
            for day in week:
                if day == 0:
                    html_parts.append('                    <div class="day-cell"></div>\n')
                else:
                    date = datetime(year, month_num, day).date()
                    is_today = date == today

                    # Check for events on this day
                    day_events = [e for e in month_events if e.month == month_num and e.day == day]
                    has_critical = any(e.priority.value == "critical" for e in day_events)

                    classes = ["day-cell"]
                    if is_today:
                        classes.append("today")
                    elif day_events:
                        if has_critical:
                            classes.append("has-critical")
                        else:
                            classes.append("has-event")

                    # Event dot color (use first event's type)
                    dot_html = ""
                    if day_events:
                        color = EVENT_COLORS.get(day_events[0].type, "#bdc3c7")
                        dot_html = f'<div class="event-dot" style="background-color: {color};"></div>'

                    html_parts.append(f'                    <div class="{" ".join(classes)}">{day}{dot_html}</div>\n')

        html_parts.append("""
                </div>
            </div>
""")

    html_parts.append("""
        </div>
    </div>
</body>
</html>
""")

    return "".join(html_parts)


def main():
    """Generate and save yearly calendar HTML"""
    from pathlib import Path

    calendar = YearlyCalendar()
    html = render_yearly_calendar_html(calendar)

    output_path = Path("demos/output/yearly_calendar.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"✓ Yearly calendar saved to {output_path}")
    print(f"✓ Open file://{output_path.absolute()} in your browser")


if __name__ == "__main__":
    main()
