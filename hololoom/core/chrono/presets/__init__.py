"""
Calendar presets — ready-made task sets for specific domains.

Each preset is a function that populates an AgentCalendar with
domain-appropriate tasks. The calendar remains generic; the preset
just provides the initial task list.

Usage:
    from hololoom.core.chrono.calendar import AgentCalendar
    from hololoom.core.chrono.presets.farm import load_farm_calendar

    cal = AgentCalendar(agent_id="farm-agent")
    load_farm_calendar(cal)
"""
