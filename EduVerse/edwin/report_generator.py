"""
EdWIN Report Generator

Generate beautiful progress reports in multiple formats (HTML, PDF, JSON).

**Features**:
- Daily, weekly, and monthly reports
- Subject-by-subject breakdown
- Learning velocity trends
- Knowledge gap analysis
- Engagement metrics
- Visual charts and graphs
- Export to HTML, PDF, JSON

**Implementation Date**: November 15, 2025

Usage:
    from EduVerse.edwin.report_generator import ReportGenerator

    generator = ReportGenerator(analytics, curriculum_kg)
    html = await generator.generate_weekly_report_html(student_model)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import base64
import io

from .analytics import LearningAnalytics
from .student_model import EdWINStudentModel
from .curriculum_graph import EdWINKnowledgeGraph
from EduVerse.education.curriculum import Subject

# Try to import reportlab for PDF generation (optional)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️  ReportLab not available - PDF generation disabled")


class ReportGenerator:
    """
    Report Generator

    Generates comprehensive progress reports in multiple formats.
    """

    def __init__(
        self,
        analytics: LearningAnalytics,
        curriculum_kg: EdWINKnowledgeGraph
    ):
        """
        Initialize Report Generator

        Args:
            analytics: Learning analytics engine
            curriculum_kg: Curriculum knowledge graph
        """
        self.analytics = analytics
        self.curriculum_kg = curriculum_kg

    # ========================================================================
    # Daily Reports
    # ========================================================================

    async def generate_daily_report_html(
        self,
        student_model: EdWINStudentModel,
        date: Optional[datetime] = None
    ) -> str:
        """
        Generate daily report in HTML format

        Args:
            student_model: Student model
            date: Optional date (defaults to today)

        Returns:
            HTML string
        """
        date = date or datetime.utcnow()
        date_str = date.strftime("%B %d, %Y")

        # Get today's interactions
        today = date.date()
        interactions = [
            i for i in self.analytics.interaction_history.get(student_model.student_id, [])
            if datetime.fromisoformat(i["timestamp"]).date() == today
        ]

        # Calculate metrics
        objectives_attempted = len(set(i.get("objective_id") for i in interactions if i.get("objective_id")))
        objectives_mastered = sum(1 for i in interactions if i.get("mastered", False))
        questions_asked = sum(1 for i in interactions if i.get("type") == "question")
        time_spent = len(interactions) * 5  # Assume 5 min per interaction
        xp_earned = sum(i.get("xp_gained", 0) for i in interactions)
        streak = student_model.player_model.stats.streak_days

        # Success rate
        successes = sum(1 for i in interactions if i.get("success", False))
        success_rate = (successes / len(interactions) * 100) if interactions else 0

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Daily Report - {student_model.name}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f9fafb;
                    padding: 20px;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .header h1 {{
                    font-size: 32px;
                    margin-bottom: 10px;
                }}
                .header p {{
                    opacity: 0.9;
                }}
                .content {{
                    padding: 40px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .metric-card {{
                    background: #f9fafb;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }}
                .metric-card .label {{
                    color: #9ca3af;
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 8px;
                }}
                .metric-card .value {{
                    color: #1f2937;
                    font-size: 32px;
                    font-weight: bold;
                }}
                .metric-card .unit {{
                    color: #6b7280;
                    font-size: 14px;
                    margin-top: 4px;
                }}
                .streak-banner {{
                    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                    color: white;
                    border-radius: 8px;
                    padding: 30px;
                    text-align: center;
                    margin-bottom: 40px;
                }}
                .streak-banner .emoji {{
                    font-size: 48px;
                    margin-bottom: 10px;
                }}
                .streak-banner .value {{
                    font-size: 48px;
                    font-weight: bold;
                }}
                .progress-section {{
                    margin-bottom: 30px;
                }}
                .progress-section h2 {{
                    color: #1f2937;
                    margin-bottom: 15px;
                }}
                .progress-bar {{
                    background: #e5e7eb;
                    border-radius: 10px;
                    height: 20px;
                    overflow: hidden;
                }}
                .progress-fill {{
                    background: linear-gradient(90deg, #10b981 0%, #059669 100%);
                    height: 100%;
                    transition: width 0.3s ease;
                }}
                .footer {{
                    background: #f9fafb;
                    padding: 20px;
                    text-align: center;
                    color: #6b7280;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Report</h1>
                    <p>{student_model.name} - {date_str}</p>
                </div>

                <div class="content">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="label">Objectives</div>
                            <div class="value">{objectives_mastered}/{objectives_attempted}</div>
                            <div class="unit">Mastered</div>
                        </div>

                        <div class="metric-card">
                            <div class="label">XP Earned</div>
                            <div class="value" style="color: #10b981;">{xp_earned}</div>
                            <div class="unit">Experience</div>
                        </div>

                        <div class="metric-card">
                            <div class="label">Time Spent</div>
                            <div class="value" style="color: #3b82f6;">{time_spent}</div>
                            <div class="unit">Minutes</div>
                        </div>

                        <div class="metric-card">
                            <div class="label">Questions</div>
                            <div class="value" style="color: #f59e0b;">{questions_asked}</div>
                            <div class="unit">Asked</div>
                        </div>
                    </div>

                    {f'''
                    <div class="streak-banner">
                        <div class="emoji">🔥</div>
                        <div class="value">{streak} Days</div>
                        <p style="margin-top: 10px; opacity: 0.9;">Current Streak</p>
                    </div>
                    ''' if streak > 0 else ''}

                    <div class="progress-section">
                        <h2>Success Rate</h2>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {success_rate}%;"></div>
                        </div>
                        <p style="color: #6b7280; margin-top: 10px; font-size: 14px;">
                            {success_rate:.0f}% of attempts successful
                        </p>
                    </div>
                </div>

                <div class="footer">
                    <p>Generated by EdWIN AI Tutor | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    # ========================================================================
    # Weekly Reports
    # ========================================================================

    async def generate_weekly_report_html(
        self,
        student_model: EdWINStudentModel
    ) -> str:
        """
        Generate weekly report in HTML format

        Args:
            student_model: Student model

        Returns:
            HTML string
        """
        # Get analytics
        report = self.analytics.generate_progress_report(student_model)
        velocity = self.analytics.calculate_learning_velocity(student_model, window_days=7)
        engagement = self.analytics.calculate_engagement(student_model, window_days=7)

        # Week ending
        week_ending = datetime.utcnow().strftime("%B %d, %Y")

        # Subject progress
        subject_rows = []
        for subject_name, data in report["subject_progress"].items():
            percentage = data["percentage"]
            color = "#10b981" if percentage >= 70 else "#f59e0b" if percentage >= 40 else "#ef4444"

            subject_rows.append(f"""
                <tr>
                    <td style="padding: 15px; border-bottom: 1px solid #e5e7eb;">{subject_name}</td>
                    <td style="padding: 15px; border-bottom: 1px solid #e5e7eb;">
                        <div style="background: #e5e7eb; border-radius: 10px; height: 10px; width: 200px;">
                            <div style="background: {color}; width: {percentage}%; height: 10px; border-radius: 10px;"></div>
                        </div>
                    </td>
                    <td style="padding: 15px; border-bottom: 1px solid #e5e7eb; text-align: right; font-weight: bold; color: {color};">
                        {percentage:.0f}%
                    </td>
                </tr>
            """)

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weekly Report - {student_model.name}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f9fafb;
                    padding: 20px;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .content {{
                    padding: 40px;
                }}
                .stat-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .stat-card {{
                    background: #f9fafb;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .section h2 {{
                    color: #1f2937;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .recommendations {{
                    background: #f0f9ff;
                    border-left: 4px solid #3b82f6;
                    padding: 20px;
                    border-radius: 4px;
                }}
                .trend-badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                    text-transform: uppercase;
                }}
                .trend-accelerating {{
                    background: #d1fae5;
                    color: #065f46;
                }}
                .trend-steady {{
                    background: #dbeafe;
                    color: #1e40af;
                }}
                .trend-decelerating {{
                    background: #fee2e2;
                    color: #991b1b;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📈 Weekly Progress Report</h1>
                    <p>{student_model.name} - Week Ending {week_ending}</p>
                </div>

                <div class="content">
                    <!-- Overall Progress -->
                    <div class="stat-grid">
                        <div class="stat-card">
                            <p style="color: #9ca3af; font-size: 12px; margin-bottom: 8px;">OBJECTIVES MASTERED</p>
                            <p style="color: #1f2937; font-size: 32px; font-weight: bold;">{report['overall_progress']['mastered_count']}</p>
                        </div>
                        <div class="stat-card">
                            <p style="color: #9ca3af; font-size: 12px; margin-bottom: 8px;">TOTAL XP</p>
                            <p style="color: #10b981; font-size: 32px; font-weight: bold;">{report['overall_progress']['total_xp']:.0f}</p>
                        </div>
                        <div class="stat-card">
                            <p style="color: #9ca3af; font-size: 12px; margin-bottom: 8px;">CURRENT LEVEL</p>
                            <p style="color: #8b5cf6; font-size: 32px; font-weight: bold;">{report['overall_progress']['level']}</p>
                        </div>
                        <div class="stat-card">
                            <p style="color: #9ca3af; font-size: 12px; margin-bottom: 8px;">STREAK DAYS</p>
                            <p style="color: #f59e0b; font-size: 32px; font-weight: bold;">{report['overall_progress']['streak_days']}</p>
                        </div>
                    </div>

                    <!-- Learning Velocity -->
                    <div class="section">
                        <h2>Learning Velocity
                            <span class="trend-badge trend-{velocity.trend}">
                                {velocity.trend}
                            </span>
                        </h2>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                            <div style="background: #f9fafb; padding: 15px; border-radius: 8px;">
                                <p style="color: #6b7280; font-size: 14px;">Objectives per Week</p>
                                <p style="color: #1f2937; font-size: 24px; font-weight: bold;">{velocity.objectives_per_week:.1f}</p>
                            </div>
                            <div style="background: #f9fafb; padding: 15px; border-radius: 8px;">
                                <p style="color: #6b7280; font-size: 14px;">XP per Week</p>
                                <p style="color: #1f2937; font-size: 24px; font-weight: bold;">{velocity.xp_per_week:.0f}</p>
                            </div>
                            <div style="background: #f9fafb; padding: 15px; border-radius: 8px;">
                                <p style="color: #6b7280; font-size: 14px;">Average Confidence</p>
                                <p style="color: #1f2937; font-size: 24px; font-weight: bold;">{velocity.avg_confidence:.1%}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Subject Breakdown -->
                    <div class="section">
                        <h2>Subject Progress</h2>
                        <table>
                            {''.join(subject_rows)}
                        </table>
                    </div>

                    <!-- Engagement -->
                    <div class="section">
                        <h2>Engagement Metrics</h2>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                            <div style="background: #f9fafb; padding: 15px; border-radius: 8px;">
                                <p style="color: #6b7280; font-size: 14px;">Total Interactions</p>
                                <p style="color: #1f2937; font-size: 24px; font-weight: bold;">{engagement.total_interactions}</p>
                            </div>
                            <div style="background: #f9fafb; padding: 15px; border-radius: 8px;">
                                <p style="color: #6b7280; font-size: 14px;">Success Rate</p>
                                <p style="color: #1f2937; font-size: 24px; font-weight: bold;">{engagement.success_rate:.1%}</p>
                            </div>
                            <div style="background: #f9fafb; padding: 15px; border-radius: 8px;">
                                <p style="color: #6b7280; font-size: 14px;">Engagement Score</p>
                                <p style="color: #1f2937; font-size: 24px; font-weight: bold;">{engagement.engagement_score:.1%}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Recommendations -->
                    <div class="section">
                        <h2>Recommended Practice</h2>
                        <div class="recommendations">
                            <ul style="margin-left: 20px; color: #1f2937; line-height: 1.8;">
                                {''.join(f'<li>{rec}</li>' for rec in report['recommendations'][:5])}
                            </ul>
                        </div>
                    </div>
                </div>

                <div style="background: #f9fafb; padding: 20px; text-align: center; color: #6b7280; font-size: 14px;">
                    <p>Generated by EdWIN AI Tutor | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    # ========================================================================
    # Monthly Reports
    # ========================================================================

    async def generate_monthly_report_html(
        self,
        student_model: EdWINStudentModel
    ) -> str:
        """
        Generate monthly report in HTML format

        Args:
            student_model: Student model

        Returns:
            HTML string
        """
        # Get analytics
        report = self.analytics.generate_progress_report(student_model)
        engagement = self.analytics.calculate_engagement(student_model, window_days=30)
        gaps = self.analytics.analyze_knowledge_gaps(student_model)

        # Month
        month = datetime.utcnow().strftime("%B %Y")

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Monthly Report - {student_model.name}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f9fafb;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%);
                    color: white;
                    padding: 50px;
                    text-align: center;
                }}
                .content {{
                    padding: 50px;
                }}
                .section {{
                    margin-bottom: 50px;
                }}
                .alert {{
                    background: #fef3c7;
                    border-left: 4px solid #f59e0b;
                    padding: 20px;
                    border-radius: 4px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📅 Monthly Progress Report</h1>
                    <p style="font-size: 20px; margin-top: 10px;">{student_model.name} - {month}</p>
                </div>

                <div class="content">
                    <div class="section">
                        <h2 style="color: #1f2937; margin-bottom: 20px;">Executive Summary</h2>
                        <p style="color: #6b7280; line-height: 1.8; font-size: 16px;">
                            {student_model.name} has mastered <strong>{report['overall_progress']['mastered_count']}</strong> objectives
                            ({report['overall_progress']['mastery_percentage']:.1f}% of curriculum) and is currently at
                            <strong>Level {report['overall_progress']['level']}</strong> with <strong>{report['overall_progress']['total_xp']:.0f} XP</strong>.
                        </p>

                        {f'''
                        <div class="alert">
                            <p style="color: #92400e; margin: 0;">
                                <strong>⚠️ Areas Needing Attention:</strong> {gaps.total_gaps} knowledge gaps identified.
                                Estimated catchup time: {gaps.estimated_catchup_hours:.1f} hours.
                            </p>
                        </div>
                        ''' if gaps.total_gaps > 3 else ''}
                    </div>

                    <div class="section">
                        <h2 style="color: #1f2937; margin-bottom: 20px;">Engagement Analysis</h2>
                        <p style="color: #6b7280; line-height: 1.8;">
                            Engagement Score: <strong style="color: {'#10b981' if engagement.engagement_score > 0.6 else '#f59e0b'};">{engagement.engagement_score:.1%}</strong>
                            <br>
                            Risk Level: <strong style="color: {'#10b981' if engagement.risk_level == 'low' else '#f59e0b' if engagement.risk_level == 'medium' else '#ef4444'};">{engagement.risk_level.upper()}</strong>
                        </p>
                    </div>
                </div>

                <div style="background: #f9fafb; padding: 20px; text-align: center; color: #6b7280; font-size: 14px;">
                    <p>Generated by EdWIN AI Tutor | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    # ========================================================================
    # JSON Export
    # ========================================================================

    async def generate_report_json(
        self,
        student_model: EdWINStudentModel,
        report_type: str = "weekly"
    ) -> Dict[str, Any]:
        """
        Generate report in JSON format

        Args:
            student_model: Student model
            report_type: Report type (daily, weekly, monthly)

        Returns:
            Report dict
        """
        report = self.analytics.generate_progress_report(student_model)

        return {
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "student": {
                "id": student_model.student_id,
                "name": student_model.name,
                "grade": student_model.grade
            },
            "data": report
        }

    # ========================================================================
    # PDF Export
    # ========================================================================

    async def generate_report_pdf(
        self,
        student_model: EdWINStudentModel,
        output_path: str,
        report_type: str = "weekly"
    ):
        """
        Generate report in PDF format

        Args:
            student_model: Student model
            output_path: Output file path
            report_type: Report type
        """
        if not HAS_REPORTLAB:
            raise ImportError("ReportLab not installed - cannot generate PDF")

        # Get report data
        report = self.analytics.generate_progress_report(student_model)

        # Create PDF
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        title = Paragraph(f"<b>{report_type.title()} Report - {student_model.name}</b>", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))

        # Overall progress table
        data = [
            ['Metric', 'Value'],
            ['Objectives Mastered', str(report['overall_progress']['mastered_count'])],
            ['Total XP', f"{report['overall_progress']['total_xp']:.0f}"],
            ['Current Level', str(report['overall_progress']['level'])],
            ['Mastery Percentage', f"{report['overall_progress']['mastery_percentage']:.1f}%"]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(table)

        # Build PDF
        doc.build(elements)

        print(f"📄 PDF report saved to {output_path}")


# ============================================================================
# Helper Functions
# ============================================================================

def create_report_generator(
    analytics: LearningAnalytics,
    curriculum_kg: EdWINKnowledgeGraph
) -> ReportGenerator:
    """
    Create report generator

    Args:
        analytics: Learning analytics engine
        curriculum_kg: Curriculum knowledge graph

    Returns:
        ReportGenerator instance
    """
    return ReportGenerator(analytics, curriculum_kg)
