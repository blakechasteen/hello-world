#!/usr/bin/env python3
"""
Content Generation Use Case
============================
Multi-stage content pipeline with quality validation and optimization.

This demo shows:
- Blog post generation pipeline (outline → draft → edit → publish)
- Quality checks at each stage
- Style consistency validation
- SEO optimization
- Multi-format content generation (blog, social, newsletter)

Requirements:
    pip install promptly

Usage:
    python examples/use_cases/content_generation.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from Promptly.promptly.promptly import Promptly


class ContentGenerationDemo:
    """Content generation pipeline demo"""

    def __init__(self):
        self.demo_dir = Path(__file__).parent / 'content_output'
        self.demo_dir.mkdir(exist_ok=True)
        self.promptly = None
        self.generated_content = {}

    def setup(self):
        """Initialize Promptly repository"""
        print("=" * 80)
        print("CONTENT GENERATION PIPELINE DEMO".center(80))
        print("=" * 80)
        print()

        repo_path = self.demo_dir / 'content_repo'
        repo_path.mkdir(exist_ok=True)

        self.promptly = Promptly(root_dir=str(repo_path))

        if not (repo_path / '.promptly').exists():
            self.promptly.init()
            self.promptly.create_branch('content-pipeline')
            self.promptly.checkout('content-pipeline')
            print("✓ Initialized content repository")

    def create_pipeline_prompts(self):
        """Create prompts for each stage of content pipeline"""
        print("\n[1/7] Creating Content Pipeline Prompts")
        print("-" * 80)

        prompts = {
            # Stage 1: Ideation & Outline
            'content_outline': {
                'content': '''Create a detailed outline for a blog post about: {topic}

Target Audience: {audience}
Tone: {tone}
Word Count Target: {word_count}

Include:
1. Compelling title (with 3 alternatives)
2. Hook/introduction (1-2 paragraphs)
3. Main sections (3-5 major points)
   - Each section with 2-3 sub-points
4. Conclusion points
5. Call-to-action ideas
6. SEO keywords (5-7 primary keywords)

Format as structured markdown with clear hierarchy.''',
                'metadata': {
                    'stage': 'outline',
                    'quality_threshold': 0.85,
                    'avg_time': '3-5 minutes'
                }
            },

            # Stage 2: Draft Generation
            'content_draft': {
                'content': '''Write a complete blog post draft based on this outline:

{outline}

Requirements:
- Target length: {word_count} words
- Tone: {tone}
- Include relevant examples and anecdotes
- Use subheadings for readability
- Incorporate these keywords naturally: {keywords}
- Add transitional phrases between sections
- Include at least one data point or statistic per major section

Write in an engaging, {tone} style that resonates with {audience}.''',
                'metadata': {
                    'stage': 'draft',
                    'quality_threshold': 0.80,
                    'avg_time': '10-15 minutes'
                }
            },

            # Stage 3: Editing & Refinement
            'content_edit': {
                'content': '''Edit and refine this blog post draft:

{draft}

Focus on:
1. **Clarity**: Simplify complex sentences, remove jargon
2. **Flow**: Improve transitions, ensure logical progression
3. **Engagement**: Add hooks, questions, vivid language
4. **Conciseness**: Remove redundancy, tighten prose
5. **Grammar**: Fix any errors
6. **Voice**: Maintain consistent {tone} tone

Target improvements:
- Readability score: {target_readability}
- Engagement score: {target_engagement}
- SEO score: {target_seo}

Provide the edited version with track changes summary.''',
                'metadata': {
                    'stage': 'edit',
                    'quality_threshold': 0.90,
                    'avg_time': '8-12 minutes'
                }
            },

            # Stage 4: SEO Optimization
            'content_seo': {
                'content': '''Optimize this content for SEO:

{content}

Primary Keywords: {primary_keywords}
Secondary Keywords: {secondary_keywords}
Target Search Intent: {search_intent}

Optimizations:
1. Meta title (50-60 chars, include primary keyword)
2. Meta description (150-160 chars, compelling, include CTA)
3. URL slug (5-7 words, keyword-rich)
4. H1 tag optimization
5. Image alt text suggestions (5 images)
6. Internal linking opportunities (3-5 links)
7. External linking suggestions (2-3 authoritative sources)
8. FAQ section (3-5 questions)

Provide complete SEO package with justifications.''',
                'metadata': {
                    'stage': 'seo',
                    'quality_threshold': 0.88,
                    'avg_time': '5-7 minutes'
                }
            },

            # Stage 5: Social Media Adaptation
            'content_social': {
                'content': '''Adapt this blog post for social media:

{blog_content}

Create:

1. **Twitter Thread** (8-10 tweets)
   - Hook tweet with engagement question
   - Key points (1 per tweet)
   - Visual description suggestions
   - Final tweet with CTA and link

2. **LinkedIn Post** (1300 chars)
   - Professional angle
   - Industry insights
   - Personal perspective
   - Hashtags (3-5 relevant)

3. **Instagram Caption** (2200 chars)
   - Engaging opening
   - Story-driven
   - Emojis (but not excessive)
   - Hashtags (20-25 mix of popular and niche)

4. **Facebook Post** (500 chars)
   - Conversational tone
   - Question to drive comments
   - Clear CTA

Include posting time recommendations for each platform.''',
                'metadata': {
                    'stage': 'social',
                    'quality_threshold': 0.82,
                    'avg_time': '6-8 minutes'
                }
            },

            # Stage 6: Newsletter Version
            'content_newsletter': {
                'content': '''Convert this blog post into a newsletter format:

{blog_content}

Newsletter Structure:
1. **Subject Line** (3 options)
   - Use power words
   - Create urgency or curiosity
   - Keep under 50 chars

2. **Preview Text** (90 chars)

3. **Header**
   - Personalized greeting
   - Context for why this matters now

4. **Main Content** (300-500 words)
   - Scannable format
   - Bullet points and short paragraphs
   - Include one "quick tip" callout box

5. **Footer**
   - Related resources (2-3 links)
   - Social proof or testimonial
   - Clear CTA button text
   - P.S. with additional value

Tone: {newsletter_tone}
Audience: {subscriber_segment}''',
                'metadata': {
                    'stage': 'newsletter',
                    'quality_threshold': 0.86,
                    'avg_time': '7-9 minutes'
                }
            },

            # Stage 7: Quality Validation
            'content_quality_check': {
                'content': '''Perform comprehensive quality check on this content:

{content}

Evaluate on scale of 0-10:

1. **Readability**
   - Sentence length variation
   - Paragraph structure
   - Use of subheadings
   - Technical jargon level

2. **Engagement**
   - Hook strength
   - Story elements
   - Examples and anecdotes
   - Call-to-action clarity

3. **Accuracy**
   - Factual claims verification needed
   - Statistics and data points
   - Attribution requirements

4. **SEO**
   - Keyword usage (natural vs forced)
   - Meta data completeness
   - Internal/external links
   - Mobile-friendliness considerations

5. **Brand Alignment**
   - Tone consistency
   - Value proposition clarity
   - Brand voice adherence

Provide:
- Score for each dimension
- Specific improvement suggestions
- Pass/fail determination (threshold: 7/10 average)
- Estimated reader time
- Reading level (grade)''',
                'metadata': {
                    'stage': 'quality_check',
                    'quality_threshold': 0.92,
                    'avg_time': '4-6 minutes'
                }
            }
        }

        for name, prompt in prompts.items():
            self.promptly.save(
                name=name,
                content=prompt['content'],
                metadata=prompt['metadata']
            )
            print(f"✓ Created: {name}")

        print(f"\n✓ Created {len(prompts)} pipeline prompts")

    def execute_pipeline(self):
        """Execute complete content generation pipeline"""
        print("\n[2/7] Executing Content Pipeline")
        print("-" * 80)

        # Define content brief
        content_brief = {
            'topic': 'The Future of Remote Work: Trends and Best Practices',
            'audience': 'HR professionals and business leaders',
            'tone': 'professional yet approachable',
            'word_count': 1500,
            'keywords': ['remote work', 'hybrid teams', 'productivity', 'work-life balance', 'digital collaboration'],
            'search_intent': 'informational',
            'newsletter_tone': 'insider tips',
            'subscriber_segment': 'HR managers'
        }

        print(f"Topic: {content_brief['topic']}")
        print(f"Target Audience: {content_brief['audience']}")
        print(f"Word Count: {content_brief['word_count']}")

        # Stage 1: Generate outline
        print("\n[Stage 1/7] Generating outline...")
        outline_result = self._simulate_content_generation('content_outline', content_brief)
        self.generated_content['outline'] = outline_result
        print(f"✓ Outline generated ({len(outline_result)} chars)")

        # Stage 2: Generate draft
        print("\n[Stage 2/7] Writing draft...")
        draft_params = {
            **content_brief,
            'outline': outline_result
        }
        draft_result = self._simulate_content_generation('content_draft', draft_params)
        self.generated_content['draft'] = draft_result
        print(f"✓ Draft written ({len(draft_result)} chars)")

        # Stage 3: Edit draft
        print("\n[Stage 3/7] Editing content...")
        edit_params = {
            **content_brief,
            'draft': draft_result,
            'target_readability': '8th grade',
            'target_engagement': '8/10',
            'target_seo': '85/100'
        }
        edited_result = self._simulate_content_generation('content_edit', edit_params)
        self.generated_content['edited'] = edited_result
        print(f"✓ Content edited ({len(edited_result)} chars)")

        # Stage 4: SEO optimization
        print("\n[Stage 4/7] Optimizing for SEO...")
        seo_params = {
            'content': edited_result,
            'primary_keywords': content_brief['keywords'][:3],
            'secondary_keywords': content_brief['keywords'][3:],
            'search_intent': content_brief['search_intent']
        }
        seo_result = self._simulate_content_generation('content_seo', seo_params)
        self.generated_content['seo'] = seo_result
        print(f"✓ SEO optimization complete")

        # Stage 5: Social media adaptation
        print("\n[Stage 5/7] Creating social media content...")
        social_result = self._simulate_content_generation('content_social', {
            'blog_content': edited_result
        })
        self.generated_content['social'] = social_result
        print(f"✓ Social media content created")

        # Stage 6: Newsletter version
        print("\n[Stage 6/7] Creating newsletter version...")
        newsletter_result = self._simulate_content_generation('content_newsletter', {
            'blog_content': edited_result,
            'newsletter_tone': content_brief['newsletter_tone'],
            'subscriber_segment': content_brief['subscriber_segment']
        })
        self.generated_content['newsletter'] = newsletter_result
        print(f"✓ Newsletter version created")

        # Stage 7: Quality check
        print("\n[Stage 7/7] Running quality check...")
        quality_result = self._simulate_quality_check(edited_result)
        self.generated_content['quality_check'] = quality_result
        print(f"✓ Quality check complete (score: {quality_result['avg_score']:.1f}/10)")

    def _simulate_content_generation(self, prompt_name: str, params: Dict) -> str:
        """Simulate content generation (in production, this would call an LLM)"""
        # Record the generation
        self.promptly.eval_prompt(
            name=prompt_name,
            test_case=json.dumps(params),
            score=0.85  # Simulated quality score
        )

        # Return simulated content based on stage
        simulations = {
            'content_outline': '''# The Future of Remote Work: Trends and Best Practices

## Hook
The pandemic accelerated a workplace revolution that's here to stay...

## Main Sections
1. Current Remote Work Landscape
   - Statistics and adoption rates
   - Industry variations
   - Geographic trends

2. Emerging Trends
   - Hybrid models
   - Async-first communication
   - Results-oriented work environments

3. Best Practices for Success
   - Technology infrastructure
   - Team communication protocols
   - Work-life balance initiatives

4. Challenges and Solutions
   - Collaboration hurdles
   - Company culture
   - Performance management

5. Future Predictions
   - 2025 outlook
   - Technology innovations
   - Workforce expectations

## Conclusion
Remote work is not just a trend but a fundamental shift...

## CTA
Download our free remote work policy template''',

            'content_draft': '''[Full draft content with 1500 words covering all outline points...]''',
            'content_edit': '''[Edited and refined version with improved flow and engagement...]''',
            'content_seo': '''[SEO-optimized version with meta tags and keyword integration...]''',
            'content_social': '''[Multi-platform social media content package...]''',
            'content_newsletter': '''[Newsletter-formatted version with subject lines and CTAs...]'''
        }

        return simulations.get(prompt_name, f"[Generated content for {prompt_name}]")

    def _simulate_quality_check(self, content: str) -> Dict:
        """Simulate quality check"""
        import random

        scores = {
            'readability': random.uniform(7.5, 9.5),
            'engagement': random.uniform(7.0, 9.0),
            'accuracy': random.uniform(8.0, 9.5),
            'seo': random.uniform(7.5, 9.0),
            'brand_alignment': random.uniform(8.0, 9.5)
        }

        avg_score = sum(scores.values()) / len(scores)

        return {
            'scores': scores,
            'avg_score': avg_score,
            'passed': avg_score >= 7.0,
            'reading_level': '8th grade',
            'estimated_reading_time': '6-7 minutes',
            'word_count': 1542
        }

    def analyze_quality_metrics(self):
        """Analyze quality metrics across pipeline"""
        print("\n[3/7] Analyzing Quality Metrics")
        print("-" * 80)

        stages = ['outline', 'draft', 'edited', 'seo', 'social', 'newsletter']

        for stage in stages:
            if stage in self.generated_content:
                content = self.generated_content[stage]
                word_count = len(content.split())
                char_count = len(content)

                print(f"\n{stage.title()}:")
                print(f"  Word Count: {word_count}")
                print(f"  Character Count: {char_count}")

        # Quality check scores
        if 'quality_check' in self.generated_content:
            qc = self.generated_content['quality_check']
            print("\nQuality Check Scores:")
            for metric, score in qc['scores'].items():
                print(f"  {metric:20s}: {score:.1f}/10")
            print(f"\n  Overall: {qc['avg_score']:.1f}/10 ({'PASS' if qc['passed'] else 'FAIL'})")

    def version_comparison(self):
        """Compare different versions for A/B testing"""
        print("\n[4/7] Version Comparison for A/B Testing")
        print("-" * 80)

        # Create alternative version
        print("Creating alternative version with different tone...")

        alt_brief = {
            'topic': 'The Future of Remote Work: Trends and Best Practices',
            'audience': 'HR professionals and business leaders',
            'tone': 'conversational and casual',
            'word_count': 1500,
        }

        # Simulate alternative version
        print("✓ Alternative version created")

        # Compare versions
        versions = [
            {
                'name': 'Professional Tone',
                'engagement_score': 8.2,
                'readability_score': 8.5,
                'seo_score': 8.7,
                'avg_time_on_page': '6m 23s'
            },
            {
                'name': 'Conversational Tone',
                'engagement_score': 8.8,
                'readability_score': 9.1,
                'seo_score': 8.3,
                'avg_time_on_page': '7m 12s'
            }
        ]

        print("\nVersion Comparison:")
        print(f"{'Version':<25} {'Engagement':<15} {'Readability':<15} {'SEO':<10}")
        print("-" * 70)
        for v in versions:
            print(f"{v['name']:<25} {v['engagement_score']:<15.1f} {v['readability_score']:<15.1f} {v['seo_score']:<10.1f}")

        winner = max(versions, key=lambda x: (x['engagement_score'] + x['readability_score'] + x['seo_score']) / 3)
        print(f"\nRecommended Version: {winner['name']}")

    def export_content(self):
        """Export content in multiple formats"""
        print("\n[5/7] Exporting Content")
        print("-" * 80)

        # Export blog post
        blog_path = self.demo_dir / 'blog_post.md'
        with open(blog_path, 'w') as f:
            f.write(f"# The Future of Remote Work\n\n")
            f.write(self.generated_content.get('edited', ''))
        print(f"✓ Blog post exported: {blog_path}")

        # Export social media package
        social_path = self.demo_dir / 'social_media_package.txt'
        with open(social_path, 'w') as f:
            f.write(self.generated_content.get('social', ''))
        print(f"✓ Social media package exported: {social_path}")

        # Export newsletter
        newsletter_path = self.demo_dir / 'newsletter.html'
        with open(newsletter_path, 'w') as f:
            f.write(f"<html><body>\n{self.generated_content.get('newsletter', '')}\n</body></html>")
        print(f"✓ Newsletter exported: {newsletter_path}")

        # Export SEO metadata
        seo_path = self.demo_dir / 'seo_metadata.json'
        with open(seo_path, 'w') as f:
            json.dump({
                'meta_title': 'Future of Remote Work: Trends & Best Practices 2024',
                'meta_description': 'Discover the latest remote work trends and best practices for HR leaders. Learn how to build successful hybrid teams.',
                'keywords': ['remote work', 'hybrid teams', 'productivity'],
                'url_slug': 'future-remote-work-trends-best-practices'
            }, f, indent=2)
        print(f"✓ SEO metadata exported: {seo_path}")

    def track_performance(self):
        """Track performance metrics"""
        print("\n[6/7] Tracking Performance Metrics")
        print("-" * 80)

        metrics = {
            'total_pipeline_time': '45 minutes',
            'prompts_used': 7,
            'versions_generated': 8,
            'quality_checks_passed': 7,
            'formats_created': 4,
            'estimated_manual_time': '8-10 hours',
            'time_saved': '88%',
            'quality_improvement': '+23%'
        }

        print("Pipeline Performance:")
        for key, value in metrics.items():
            print(f"  {key.replace('_', ' ').title():<30}: {value}")

    def generate_report(self):
        """Generate comprehensive report"""
        print("\n[7/7] Generating Report")
        print("-" * 80)

        report = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_summary': {
                'stages_completed': 7,
                'content_pieces_generated': len(self.generated_content),
                'quality_score': self.generated_content.get('quality_check', {}).get('avg_score', 0),
                'total_word_count': sum(len(c.split()) if isinstance(c, str) else 0
                                       for c in self.generated_content.values())
            },
            'content_outputs': list(self.generated_content.keys()),
            'quality_metrics': self.generated_content.get('quality_check', {}),
            'recommendations': [
                'Conversational tone performs better for engagement',
                'SEO optimization increased search visibility by estimated 23%',
                'Multi-format approach allows for broader distribution',
                'Quality checks ensure consistent brand voice',
                'Pipeline reduces content creation time by 88%'
            ]
        }

        report_path = self.demo_dir / 'content_generation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Report saved to: {report_path}")

        print("\n" + "=" * 80)
        print("CONTENT GENERATION PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Stages Completed: {report['pipeline_summary']['stages_completed']}")
        print(f"Content Pieces: {report['pipeline_summary']['content_pieces_generated']}")
        print(f"Quality Score: {report['pipeline_summary']['quality_score']:.1f}/10")
        print(f"Total Word Count: {report['pipeline_summary']['total_word_count']}")

    def run(self):
        """Run complete demo"""
        try:
            self.setup()
            self.create_pipeline_prompts()
            self.execute_pipeline()
            self.analyze_quality_metrics()
            self.version_comparison()
            self.export_content()
            self.track_performance()
            self.generate_report()

            print("\n" + "=" * 80)
            print("✓ CONTENT GENERATION DEMO COMPLETED")
            print("=" * 80)
            return True

        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    demo = ContentGenerationDemo()
    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
