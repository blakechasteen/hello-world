#!/usr/bin/env python3
"""
Template CLI Commands - CLI integration for template system
"""

import click
import yaml
import json
from pathlib import Path
from typing import Optional, List

from .templates import (
    TemplateEngine,
    TemplateContext,
    TemplateRegistry,
    TemplateDiscovery,
    TemplateValidator,
    TemplateComposer,
)
from .templates.library import get_library_path


def get_template_registry_path() -> Path:
    """Get the path to the template registry file"""
    return Path.cwd() / '.promptly' / 'template_registry.yaml'


def ensure_template_system_initialized() -> TemplateRegistry:
    """Ensure template system is initialized and return registry"""
    registry_path = get_template_registry_path()

    # Create registry file if it doesn't exist
    if not registry_path.exists():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry = TemplateRegistry(str(registry_path))

        # Discover built-in templates
        library_path = get_library_path()
        discovery = TemplateDiscovery(registry)
        count = discovery.discover_builtin_templates(library_path)

        # Save initial registry
        registry.save()
        click.echo(click.style(f"Initialized template system with {count} built-in templates", fg='green'))
    else:
        registry = TemplateRegistry(str(registry_path))

    return registry


@click.group()
def template():
    """Manage prompt templates"""
    pass


@template.command(name='create')
@click.argument('name')
@click.option('--from', 'base_template', help='Base template to extend from')
@click.option('--category', default='general', help='Template category')
@click.option('--description', '-d', help='Template description')
@click.option('--content', '-c', help='Template content (or use --file)')
@click.option('--file', '-f', 'content_file', type=click.Path(exists=True), help='File containing template content')
@click.option('--interactive', '-i', is_flag=True, help='Interactive template creation')
def template_create(name, base_template, category, description, content, content_file, interactive):
    """Create a new template"""
    try:
        registry = ensure_template_system_initialized()

        if interactive:
            # Interactive mode
            click.echo(click.style("Interactive Template Creation", fg='cyan', bold=True))
            name = click.prompt('Template name', default=name)
            category = click.prompt('Category', default=category)
            description = click.prompt('Description', default=description or '')

            click.echo('\nEnter template content (press Ctrl+D when done):')
            content_lines = []
            try:
                while True:
                    line = input()
                    content_lines.append(line)
            except EOFError:
                content = '\n'.join(content_lines)

        # Get content from file if specified
        if content_file:
            with open(content_file, 'r') as f:
                content = f.read()

        # Check if extending from base template
        if base_template:
            base = registry.get(base_template)
            if not base:
                click.echo(click.style(f"Error: Base template '{base_template}' not found", fg='red'), err=True)
                return

            if not content:
                content = base.content
            else:
                # Compose with base template
                content = f"{{% extends '{base_template}' %}}\n\n{content}"

        if not content:
            click.echo(click.style("Error: No content provided. Use --content, --file, or --interactive", fg='red'), err=True)
            return

        # Register the template
        entry = registry.register(
            name=name,
            content=content,
            category=category,
            description=description
        )

        registry.save()

        click.echo(click.style(f"✓ Created template '{name}' in category '{category}'", fg='green'))
        if description:
            click.echo(f"  {description}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='list')
@click.option('--category', '-c', help='Filter by category')
@click.option('--tag', '-t', multiple=True, help='Filter by tag(s)')
@click.option('--search', '-s', help='Search in name/description')
def template_list(category, tag, search):
    """List available templates"""
    try:
        registry = ensure_template_system_initialized()

        templates = registry.list_templates(
            category=category,
            tags=list(tag) if tag else None,
            search=search
        )

        if not templates:
            click.echo(click.style("No templates found", fg='yellow'))
            return

        click.echo(click.style(f"\n{len(templates)} template(s) found:", fg='cyan', bold=True))
        click.echo()

        current_category = None
        for tmpl in templates:
            if tmpl.category != current_category:
                current_category = tmpl.category
                click.echo(click.style(f"\n{current_category.upper()}", fg='magenta', bold=True))

            version_info = f"v{tmpl.current_version}"
            tags_str = f" [{', '.join(sorted(tmpl.tags))}]" if tmpl.tags else ""

            click.echo(f"  {click.style(tmpl.name, fg='green')} ({version_info}){tags_str}")
            if tmpl.description:
                click.echo(f"    {tmpl.description}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='show')
@click.argument('name')
@click.option('--version', '-v', type=int, help='Specific version')
def template_show(name, version):
    """Show template details"""
    try:
        registry = ensure_template_system_initialized()

        entry = registry.get_entry(name)
        if not entry:
            click.echo(click.style(f"Template '{name}' not found", fg='red'), err=True)
            return

        template = entry.get_version(version)
        if not template:
            click.echo(click.style(f"Version {version} not found", fg='red'), err=True)
            return

        click.echo(click.style(f"\nTemplate: {entry.name}", fg='cyan', bold=True))
        click.echo(f"Category: {entry.category}")
        click.echo(f"Version: {template.version}")
        if entry.description:
            click.echo(f"Description: {entry.description}")
        if entry.tags:
            click.echo(f"Tags: {', '.join(sorted(entry.tags))}")

        click.echo(click.style("\nContent:", fg='cyan'))
        click.echo(template.content)

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='render')
@click.argument('name')
@click.option('--vars', help='Variables as JSON string')
@click.option('--vars-file', type=click.Path(exists=True), help='Variables from YAML/JSON file')
@click.option('--output', '-o', type=click.Path(), help='Output file (default: stdout)')
def template_render(name, vars, vars_file, output):
    """Render a template with variables"""
    try:
        registry = ensure_template_system_initialized()

        template = registry.get(name)
        if not template:
            click.echo(click.style(f"Template '{name}' not found", fg='red'), err=True)
            return

        # Load variables
        variables = {}

        if vars_file:
            with open(vars_file, 'r') as f:
                if vars_file.endswith('.yaml') or vars_file.endswith('.yml'):
                    variables = yaml.safe_load(f)
                elif vars_file.endswith('.json'):
                    variables = json.load(f)

        if vars:
            vars_dict = json.loads(vars)
            variables.update(vars_dict)

        # Create engine and render
        engine = TemplateEngine()
        context = TemplateContext(variables=variables)
        rendered = engine.render_string(template.content, context)

        if output:
            with open(output, 'w') as f:
                f.write(rendered)
            click.echo(click.style(f"✓ Rendered to {output}", fg='green'))
        else:
            click.echo(rendered)

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='validate')
@click.argument('name')
@click.option('--strict', is_flag=True, help='Enable strict validation')
def template_validate(name, strict):
    """Validate a template"""
    try:
        registry = ensure_template_system_initialized()

        entry = registry.get_entry(name)
        if not entry:
            click.echo(click.style(f"Template '{name}' not found", fg='red'), err=True)
            return

        template = entry.get_latest_version()

        # Create template data for validation
        template_data = {
            'name': entry.name,
            'content': template.content,
            'description': entry.description,
            'category': entry.category,
            'tags': list(entry.tags)
        }

        validator = TemplateValidator()
        result = validator.validate(template_data, strict=strict)

        # Display results
        if result.valid:
            click.echo(click.style("✓ Validation PASSED", fg='green', bold=True))
        else:
            click.echo(click.style("✗ Validation FAILED", fg='red', bold=True))

        # Show issues by level
        errors = result.get_errors()
        warnings = result.get_warnings()
        info = result.get_info()

        if errors:
            click.echo(click.style(f"\nErrors ({len(errors)}):", fg='red', bold=True))
            for issue in errors:
                click.echo(f"  {issue}")

        if warnings:
            click.echo(click.style(f"\nWarnings ({len(warnings)}):", fg='yellow', bold=True))
            for issue in warnings:
                click.echo(f"  {issue}")

        if info:
            click.echo(click.style(f"\nInfo ({len(info)}):", fg='cyan', bold=True))
            for issue in info:
                click.echo(f"  {issue}")

        # Show variable metadata
        if 'variables' in result.metadata:
            var_info = result.metadata['variables']
            click.echo(click.style(f"\nVariables:", fg='cyan'))
            click.echo(f"  Total: {var_info['total']}")
            if var_info['required']:
                click.echo(f"  Required: {', '.join(var_info['required'])}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='delete')
@click.argument('name')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
def template_delete(name, yes):
    """Delete a template"""
    try:
        registry = ensure_template_system_initialized()

        entry = registry.get_entry(name)
        if not entry:
            click.echo(click.style(f"Template '{name}' not found", fg='red'), err=True)
            return

        if not yes:
            click.confirm(f"Delete template '{name}'?", abort=True)

        registry.delete(name)
        registry.save()

        click.echo(click.style(f"✓ Deleted template '{name}'", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='export')
@click.argument('name')
@click.argument('output_file', type=click.Path())
@click.option('--version', '-v', type=int, help='Specific version to export')
def template_export(name, output_file, version):
    """Export a template to a file"""
    try:
        registry = ensure_template_system_initialized()

        registry.export_template(name, output_file, version=version)

        click.echo(click.style(f"✓ Exported template '{name}' to {output_file}", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='import')
@click.argument('input_file', type=click.Path(exists=True))
def template_import(input_file):
    """Import a template from a file"""
    try:
        registry = ensure_template_system_initialized()

        entry = registry.import_template(input_file)
        registry.save()

        click.echo(click.style(f"✓ Imported template '{entry.name}'", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='categories')
def template_categories():
    """List all template categories"""
    try:
        registry = ensure_template_system_initialized()

        categories = registry.list_categories()

        if not categories:
            click.echo(click.style("No categories found", fg='yellow'))
            return

        click.echo(click.style("\nTemplate Categories:", fg='cyan', bold=True))
        for cat in categories:
            templates = registry.list_templates(category=cat)
            click.echo(f"  {click.style(cat, fg='green')} ({len(templates)} templates)")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@template.command(name='discover')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--category', help='Category for discovered templates')
def template_discover(directory, category):
    """Discover templates from a directory"""
    try:
        registry = ensure_template_system_initialized()

        discovery = TemplateDiscovery(registry)
        count = discovery.discover_from_directory(
            directory,
            category=category,
            recursive=True
        )

        registry.save()

        click.echo(click.style(f"✓ Discovered {count} template(s) from {directory}", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


# Add template command group to main CLI
def add_template_commands(cli_group):
    """Add template commands to the main CLI"""
    cli_group.add_command(template)
