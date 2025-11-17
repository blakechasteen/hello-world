#!/usr/bin/env bash
# Bash completion for promptly

_promptly_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    local commands="init add get list log branch checkout eval chain wizard tui interactive status show diff export import help version"

    # Subcommands for eval
    local eval_commands="run list"

    # Subcommands for chain
    local chain_commands="create run list"

    # Subcommands for wizard
    local wizard_commands="project prompt chain evaluation template"

    # If we're completing the first argument
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
        return 0
    fi

    # Handle subcommands
    local main_command="${COMP_WORDS[1]}"

    case "${main_command}" in
        eval)
            if [ $COMP_CWORD -eq 2 ]; then
                COMPREPLY=( $(compgen -W "${eval_commands}" -- ${cur}) )
            elif [ $COMP_CWORD -eq 3 ]; then
                # Complete with prompt names
                local prompts=$(promptly list 2>/dev/null | grep -v "Prompts on branch" | awk '{print $1}')
                COMPREPLY=( $(compgen -W "${prompts}" -- ${cur}) )
            elif [ $COMP_CWORD -eq 4 ]; then
                # Complete with file names
                COMPREPLY=( $(compgen -f -- ${cur}) )
            fi
            return 0
            ;;
        chain)
            if [ $COMP_CWORD -eq 2 ]; then
                COMPREPLY=( $(compgen -W "${chain_commands}" -- ${cur}) )
            elif [ $COMP_CWORD -eq 3 ]; then
                if [ "${COMP_WORDS[2]}" == "run" ]; then
                    # Complete with chain names from .promptly/chains/
                    if [ -d ".promptly/chains" ]; then
                        local chains=$(ls .promptly/chains/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//')
                        COMPREPLY=( $(compgen -W "${chains}" -- ${cur}) )
                    fi
                fi
            fi
            return 0
            ;;
        wizard)
            if [ $COMP_CWORD -eq 2 ]; then
                COMPREPLY=( $(compgen -W "${wizard_commands}" -- ${cur}) )
            fi
            return 0
            ;;
        get|show|log)
            if [ $COMP_CWORD -eq 2 ]; then
                # Complete with prompt names
                local prompts=$(promptly list 2>/dev/null | grep -v "Prompts on branch" | awk '{print $1}')
                COMPREPLY=( $(compgen -W "${prompts}" -- ${cur}) )
            fi
            return 0
            ;;
        checkout|branch)
            if [ $COMP_CWORD -eq 2 ]; then
                # Complete with branch names
                local branches=$(promptly list --branch 2>/dev/null | grep "Prompts on branch" | cut -d"'" -f2)
                COMPREPLY=( $(compgen -W "${branches}" -- ${cur}) )
            fi
            return 0
            ;;
        add)
            # No specific completion for add
            return 0
            ;;
        export)
            if [ "${prev}" == "--format" ] || [ "${prev}" == "-f" ]; then
                COMPREPLY=( $(compgen -W "json yaml" -- ${cur}) )
            elif [ $COMP_CWORD -eq 2 ]; then
                # Complete with filename
                COMPREPLY=( $(compgen -f -- ${cur}) )
            fi
            return 0
            ;;
        *)
            return 0
            ;;
    esac
}

# Register completion function
complete -F _promptly_completions promptly
complete -F _promptly_completions promptly-enhanced
complete -F _promptly_completions promptly-wizard
