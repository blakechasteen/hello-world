# Bash completion for promptly
# Source this file or copy to /etc/bash_completion.d/

_promptly_completions()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    local commands="search create update delete list execute branch merge diff analytics skills help version"

    # Options for each command
    case "${prev}" in
        search)
            opts="--tags --limit --format"
            ;;
        create)
            opts="--name --content --tags --from-file"
            ;;
        update)
            opts="--name --content --tags --version"
            ;;
        delete)
            opts="--confirm --force"
            ;;
        list)
            opts="--tags --format --sort"
            ;;
        execute)
            opts="--input --loop --iterations --quality-threshold"
            ;;
        branch)
            opts="--from --description"
            ;;
        merge)
            opts="--strategy --resolve-conflicts"
            ;;
        diff)
            opts="--format --context"
            ;;
        analytics)
            opts="--summary --top --recommendations"
            ;;
        skills)
            opts="--list --add --remove --update"
            ;;
        *)
            ;;
    esac

    # Complete based on context
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    else
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
    fi

    return 0
}

# Register completion
complete -F _promptly_completions promptly
complete -F _promptly_completions p  # Short alias
