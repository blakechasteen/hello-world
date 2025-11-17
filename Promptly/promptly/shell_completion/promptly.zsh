#compdef promptly promptly-enhanced promptly-wizard

# Zsh completion for promptly

_promptly() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    local -a commands
    commands=(
        'init:Initialize a new promptly repository'
        'add:Add a new prompt or update existing'
        'get:Get a prompt by name'
        'list:List all prompts'
        'log:Show commit history'
        'branch:Create a new branch'
        'checkout:Switch to a different branch'
        'status:Show repository status'
        'show:Show prompt with syntax highlighting'
        'diff:Compare two versions'
        'export:Export repository'
        'import:Import prompts'
        'eval:Evaluate prompts'
        'chain:Manage prompt chains'
        'wizard:Interactive setup wizards'
        'tui:Launch terminal UI'
        'interactive:Start interactive REPL'
        'help:Show help'
        'version:Show version'
    )

    _arguments -C \
        '1: :->command' \
        '*:: :->args'

    case $state in
        command)
            _describe 'promptly commands' commands
            ;;
        args)
            case $words[1] in
                eval)
                    local -a eval_commands
                    eval_commands=(
                        'run:Run evaluation on a prompt'
                        'list:List evaluations'
                    )
                    _arguments \
                        '1: :_describe "eval commands" eval_commands' \
                        '2: :_promptly_prompts' \
                        '3: :_files'
                    ;;
                chain)
                    local -a chain_commands
                    chain_commands=(
                        'create:Create a new chain'
                        'run:Execute a chain'
                        'list:List chains'
                    )
                    _arguments \
                        '1: :_describe "chain commands" chain_commands' \
                        '2: :_promptly_chains'
                    ;;
                wizard)
                    local -a wizard_commands
                    wizard_commands=(
                        'project:Project setup wizard'
                        'prompt:Prompt creation wizard'
                        'chain:Chain creation wizard'
                        'evaluation:Evaluation setup wizard'
                        'template:Template wizard'
                    )
                    _describe 'wizard commands' wizard_commands
                    ;;
                get|show|log)
                    _promptly_prompts
                    ;;
                checkout|branch)
                    _promptly_branches
                    ;;
                export)
                    _arguments \
                        '1: :_files' \
                        '--format[Output format]:format:(json yaml)'
                    ;;
                add)
                    _arguments \
                        '1:prompt name' \
                        '2:content' \
                        '--metadata[JSON metadata]:metadata'
                    ;;
            esac
            ;;
    esac
}

# Complete prompt names
_promptly_prompts() {
    local -a prompts
    if [[ -d .promptly ]]; then
        prompts=(${(f)"$(promptly list 2>/dev/null | grep -v 'Prompts on branch' | awk '{print $1":"$1}')"})
        _describe 'prompts' prompts
    fi
}

# Complete branch names
_promptly_branches() {
    local -a branches
    if [[ -d .promptly ]]; then
        # Try to get branches from database
        branches=(main development production)
        _describe 'branches' branches
    fi
}

# Complete chain names
_promptly_chains() {
    local -a chains
    if [[ -d .promptly/chains ]]; then
        chains=(${(f)"$(ls .promptly/chains/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//')"})
        _describe 'chains' chains
    fi
}

_promptly "$@"
