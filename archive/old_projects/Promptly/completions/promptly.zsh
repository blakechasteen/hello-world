#compdef promptly p

# Zsh completion for promptly

_promptly() {
    local -a commands
    commands=(
        'search:Search for prompts'
        'create:Create a new prompt'
        'update:Update an existing prompt'
        'delete:Delete a prompt'
        'list:List all prompts'
        'execute:Execute a prompt with recursive loops'
        'branch:Create a branch from a prompt'
        'merge:Merge prompt branches'
        'diff:Show differences between prompts'
        'analytics:View prompt analytics'
        'skills:Manage UltraPrompt skills'
        'help:Show help message'
        'version:Show version'
    )

    _arguments -C \
        '1: :->command' \
        '*:: :->args'

    case $state in
        command)
            _describe 'command' commands
            ;;
        args)
            case $words[1] in
                search)
                    _arguments \
                        '--tags[Filter by tags]:tags:' \
                        '--limit[Maximum results]:limit:' \
                        '--format[Output format]:format:(table json yaml)'
                    ;;
                create)
                    _arguments \
                        '--name[Prompt name]:name:' \
                        '--content[Prompt content]:content:' \
                        '--tags[Comma-separated tags]:tags:' \
                        '--from-file[Load from file]:file:_files'
                    ;;
                update)
                    _arguments \
                        '--name[Prompt name]:name:' \
                        '--content[New content]:content:' \
                        '--tags[New tags]:tags:' \
                        '--version[Version number]:version:'
                    ;;
                delete)
                    _arguments \
                        '--confirm[Confirm deletion]' \
                        '--force[Force deletion]'
                    ;;
                list)
                    _arguments \
                        '--tags[Filter by tags]:tags:' \
                        '--format[Output format]:format:(table json yaml tree)' \
                        '--sort[Sort by]:sort:(name date usage quality)'
                    ;;
                execute)
                    _arguments \
                        '--input[Input data]:input:' \
                        '--loop[Loop type]:loop:(REFINE CRITIQUE DECOMPOSE VERIFY EXPLORE HOFSTADTER)' \
                        '--iterations[Max iterations]:iterations:' \
                        '--quality-threshold[Quality threshold]:threshold:'
                    ;;
                branch)
                    _arguments \
                        '--from[Source prompt]:prompt:' \
                        '--description[Branch description]:description:'
                    ;;
                merge)
                    _arguments \
                        '--strategy[Merge strategy]:strategy:(latest best consensus manual)' \
                        '--resolve-conflicts[Conflict resolution]:resolution:'
                    ;;
                diff)
                    _arguments \
                        '--format[Output format]:format:(unified context side-by-side)' \
                        '--context[Context lines]:lines:'
                    ;;
                analytics)
                    _arguments \
                        '--summary[Show summary]' \
                        '--top[Top N prompts]:n:' \
                        '--recommendations[Show recommendations]'
                    ;;
                skills)
                    _arguments \
                        '--list[List all skills]' \
                        '--add[Add new skill]:skill:' \
                        '--remove[Remove skill]:skill:' \
                        '--update[Update skill]:skill:'
                    ;;
            esac
            ;;
    esac
}

_promptly "$@"
