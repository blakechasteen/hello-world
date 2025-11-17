# Fish completion for promptly

# Main commands
complete -c promptly -f -n "__fish_use_subcommand" -a "init" -d "Initialize a new promptly repository"
complete -c promptly -f -n "__fish_use_subcommand" -a "add" -d "Add a new prompt or update existing"
complete -c promptly -f -n "__fish_use_subcommand" -a "get" -d "Get a prompt by name"
complete -c promptly -f -n "__fish_use_subcommand" -a "list" -d "List all prompts"
complete -c promptly -f -n "__fish_use_subcommand" -a "log" -d "Show commit history"
complete -c promptly -f -n "__fish_use_subcommand" -a "branch" -d "Create a new branch"
complete -c promptly -f -n "__fish_use_subcommand" -a "checkout" -d "Switch to a different branch"
complete -c promptly -f -n "__fish_use_subcommand" -a "status" -d "Show repository status"
complete -c promptly -f -n "__fish_use_subcommand" -a "show" -d "Show prompt with syntax highlighting"
complete -c promptly -f -n "__fish_use_subcommand" -a "diff" -d "Compare two versions"
complete -c promptly -f -n "__fish_use_subcommand" -a "export" -d "Export repository"
complete -c promptly -f -n "__fish_use_subcommand" -a "import" -d "Import prompts"
complete -c promptly -f -n "__fish_use_subcommand" -a "eval" -d "Evaluate prompts"
complete -c promptly -f -n "__fish_use_subcommand" -a "chain" -d "Manage prompt chains"
complete -c promptly -f -n "__fish_use_subcommand" -a "wizard" -d "Interactive setup wizards"
complete -c promptly -f -n "__fish_use_subcommand" -a "tui" -d "Launch terminal UI"
complete -c promptly -f -n "__fish_use_subcommand" -a "interactive" -d "Start interactive REPL"
complete -c promptly -f -n "__fish_use_subcommand" -a "help" -d "Show help"
complete -c promptly -f -n "__fish_use_subcommand" -a "version" -d "Show version"

# eval subcommands
complete -c promptly -f -n "__fish_seen_subcommand_from eval" -a "run" -d "Run evaluation"
complete -c promptly -f -n "__fish_seen_subcommand_from eval" -a "list" -d "List evaluations"

# chain subcommands
complete -c promptly -f -n "__fish_seen_subcommand_from chain" -a "create" -d "Create a chain"
complete -c promptly -f -n "__fish_seen_subcommand_from chain" -a "run" -d "Run a chain"
complete -c promptly -f -n "__fish_seen_subcommand_from chain" -a "list" -d "List chains"

# wizard subcommands
complete -c promptly -f -n "__fish_seen_subcommand_from wizard" -a "project" -d "Project setup wizard"
complete -c promptly -f -n "__fish_seen_subcommand_from wizard" -a "prompt" -d "Prompt creation wizard"
complete -c promptly -f -n "__fish_seen_subcommand_from wizard" -a "chain" -d "Chain creation wizard"
complete -c promptly -f -n "__fish_seen_subcommand_from wizard" -a "evaluation" -d "Evaluation setup wizard"
complete -c promptly -f -n "__fish_seen_subcommand_from wizard" -a "template" -d "Template wizard"

# Options for specific commands
complete -c promptly -f -n "__fish_seen_subcommand_from add" -l "metadata" -s "m" -d "JSON metadata"
complete -c promptly -f -n "__fish_seen_subcommand_from get" -l "version" -s "v" -d "Specific version"
complete -c promptly -f -n "__fish_seen_subcommand_from get" -l "commit" -s "c" -d "Specific commit"
complete -c promptly -f -n "__fish_seen_subcommand_from list" -l "branch" -s "b" -d "List from branch"
complete -c promptly -f -n "__fish_seen_subcommand_from log" -l "name" -s "n" -d "Show log for prompt"
complete -c promptly -f -n "__fish_seen_subcommand_from log" -l "limit" -s "l" -d "Number of commits"
complete -c promptly -f -n "__fish_seen_subcommand_from export" -l "format" -s "f" -a "json yaml" -d "Output format"

# Dynamic completions for prompt names
function __fish_promptly_prompts
    if test -d .promptly
        promptly list 2>/dev/null | grep -v "Prompts on branch" | awk '{print $1}'
    end
end

# Dynamic completions for chains
function __fish_promptly_chains
    if test -d .promptly/chains
        for chain in .promptly/chains/*.yaml
            basename $chain .yaml
        end
    end
end

# Complete prompt names for get, show, log
complete -c promptly -f -n "__fish_seen_subcommand_from get show log" -a "(__fish_promptly_prompts)"

# Complete chain names for chain run
complete -c promptly -f -n "__fish_seen_subcommand_from chain; and __fish_seen_subcommand_from run" -a "(__fish_promptly_chains)"
