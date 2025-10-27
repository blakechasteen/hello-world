# Fish completion for promptly

# Main commands
complete -c promptly -f -n '__fish_use_subcommand' -a search -d 'Search for prompts'
complete -c promptly -f -n '__fish_use_subcommand' -a create -d 'Create a new prompt'
complete -c promptly -f -n '__fish_use_subcommand' -a update -d 'Update an existing prompt'
complete -c promptly -f -n '__fish_use_subcommand' -a delete -d 'Delete a prompt'
complete -c promptly -f -n '__fish_use_subcommand' -a list -d 'List all prompts'
complete -c promptly -f -n '__fish_use_subcommand' -a execute -d 'Execute a prompt'
complete -c promptly -f -n '__fish_use_subcommand' -a branch -d 'Create a branch'
complete -c promptly -f -n '__fish_use_subcommand' -a merge -d 'Merge branches'
complete -c promptly -f -n '__fish_use_subcommand' -a diff -d 'Show differences'
complete -c promptly -f -n '__fish_use_subcommand' -a analytics -d 'View analytics'
complete -c promptly -f -n '__fish_use_subcommand' -a skills -d 'Manage skills'
complete -c promptly -f -n '__fish_use_subcommand' -a help -d 'Show help'
complete -c promptly -f -n '__fish_use_subcommand' -a version -d 'Show version'

# Short alias
complete -c p -w promptly

# Search options
complete -c promptly -n '__fish_seen_subcommand_from search' -l tags -d 'Filter by tags'
complete -c promptly -n '__fish_seen_subcommand_from search' -l limit -d 'Maximum results'
complete -c promptly -n '__fish_seen_subcommand_from search' -l format -d 'Output format' -a 'table json yaml'

# Create options
complete -c promptly -n '__fish_seen_subcommand_from create' -l name -d 'Prompt name'
complete -c promptly -n '__fish_seen_subcommand_from create' -l content -d 'Prompt content'
complete -c promptly -n '__fish_seen_subcommand_from create' -l tags -d 'Tags'
complete -c promptly -n '__fish_seen_subcommand_from create' -l from-file -d 'Load from file'

# Update options
complete -c promptly -n '__fish_seen_subcommand_from update' -l name -d 'Prompt name'
complete -c promptly -n '__fish_seen_subcommand_from update' -l content -d 'New content'
complete -c promptly -n '__fish_seen_subcommand_from update' -l tags -d 'New tags'
complete -c promptly -n '__fish_seen_subcommand_from update' -l version -d 'Version'

# Delete options
complete -c promptly -n '__fish_seen_subcommand_from delete' -l confirm -d 'Confirm deletion'
complete -c promptly -n '__fish_seen_subcommand_from delete' -l force -d 'Force deletion'

# List options
complete -c promptly -n '__fish_seen_subcommand_from list' -l tags -d 'Filter by tags'
complete -c promptly -n '__fish_seen_subcommand_from list' -l format -d 'Output format' -a 'table json yaml tree'
complete -c promptly -n '__fish_seen_subcommand_from list' -l sort -d 'Sort by' -a 'name date usage quality'

# Execute options
complete -c promptly -n '__fish_seen_subcommand_from execute' -l input -d 'Input data'
complete -c promptly -n '__fish_seen_subcommand_from execute' -l loop -d 'Loop type' -a 'REFINE CRITIQUE DECOMPOSE VERIFY EXPLORE HOFSTADTER'
complete -c promptly -n '__fish_seen_subcommand_from execute' -l iterations -d 'Max iterations'
complete -c promptly -n '__fish_seen_subcommand_from execute' -l quality-threshold -d 'Quality threshold'

# Branch options
complete -c promptly -n '__fish_seen_subcommand_from branch' -l from -d 'Source prompt'
complete -c promptly -n '__fish_seen_subcommand_from branch' -l description -d 'Branch description'

# Merge options
complete -c promptly -n '__fish_seen_subcommand_from merge' -l strategy -d 'Merge strategy' -a 'latest best consensus manual'
complete -c promptly -n '__fish_seen_subcommand_from merge' -l resolve-conflicts -d 'Conflict resolution'

# Diff options
complete -c promptly -n '__fish_seen_subcommand_from diff' -l format -d 'Output format' -a 'unified context side-by-side'
complete -c promptly -n '__fish_seen_subcommand_from diff' -l context -d 'Context lines'

# Analytics options
complete -c promptly -n '__fish_seen_subcommand_from analytics' -l summary -d 'Show summary'
complete -c promptly -n '__fish_seen_subcommand_from analytics' -l top -d 'Top N prompts'
complete -c promptly -n '__fish_seen_subcommand_from analytics' -l recommendations -d 'Show recommendations'

# Skills options
complete -c promptly -n '__fish_seen_subcommand_from skills' -l list -d 'List all skills'
complete -c promptly -n '__fish_seen_subcommand_from skills' -l add -d 'Add new skill'
complete -c promptly -n '__fish_seen_subcommand_from skills' -l remove -d 'Remove skill'
complete -c promptly -n '__fish_seen_subcommand_from skills' -l update -d 'Update skill'
