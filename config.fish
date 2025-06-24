# ~/.config/fish/config.fish

# 1. Disable the default Fish greeting message
set -g fish_greeting ""

# 2. Run commands only in interactive sessions
if status is-interactive
    # Enable syntax highlighting (if installed)
    # fisher install PatrickF1/fish-syntax  # Uncomment if you want syntax highlighting
    if status is-interactive
        cd /home/pope
end
    # Enable autosuggestions (if installed)
    # fisher install PatrickF1/fish-autosuggestions  # Uncomment if you want autosuggestions
    
    # Set up abbreviations for common commands
    abbr gs 'git status'
    abbr ga 'git add'
    abbr gc 'git commit'
    abbr gp 'git push'
    
    # Set up useful environment variables
    set -gx EDITOR nano
    set -gx VISUAL nano

    # Set up a nice pager for man pages
    set -gx MANPAGER 'less -X'

    # Enable universal variables for history sharing across sessions
    set -U fish_history 5000

    # Add your custom functions or aliases here
end

# 3. Custom prompt showing username, current directory, and git branch if any
function fish_prompt
    # Username and current directory with colon separator
    set user_dir (string join '' -- (whoami) ":" (basename (pwd)))

    # Get current git branch if inside a git repo
    set -l git_branch ''
    if type -q git
        set -l branch_name (git symbolic-ref --short HEAD ^/dev/null 2>/dev/null)
        if test -n "$branch_name"
            set git_branch " (git:$branch_name)"
        end
    end

    # Compose prompt with no spaces: user:dir(gitbranch)>
    echo -n $user_dir$git_branch'> '
end

# 4. Useful key bindings (example: Ctrl+R for history search with fzf if installed)
function fish_user_key_bindings
    # Check if fzf is installed and fzf.fish plugin loaded
    if type -q fzf
        # Bind Ctrl+R to fuzzy history search
        bind \cr 'fzf-history-widget'
    end 
end

# 5. Set up colorful directory listing alias
alias ls='ls --color=auto -lah'

# 6. Set up a quick directory jump function (requires 'z' or 'fasd' installed)
function j
    if test (count $argv) -eq 0
        cd /media/pope/projecteo/
    else
        z $argv[1]
    end
end

