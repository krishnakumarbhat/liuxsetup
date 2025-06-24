local wezterm = require 'wezterm'

-- Show date/time in the right status bar
wezterm.on("update-right-status", function(window)
  local date = wezterm.strftime("%Y-%m-%d %H:%M:%S")
  window:set_right_status(wezterm.format({ { Text = date } }))
end)

local config = {
  -- Launch fish shell by default
  default_prog = {"/usr/bin/fish"},

  -- Appearance
  font = wezterm.font_with_fallback({ "JetBrains Mono", "Fira Code", "Noto Sans Mono" }),
  font_size = 14,
  color_scheme = "Dark Base",
  window_background_opacity = 0.9,
  text_background_opacity = 0.9,
  window_decorations = "RESIZE|TITLE",
  window_padding = {
    left = 8, right = 8, top = 4, bottom = 4,
  },

  -- Tab Bar
  enable_tab_bar = true,
  use_fancy_tab_bar = false,
  hide_tab_bar_if_only_one_tab = false,
  tab_bar_at_bottom = false,

  -- Productivity Key Bindings
  keys = {
    -- Cycle through tabs (Ctrl+Tab for next, Ctrl+Shift+Tab for prev)
    { key = "Tab", mods = "CTRL", action = wezterm.action.ActivateTabRelative(1) },
    { key = "Tab", mods = "CTRL|SHIFT", action = wezterm.action.ActivateTabRelative(-1) },
    -- New tab with Ctrl+N (in addition to Ctrl+T)
    { key = "n", mods = "CTRL", action = wezterm.action.SpawnTab "CurrentPaneDomain" },
    -- Split pane horizontally (Ctrl+Alt+Right)
    { key = "RightArrow", mods = "CTRL|ALT", action = wezterm.action.SplitHorizontal { domain = "CurrentPaneDomain" } },
    -- Split pane vertically (Ctrl+Alt+Down)
    { key = "DownArrow", mods = "CTRL|ALT", action = wezterm.action.SplitVertical { domain = "CurrentPaneDomain" } },
    -- Move between panes (Ctrl+Shift+Arrows)
    { key = "LeftArrow", mods = "CTRL|SHIFT", action = wezterm.action.ActivatePaneDirection "Left" },
    { key = "RightArrow", mods = "CTRL|SHIFT", action = wezterm.action.ActivatePaneDirection "Right" },
    { key = "UpArrow", mods = "CTRL|SHIFT", action = wezterm.action.ActivatePaneDirection "Up" },
    { key = "DownArrow", mods = "CTRL|SHIFT", action = wezterm.action.ActivatePaneDirection "Down" },
    -- Move tab left/right (Shift+Alt+Arrows)
    { key = "LeftArrow", mods = "SHIFT|ALT", action = wezterm.action.MoveTabRelative(-1) },
    -- cycle through panes (Ctrl+`)
    { key = "`", mods = "CTRL", action = wezterm.action.ActivatePaneDirection("Next") },
    -- { key = "`", mods = "CTRL", action = wezterm.action.ActivatePaneDirection("Right") },
    { key = "RightArrow", mods = "SHIFT|ALT", action = wezterm.action.MoveTabRelative(1) },
    -- New tab (Ctrl+n)
    { key = "n", mods = "CTRL", action = wezterm.action.SpawnTab "CurrentPaneDomain" },
    -- Close tab (Ctrl+W)
    { key = "w", mods = "CTRL", action = wezterm.action.CloseCurrentTab { confirm = true } },
    -- Quick open config in editor (Ctrl+,)
    { key = ",", mods = "CTRL", action = wezterm.action.SpawnCommandInNewWindow {
        args = { os.getenv("EDITOR") or "vim", os.getenv("WEZTERM_CONFIG_FILE") or "~/.config/wezterm/wezterm.lua" },
      },
    },
  },

  -- Leader key (optional, for modal keybindings)
  -- leader = { key = "a", mods = "CTRL", timeout_milliseconds = 1000 },

  -- Other productivity settings
  automatically_reload_config = true,
  adjust_window_size_when_changing_font_size = false,
  scrollback_lines = 5000,
  term = "xterm-256color",
}

return config
