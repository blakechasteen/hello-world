-- =============================================================================
-- Example Neovim Configuration Integration with HoloLoom LSP
-- =============================================================================
--
-- This file shows how to integrate HoloLoom LSP into your existing Neovim
-- configuration. There are three methods shown below based on your setup:
--
-- 1. Using lazy.nvim plugin manager (RECOMMENDED)
-- 2. Using packer.nvim plugin manager
-- 3. Manual setup (if you don't use a plugin manager)
--
-- Choose the method that matches your Neovim setup.
--
-- =============================================================================

-- =============================================================================
-- METHOD 1: Using lazy.nvim (RECOMMENDED for modern Neovim setups)
-- =============================================================================
--
-- Add this to your lazy.nvim spec in your init.lua:
--
-- {
--     -- Optional: only load HoloLoom LSP in Python/TypeScript projects
--     'local-config/hololoom-lsp',
--     ft = { 'python', 'typescript', 'javascript', 'tsx', 'jsx' },
--     config = function()
--         require('hololoom').setup({
--             enabled = true,
--             auto_start = true,
--             log_level = "INFO",
--         })
--     end,
-- }
--
-- Then place the hololoom.lua file in:
--   ~/.config/nvim/lua/local-config/hololoom-lsp/init.lua
--
-- Or copy hololoom.lua to:
--   ~/.config/nvim/lua/hololoom.lua
--
-- Usage in your lazy spec:
-- {
--     'hololoom-lsp',
--     ft = { 'python', 'typescript', 'javascript' },
--     config = function()
--         require('hololoom').setup()
--     end,
-- }

-- =============================================================================
-- METHOD 2: Using packer.nvim
-- =============================================================================
--
-- Add this to your packer spec:
--
-- use {
--     'hololoom/lsp',  -- or local path
--     config = function()
--         require('hololoom').setup({
--             enabled = true,
--             log_level = "INFO",
--         })
--     end,
-- }
--
-- Then run :PackerSync

-- =============================================================================
-- METHOD 3: Manual Setup (Recommended if you don't use a plugin manager)
-- =============================================================================

-- Step 1: Copy hololoom.lua to your config directory
--   cp lsp-clients/neovim/hololoom.lua ~/.config/nvim/lua/hololoom.lua

-- Step 2: In your init.lua, add this configuration:

-- First, ensure nvim-lspconfig is installed manually or via plugin manager:
-- git clone https://github.com/neovim/nvim-lspconfig ~/.config/nvim/pack/packer/start/nvim-lspconfig

-- Then require and setup HoloLoom:
if pcall(require, 'hololoom') then
    require('hololoom').setup({
        -- Configuration options (all optional, these are defaults)
        enabled = true,
        auto_start = true,
        cmd_path = "python",  -- Path to Python executable
        log_level = "INFO",   -- "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        filetypes = { "python", "typescript", "javascript", "jsx", "tsx" },
        root_patterns = { ".git", "pyproject.toml", "package.json", "setup.py" },

        -- Completion settings
        completion = {
            enabled = true,
            trigger_characters = { "." },
        },

        -- Hover settings
        hover = {
            enabled = true,
            border = "rounded",
        },

        -- Definition settings
        definition = {
            enabled = true,
        },

        -- Symbol search settings
        symbol = {
            enabled = true,
        },
    })
else
    vim.notify("Failed to load HoloLoom LSP client", vim.log.levels.WARN)
end

-- =============================================================================
-- OPTIONAL: Additional LSP Configuration
-- =============================================================================

-- Configure Neovim diagnostics appearance
vim.diagnostic.config({
    virtual_text = true,
    signs = true,
    underline = true,
    update_in_insert = false,
    severity_sort = true,
})

-- Optional: Setup nvim-cmp for better completion (requires nvim-cmp plugin)
-- This provides autocompletion with window and documentation
local cmp_ok, cmp = pcall(require, "cmp")
if cmp_ok then
    cmp.setup({
        snippet = {
            expand = function(args)
                -- You need a snippet engine, e.g., luasnip
                -- require('luasnip').lsp_expand(args.body)
            end,
        },
        mapping = cmp.mapping.preset.insert({
            ['<C-b>'] = cmp.mapping.scroll_docs(-4),
            ['<C-f>'] = cmp.mapping.scroll_docs(4),
            ['<C-Space>'] = cmp.mapping.complete(),
            ['<C-e>'] = cmp.mapping.abort(),
            ['<CR>'] = cmp.mapping.confirm({ select = true }),
        }),
        sources = cmp.config.sources({
            { name = 'nvim_lsp' },
            { name = 'buffer' },
        }),
    })
end

-- =============================================================================
-- EXAMPLE: Custom Commands
-- =============================================================================

-- Command to reload the HoloLoom LSP server
vim.api.nvim_create_user_command(
    'HoloLoomReload',
    function()
        require('hololoom').reload()
    end,
    { desc = 'Reload HoloLoom LSP server' }
)

-- Command to stop the HoloLoom LSP server
vim.api.nvim_create_user_command(
    'HoloLoomStop',
    function()
        require('hololoom').stop()
    end,
    { desc = 'Stop HoloLoom LSP server' }
)

-- Command to restart the HoloLoom LSP server
vim.api.nvim_create_user_command(
    'HoloLoomRestart',
    function()
        require('hololoom').restart()
    end,
    { desc = 'Restart HoloLoom LSP server' }
)

-- =============================================================================
-- EXAMPLE: Custom Keybindings (override defaults if needed)
-- =============================================================================

-- You can add buffer-local keybindings or global keybindings here
-- Example: Map <leader>lm to format document
-- vim.keymap.set('n', '<leader>lm', vim.lsp.buf.format, { noremap = true, silent = true })

-- Example: Map <leader>ll to toggle diagnostics
-- vim.keymap.set('n', '<leader>ll', function()
--     vim.diagnostic.enable(not vim.diagnostic.is_enabled())
-- end, { noremap = true, silent = true })

print("HoloLoom LSP configuration loaded successfully!")
