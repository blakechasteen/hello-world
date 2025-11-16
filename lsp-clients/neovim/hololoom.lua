-- =============================================================================
-- HoloLoom Language Server Protocol (LSP) Client Configuration for Neovim
-- =============================================================================
--
-- This module configures the HoloLoom LSP server for Neovim, providing:
-- - Code completion from HoloLoom memories
-- - Hover information with entity context
-- - Go-to-definition via knowledge graph relationships
-- - Workspace symbol search (semantic)
-- - Diagnostic reports
--
-- Installation:
--   1. Add this module to your Neovim config directory
--   2. Require it from your init.lua
--   3. Call setup() to configure
--
-- Usage:
--   require('hololoom').setup()
--
-- Configuration:
--   require('hololoom').setup({
--       enabled = true,
--       auto_start = true,
--       cmd_path = "python",  -- Path to Python executable
--       log_level = "INFO",
--   })
--
-- =============================================================================

local M = {}

-- Default configuration
local defaults = {
    enabled = true,
    auto_start = true,
    cmd_path = "python",  -- Path to python executable
    cmd_args = { "-m", "HoloLoom.lsp.server" },
    log_level = "INFO",
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
}

-- Global configuration
local config = {}

-- Check if lspconfig is available
local function check_lspconfig()
    local ok, _ = pcall(require, "lspconfig")
    if not ok then
        vim.notify(
            "HoloLoom LSP requires 'nvim-lspconfig' plugin. Install it first.",
            vim.log.levels.ERROR
        )
        return false
    end
    return true
end

-- Find project root directory
local function find_root_dir(bufnr, patterns)
    local filepath = vim.api.nvim_buf_get_name(bufnr)
    if filepath == "" then
        return vim.fn.getcwd()
    end

    local lspconfig = require("lspconfig")
    for _, pattern in ipairs(patterns) do
        local root = lspconfig.util.root_pattern(pattern)(filepath)
        if root then
            return root
        end
    end

    return vim.fn.getcwd()
end

-- Setup LSP server on attach
local function on_attach(client, bufnr)
    -- Enable completion triggered with <c-x><c-o>
    vim.api.nvim_buf_set_option(bufnr, "omnifunc", "v:lua.vim.lsp.omnifunc")

    -- Keybindings for LSP functionality
    local opts = { noremap = true, silent = true, buffer = bufnr }

    -- Go to definition (gd)
    vim.keymap.set("n", "gd", vim.lsp.buf.definition, opts)

    -- Go to declaration (gD)
    vim.keymap.set("n", "gD", vim.lsp.buf.declaration, opts)

    -- Hover information (K)
    vim.keymap.set("n", "K", vim.lsp.buf.hover, opts)

    -- Find references (gr)
    vim.keymap.set("n", "gr", vim.lsp.buf.references, opts)

    -- Workspace symbol search (<leader>ws)
    vim.keymap.set("n", "<leader>ws", vim.lsp.buf.workspace_symbol, opts)

    -- Document symbols (<leader>ds)
    vim.keymap.set("n", "<leader>ds", vim.lsp.buf.document_symbol, opts)

    -- Rename symbol (<leader>rn)
    vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename, opts)

    -- Code actions (<leader>ca)
    vim.keymap.set("n", "<leader>ca", vim.lsp.buf.code_action, opts)

    -- Type definition (<leader>td)
    vim.keymap.set("n", "<leader>td", vim.lsp.buf.type_definition, opts)

    -- Implementation (<leader>gi)
    vim.keymap.set("n", "<leader>gi", vim.lsp.buf.implementation, opts)

    -- Signature help (<C-k> in insert mode)
    vim.keymap.set("i", "<C-k>", vim.lsp.buf.signature_help, opts)

    -- Diagnostic jump to next (<leader>dn)
    vim.keymap.set("n", "<leader>dn", vim.diagnostic.goto_next, opts)

    -- Diagnostic jump to previous (<leader>dp)
    vim.keymap.set("n", "<leader>dp", vim.diagnostic.goto_prev, opts)

    -- Show line diagnostics (<leader>dd)
    vim.keymap.set("n", "<leader>dd", vim.diagnostic.open_float, opts)

    -- Diagnostic setloclist (<leader>dl)
    vim.keymap.set("n", "<leader>dl", vim.diagnostic.setloclist, opts)

    -- Format document (<leader>fm)
    if client.server_capabilities.documentFormattingProvider then
        vim.keymap.set("n", "<leader>fm", vim.lsp.buf.format, opts)
    end

    -- Log attachment
    vim.notify(
        "HoloLoom LSP attached to buffer " .. bufnr,
        vim.log.levels.INFO
    )
end

-- Get LSP capabilities with completion support
local function get_capabilities()
    local cmp_nvim_lsp_ok, cmp_nvim_lsp = pcall(require, "cmp_nvim_lsp")

    if cmp_nvim_lsp_ok then
        return cmp_nvim_lsp.default_capabilities()
    else
        -- Fallback capabilities if cmp_nvim_lsp not available
        local capabilities = vim.lsp.protocol.make_client_capabilities()
        capabilities.textDocument.completion.completionItem.snippetSupport = true
        capabilities.textDocument.completion.completionItem.resolveSupport = {
            properties = { "documentation", "detail", "additionalTextEdits" },
        }
        return capabilities
    end
end

-- Main setup function
function M.setup(opts)
    -- Merge options with defaults
    config = vim.tbl_deep_extend("force", defaults, opts or {})

    -- Check if LSP is enabled
    if not config.enabled then
        return
    end

    -- Verify lspconfig is available
    if not check_lspconfig() then
        return
    end

    local lspconfig = require("lspconfig")
    local configs = require("lspconfig.configs")

    -- Define HoloLoom LSP configuration if not already defined
    if not configs.hololoom then
        configs.hololoom = {
            default_config = {
                cmd = { config.cmd_path, unpack(config.cmd_args) },
                filetypes = config.filetypes,
                root_dir = function(bufnr)
                    return find_root_dir(bufnr, config.root_patterns)
                end,
                settings = {},
                single_file_support = true,
                init_options = {
                    log_level = config.log_level,
                },
            },
        }
    end

    -- Setup the server with capabilities and on_attach
    lspconfig.hololoom.setup({
        capabilities = get_capabilities(),
        on_attach = on_attach,
        settings = {
            hololoom = {
                log_level = config.log_level,
                completion = config.completion,
                hover = config.hover,
                definition = config.definition,
                symbol = config.symbol,
            },
        },
    })

    -- Diagnostic configuration
    vim.diagnostic.config({
        virtual_text = true,
        signs = true,
        underline = true,
        update_in_insert = false,
        severity_sort = true,
    })

    -- Diagnostic sign configuration
    local signs = {
        Error = "✗",
        Warn = "⚠",
        Hint = "ℹ",
        Info = "ℹ",
    }

    for type, icon in pairs(signs) do
        local hl = "DiagnosticSign" .. type
        vim.fn.sign_define(hl, { text = icon, texthl = hl, numhl = hl })
    end

    -- Enable autoformat on save if not already configured
    vim.api.nvim_create_autocmd("BufWritePre", {
        group = vim.api.nvim_create_augroup("HoloLoomLSP", { clear = true }),
        pattern = table.concat(config.filetypes, ","),
        callback = function()
            -- Uncomment to enable autoformat
            -- vim.lsp.buf.format()
        end,
    })

    vim.notify("HoloLoom LSP configured successfully", vim.log.levels.INFO)
end

-- Get current configuration
function M.get_config()
    return config
end

-- Reload server
function M.reload()
    local lspconfig = require("lspconfig")
    lspconfig.hololoom.manager:try_add()
    vim.notify("HoloLoom LSP reloaded", vim.log.levels.INFO)
end

-- Stop server
function M.stop()
    local lspconfig = require("lspconfig")
    lspconfig.hololoom.manager:server_stop()
    vim.notify("HoloLoom LSP stopped", vim.log.levels.INFO)
end

-- Restart server
function M.restart()
    M.stop()
    vim.defer_fn(M.reload, 500)
    vim.notify("HoloLoom LSP restarted", vim.log.levels.INFO)
end

return M
