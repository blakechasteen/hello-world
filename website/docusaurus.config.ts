import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'HoloLoom',
  tagline: 'Neural Decision-Making System with Multi-Department AI',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // Production URL - update when you have a domain
  url: 'https://hololoom.pages.dev',
  baseUrl: '/',

  // Cloudflare Pages deployment
  organizationName: 'hololoom',
  projectName: 'mythRL',

  // Be lenient during migration - switch to 'throw' once links are fixed
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Markdown configuration
  markdown: {
    mermaid: true,
    // Use plain markdown format to avoid MDX parsing issues with existing docs
    format: 'md',
  },
  themes: [
    '@docusaurus/theme-mermaid',
    // Local search - no Algolia account needed
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        indexDocs: true,
        indexBlog: true,
        docsRouteBasePath: ['docs', 'api', 'archive'],
        highlightSearchTermsOnTargetPage: true,
        searchResultLimits: 10,
        searchResultContextMaxLength: 50,
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          // ZERO-COPY: Read directly from existing docs/ folder
          path: '../docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/blakestorm/mythRL/edit/main/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
          // Include all markdown files
          include: ['**/*.md', '**/*.mdx'],
          exclude: [
            '**/_*.{js,jsx,ts,tsx,md,mdx}',
            '**/_*/**',
            '**/*.test.{js,jsx,ts,tsx}',
            '**/__tests__/**',
            // Exclude directories with problematic MDX content
            '**/archive/**',
            '**/sessions/**',
            '**/completion_reports/**',
          ],
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/blakestorm/mythRL/edit/main/',
          blogTitle: 'HoloLoom Changelog',
          blogDescription: 'Updates and releases for HoloLoom',
          postsPerPage: 10,
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    // ZERO-COPY: API docs from HoloLoom/*/README.md
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'api',
        path: '../HoloLoom',
        routeBasePath: 'api',
        sidebarPath: './sidebarsApi.ts',
        include: ['**/README.md', '**/README_*.md'],
        exclude: [
          '**/tests/**',
          '**/__pycache__/**',
          '**/node_modules/**',
          '**/.pytest_cache/**',
          '**/venv/**',
          '**/.venv/**',
        ],
      },
    ],
    // ZERO-COPY: Archive docs
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'archive',
        path: '../archive',
        routeBasePath: 'archive',
        sidebarPath: './sidebarsArchive.ts',
        include: ['**/*.md'],
        exclude: [
          '**/node_modules/**',
          '**/__pycache__/**',
        ],
      },
    ],
  ],

  themeConfig: {
    image: 'img/hololoom-social-card.jpg',

    // Dark mode enabled by default
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    // Announcement bar (optional)
    announcementBar: {
      id: 'welcome',
      content: '🚀 Welcome to HoloLoom Documentation! <a href="/docs">Get Started</a>',
      backgroundColor: '#1a1a2e',
      textColor: '#ffffff',
      isCloseable: true,
    },

    navbar: {
      title: 'HoloLoom',
      logo: {
        alt: 'HoloLoom Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/api/README',
          label: 'API',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Changelog',
          position: 'left',
        },
        {
          to: '/archive/README',
          label: 'Archive',
          position: 'left',
        },
        {
          href: 'https://github.com/blakestorm/mythRL/discussions',
          label: 'Forum',
          position: 'right',
        },
        {
          href: 'https://github.com/blakestorm/mythRL',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting-started/quickstart',
            },
            {
              label: 'Architecture',
              to: '/docs/architecture',
            },
            {
              label: 'API Reference',
              to: '/api/README',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/blakestorm/mythRL/discussions',
            },
            {
              label: 'GitHub Issues',
              href: 'https://github.com/blakestorm/mythRL/issues',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Changelog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/blakestorm/mythRL',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} HoloLoom. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'yaml', 'typescript', 'javascript'],
    },

    // Table of contents configuration
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
