# GitHub Pages Site

This directory contains the GitHub Pages site for the LMS Orchestration Ecosystem.

## Enabling GitHub Pages

1. Go to your repository settings
2. Navigate to **Pages** section
3. Under **Source**, select:
   - Branch: `main` (or your deployment branch)
   - Folder: `/docs`
4. Click **Save**
5. Wait 1-2 minutes for deployment

Your site will be available at: `https://blakechasteen.github.io/hello-world/`

## Local Development

To preview locally:

```bash
# Simple HTTP server
cd docs
python -m http.server 8000

# Or use live-server (npm)
npx live-server docs/
```

Then open: http://localhost:8000

## Features

The GitHub Pages site includes:

- **Hero Section**: Overview and CTA buttons
- **Features Grid**: 6 key features with icons
- **Architecture Diagram**: Visual system overview
- **Plugin Ecosystem**: 5 categories with examples
- **Statistics**: Key metrics
- **Comparison Table**: vs Canvas, Moodle, Google Classroom
- **Roadmap**: 4-phase implementation plan
- **Responsive Design**: Mobile-friendly
- **Smooth Scrolling**: Anchor link navigation
- **Animation Effects**: Fade-in on scroll

## Customization

Edit `index.html` to customize:

- Colors: Modify CSS variables in `:root`
- Content: Update HTML sections
- Links: Point to your actual GitHub repo
- Images: Add to `docs/` directory

## SEO

The site includes:

- Meta description
- Semantic HTML
- Proper heading hierarchy
- Alt text for icons (emoji)
- Mobile viewport configuration

## License

Open source under MIT License (same as parent project).
