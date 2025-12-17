import type { Config } from 'tailwindcss';
import designSystemPreset from '@hololoom/design-system/tailwind';

const config: Config = {
  presets: [designSystemPreset],
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
    '../../packages/design-system/src/**/*.{js,ts,jsx,tsx}',
  ],
};

export default config;
