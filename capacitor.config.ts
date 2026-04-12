import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'jp.shiretto.cat',
  appName: 'ShirettoCat',
  webDir: 'dist',
  server: {
    androidScheme: 'https'
  }
};

export default config;
