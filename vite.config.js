import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
    base: './',
    server: {
        host: true,
        port: 8000,
        https: true
    },
    plugins: [
        basicSsl()
    ],
    build: {
        outDir: 'dist',
        assetsDir: 'assets'
    }
});
