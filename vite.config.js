import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig(({ command }) => {
    return {
        base: './',
        server: {
            host: true,
            port: 8000,
            https: true
        },
        plugins: command === 'serve' ? [basicSsl()] : [],
        build: {
            outDir: 'dist',
            assetsDir: 'assets'
        }
    };
});
