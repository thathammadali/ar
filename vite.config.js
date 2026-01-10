import { defineConfig } from 'vite';

export default defineConfig({
    // publicDir defaults to 'public', which is where we put assets
    server: {
        host: true, // Listen on all addresses
        port: 8000
    },
    build: {
        outDir: 'dist',
        assetsDir: 'assets'
    }
});
