import { defineConfig } from 'vite';

export default defineConfig({
    base: './',
    // publicDir defaults to 'public', which is where we put asserts
    server: {
        host: true, // Listen on all addresses
        port: 8000
    },
    build: {
        outDir: 'dist',
        assetsDir: 'assets'
    }
});
