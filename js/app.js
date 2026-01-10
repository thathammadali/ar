import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MindARThree } from 'mind-ar/mindar-image-three.prod.js';
import { ARPoseFilter } from './OneEuroFilter.js';

// Configuration
const CONFIG = {
    TARGET_PATH: 'targets.mind',
    SMOOTHING_MIN_CUTOFF: 0.01,
    SMOOTHING_BETA: 0.05,

    // Map Marker Index (from targets.mind) to Model File
    // SCALE: Now interpreted as "Multiplier of Normalized Size". 
    // 1.0 = Width of Marker. 0.5 = Half Width.
    MAPPINGS: [
        { index: 0, model: 'Stupid2.glb', scale: 4 },
        { index: 1, model: 'staircase.glb', scale: 4 },
        { index: 2, model: 'Buddha.glb', scale: 4 },
    ]
};

const setupAR = async () => {
    const statusEl = document.getElementById('status');
    const loadingEl = document.getElementById('loading');

    const setStatus = (msg) => {
        statusEl.innerText = msg;
        console.log(msg);
    };

    // 1. Initialize MindAR
    setStatus("Initializing MindAR...");
    const mindarThree = new MindARThree({
        container: document.body,
        imageTargetSrc: CONFIG.TARGET_PATH,
        // UI: Enable antialias for sharp edges
        rendererParameters: {
            antialias: true,
            alpha: true,
            precision: 'highp',
            logarithmicDepthBuffer: true // Helps with z-fighting/clipping
        }
    });

    // 2. Setup Three.js Scene
    const { renderer, scene, camera } = mindarThree;

    // FIX: Adjust Camera Near/Far planes to prevent clipping of small objects
    camera.near = 0.001;
    camera.far = 10000;
    camera.updateProjectionMatrix();

    // FIX: Pixelation - Set correct resolution and pixel ratio
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);

    // Handle window resize to keep resolution sharp
    window.addEventListener('resize', () => {
        renderer.setSize(window.innerWidth, window.innerHeight);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
    });

    // Lighting
    const light = new THREE.HemisphereLight(0xffffff, 0xbbbbff, 3.0); // Increased intensity
    scene.add(light);

    const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight.position.set(5, 5, 5);
    scene.add(dirLight);

    // 3. Load Models dynamically based on CONFIG.MAPPINGS
    setStatus("Loading Models...");
    const loader = new GLTFLoader();

    // Store logic objects for each marker
    const trackedItems = [];

    // Helper to load a single model
    const setupItem = async (mapping) => {
        try {
            console.log(`Loading ${mapping.model} for Marker ${mapping.index}...`);
            const gltf = await new Promise((resolve, reject) => {
                loader.load(mapping.model, resolve, undefined, reject);
            });

            const model = gltf.scene;

            // FIX: Texture Pixelation
            // Traverse model and set texture filters to Linear (Smooth)
            model.traverse((node) => {
                if (node.isMesh && node.material && node.material.map) {
                    node.material.map.minFilter = THREE.LinearFilter;
                    node.material.map.magFilter = THREE.LinearFilter;
                    node.material.needsUpdate = true;
                }
            });

            // --- Normalization Logic ---
            // 1. Compute Bounding Box
            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            console.log(`[${mapping.model}] Original Size:`, size);

            // 2. Normalize Scale (Fit largest dimension to 0.2 - visible but safe)
            const maxDim = Math.max(size.x, size.y, size.z);
            let scaleFactor = 0.2 / maxDim; // Baseline: 20% of marker width

            // Apply User Scale Override (e.g. 0.5 * 0.2 = 0.1 total)
            if (mapping.scale) {
                scaleFactor *= mapping.scale;
            }

            model.scale.set(scaleFactor, scaleFactor, scaleFactor);

            // FIX: Rotate 90 degrees on X axis per user request
            model.rotation.x = Math.PI / 2;

            // --- Center Geometry (Full Center) ---
            // If the model is "far away", it means X/Z are offset.
            // We MUST return them to 0 (marker center).
            model.position.set(
                -center.x * scaleFactor,
                -box.min.y * scaleFactor,
                -center.z * scaleFactor
            );

            console.log(`[${mapping.model}] Normalized Scale:`, scaleFactor);

            // --- Standard MindAR Setup ---
            const anchor = mindarThree.addAnchor(mapping.index);

            // Add Red Cube (Small) for center reference
            const debugGeo = new THREE.BoxGeometry(0.02, 0.02, 0.02);
            const debugMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const debugCube = new THREE.Mesh(debugGeo, debugMat);
            anchor.group.add(debugCube);

            // DEBUG: Axes Helper
            const axesHelper = new THREE.AxesHelper(0.2);
            anchor.group.add(axesHelper);

            // Add Model to Anchor Group
            // Push model slightly "back" (into the marker) to avoid near-plane clipping if marker is too close
            // model.position.z -= 0.1; 

            anchor.group.add(model);

            console.log(`Model ${mapping.index} attached to anchor.`);

            return {
                index: mapping.index,
                anchor: anchor
            };

        } catch (e) {
            console.error(`Error loading model for index ${mapping.index}:`, e);
            setStatus(`Failed to load ${mapping.model}`);
            return null;
        }
    };

    // Load all items in parallel
    const itemPromises = CONFIG.MAPPINGS.map(m => setupItem(m));
    const results = await Promise.all(itemPromises);

    // Filter out failed loads
    results.forEach(item => {
        if (item) trackedItems.push(item);
    });

    if (trackedItems.length === 0) {
        loadingEl.innerText = "No models loaded. Check console.";
        return;
    }

    setStatus("Models Ready. Starting AR...");

    // 6. Start Loop
    try {
        await mindarThree.start();
    } catch (e) {
        console.error("MindAR failed to start:", e);
        if (e.name === 'NotAllowedError' || e.message.includes('Permission denied')) {
            loadingEl.innerText = "Camera Access Denied.\nPlease allow camera permissions in your browser settings and refresh.";
            loadingEl.style.color = "red";
        } else {
            loadingEl.innerText = "Error starting AR: " + e.message;
        }
        return;
    }

    loadingEl.style.display = 'none';
    setStatus("Scanning...");

    renderer.setAnimationLoop(() => {
        let anyVisible = false;

        trackedItems.forEach(item => {
            if (item.anchor.group.visible) {
                anyVisible = true;
                // console.log(`Anchor ${item.index} Visible!`); // Commented out to reduce spam
            }
        });

        if (anyVisible) {
            setStatus("Tracking");
        } else {
            setStatus("Scanning...");
        }

        renderer.render(scene, camera);
    });
};

setupAR();
