import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MindARThree } from 'mind-ar/dist/mindar-image-three.prod.js';
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
        { index: 1, model: 'Temple.glb', scale: 3 },
        { index: 0, model: 'Staircase.glb', scale: 3 },
        { index: 2, model: 'Buddha.glb', scale: 3 },
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

    // Lighting - Boosted for better visibility
    const light = new THREE.HemisphereLight(0xffffff, 0xbbbbff, 5.0);
    scene.add(light);

    const dirLight = new THREE.DirectionalLight(0xffffff, 4.0);
    dirLight.position.set(5, 5, 5);
    scene.add(dirLight);

    // UI: Brightness Slider Logic
    const slider = document.getElementById('brightness');
    if (slider) {
        slider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            light.intensity = val;
            dirLight.intensity = val * 0.8; // Increased multiplier
        });
    }

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

            // FIX: Robust Centering with Wrapper
            // 1. Create a Wrapper Group (To be attached to Anchor)
            const wrapper = new THREE.Group();
            wrapper.add(model);

            // 2. Apply Rotations (User Request)
            model.rotation.x = Math.PI / 2; // 90 deg X Rotation

            // 3. Compute Box of result (in Wrapper Space)
            model.updateMatrixWorld(true);
            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            console.log(`[${mapping.model}] Rotated Size:`, size);

            // 4. Calculate Scale
            // Fit largest dimension to marker size * multiplier
            const maxDim = Math.max(size.x, size.y, size.z);
            let scaleFactor = (0.2 * 2) / maxDim; // Baseline: 0.4 (2x marker)
            if (mapping.scale) scaleFactor *= mapping.scale;

            model.scale.set(scaleFactor, scaleFactor, scaleFactor);

            // 5. Center Logic
            // Shift model in Wrapper Space so its VISUAL bottom-center is at (0,0,0)
            // Since we scaled the model, we must scale the offset too.
            model.position.set(
                -center.x * scaleFactor,
                -box.min.y * scaleFactor, // Align Bottom
                -center.z * scaleFactor
            );

            console.log(`[${mapping.model}] Final Scale: ${scaleFactor}`);

            // --- Standard MindAR Setup ---
            const anchor = mindarThree.addAnchor(mapping.index);

            // Add Wrapper to Anchor
            anchor.group.add(wrapper);



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
