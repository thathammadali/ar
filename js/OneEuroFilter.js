/**
 * OneEuroFilter.js
 * 
 * JavaScript port of the One Euro Filter.
 * Ref: https://github.com/mjanczak/OneEuroFilter
 */

class OneEuroFilter {
    constructor(minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        
        this.xPrev = null;
        this.dxPrev = null;
        this.tPrev = null;
    }

    /**
     * Filter the value.
     * @param {number|Array} x Input value (number or array of numbers)
     * @param {number} t Timestamp in seconds
     */
    filter(x, t = null) {
        if (t === null) {
            t = performance.now() / 1000.0;
        }

        if (this.xPrev === null) {
            this.xPrev = x;
            this.dxPrev = Array.isArray(x) ? x.map(() => 0) : 0;
            this.tPrev = t;
            return x;
        }

        const dt = t - this.tPrev;
        this.tPrev = t;

        // Avoid division by zero
        if (dt <= 0) return this.xPrev;

        // Compute alpha for derivative
        const alphaD = this.smoothingFactor(dt, this.dCutoff);
        
        let dx, dxHat;

        if (Array.isArray(x)) {
            dx = x.map((val, i) => (val - this.xPrev[i]) / dt);
            dxHat = dx.map((val, i) => this.exponentialSmoothing(alphaD, val, this.dxPrev[i]));
        } else {
            dx = (x - this.xPrev) / dt;
            dxHat = this.exponentialSmoothing(alphaD, dx, this.dxPrev);
        }

        // Compute cutoff using speed (magnitude of dxHat)
        // For array (vector), use Euclidean norm
        let speed = 0;
        if (Array.isArray(dxHat)) {
            speed = Math.sqrt(dxHat.reduce((sum, val) => sum + val * val, 0));
        } else {
            speed = Math.abs(dxHat);
        }

        const cutoff = this.minCutoff + this.beta * speed;
        const alpha = this.smoothingFactor(dt, cutoff);

        let xHat;
        if (Array.isArray(x)) {
            xHat = x.map((val, i) => this.exponentialSmoothing(alpha, val, this.xPrev[i]));
        } else {
            xHat = this.exponentialSmoothing(alpha, x, this.xPrev);
        }

        this.xPrev = xHat;
        this.dxPrev = dxHat;

        return xHat;
    }

    smoothingFactor(dt, cutoff) {
        const r = 2 * Math.PI * cutoff * dt;
        return r / (r + 1);
    }

    exponentialSmoothing(alpha, x, xPrev) {
        return alpha * x + (1 - alpha) * xPrev;
    }
}

/**
 * Wrapper for AR Pose Smoothing
 * Handles Vector3 (Position) and Quaternion (Rotation)
 */
class ARPoseFilter {
    constructor(minCutoff = 0.01, beta = 0.05, dCutoff = 1.0) {
        // Position is an array [x, y, z]
        this.posFilter = new OneEuroFilter(minCutoff, beta, dCutoff);
        // Quaternion is an array [x, y, z, w]
        this.rotFilter = new OneEuroFilter(minCutoff, beta, dCutoff);
    }

    filter(position, quaternion) {
        // Helper to convert Three.js Vector3/Quaternion to arrays if needed
        const posArr = [position.x, position.y, position.z];
        const rotArr = [quaternion.x, quaternion.y, quaternion.z, quaternion.w];

        const filteredPos = this.posFilter.filter(posArr);
        const filteredRot = this.rotFilter.filter(rotArr);

        // NORMALIZE Quaternion after filtering!
        // Linear interpolation of quaternions can result in non-unit length
        const len = Math.sqrt(
            filteredRot[0]*filteredRot[0] + 
            filteredRot[1]*filteredRot[1] + 
            filteredRot[2]*filteredRot[2] + 
            filteredRot[3]*filteredRot[3]
        );
        
        return {
            position: { x: filteredPos[0], y: filteredPos[1], z: filteredPos[2] },
            quaternion: { 
                x: filteredRot[0] / len, 
                y: filteredRot[1] / len, 
                z: filteredRot[2] / len, 
                w: filteredRot[3] / len 
            }
        };
    }
    
    reset() {
        this.posFilter = new OneEuroFilter(this.posFilter.minCutoff, this.posFilter.beta, this.posFilter.dCutoff);
        this.rotFilter = new OneEuroFilter(this.rotFilter.minCutoff, this.rotFilter.beta, this.rotFilter.dCutoff);
    }
}

export { OneEuroFilter, ARPoseFilter };
