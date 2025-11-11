const liveVideoFeed = document.getElementById('liveVideoFeed');
const liveCameraButton = document.getElementById('liveCameraButton');
const cameraStatus = document.getElementById('cameraStatus');
const captureButton = document.getElementById('captureButton');
const liveCanvas = document.getElementById('liveCanvas');
let currentStream;

// Function to start the camera stream
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        currentStream = stream;
        
        liveVideoFeed.srcObject = stream;
        liveVideoFeed.classList.remove('hidden');
        cameraStatus.classList.add('hidden');
        liveCameraButton.innerHTML = '<i class="fas fa-stop mr-2"></i> Stop Live Camera';
        captureButton.classList.remove('hidden');

    } catch (err) {
        console.error("Error accessing camera: ", err);
        cameraStatus.textContent = 'Camera access denied or device not found.';
        cameraStatus.classList.remove('hidden');
        liveVideoFeed.classList.add('hidden');
    }
}

// Function to stop the camera stream
function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    liveVideoFeed.classList.add('hidden');
    liveVideoFeed.srcObject = null;
    cameraStatus.textContent = 'Click "Start Live Camera" to begin.';
    cameraStatus.classList.remove('hidden');
    liveCameraButton.innerHTML = '<i class="fas fa-video mr-2"></i> Start Live Camera';
    captureButton.classList.add('hidden');
}

// Captures a single frame, sends it to the Flask API, and updates the dashboard.
async function captureAndSendFrame() {
    const context = liveCanvas.getContext('2d');
    liveCanvas.width = liveVideoFeed.videoWidth;
    liveCanvas.height = liveVideoFeed.videoHeight;
    context.drawImage(liveVideoFeed, 0, 0, liveCanvas.width, liveCanvas.height);

    captureButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Detecting...';
    captureButton.disabled = true;

    liveCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'live_frame.jpg');

        try {
            const response = await fetch('/api/live_detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Live analysis failed: ' + response.statusText);
            }
            
            const result = await response.json();
            window.updateDashboard(result); // Calls the global function in index.html

        } catch (error) {
            console.error('Error in live detection:', error);
            document.getElementById('diseaseName').textContent = 'Live Error!';
            document.getElementById('diseaseDetails').textContent = 'Detection failed.';
        } finally {
            captureButton.innerHTML = '<i class="fas fa-camera mr-2"></i> Capture Frame';
            captureButton.disabled = false;
        }

    }, 'image/jpeg');
}

// Event Listeners
liveCameraButton.addEventListener('click', () => {
    if (liveVideoFeed.srcObject) {
        stopCamera();
    } else {
        startCamera();
    }
});

captureButton.addEventListener('click', captureAndSendFrame);