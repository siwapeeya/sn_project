const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const socket = io();

// ฟังก์ชันจัดการดึงพิกัด
function extractKeypoints(results) {
    const extract = (landmarks, count) => {
        if (!landmarks) return new Array(count * 3).fill(0);
        return landmarks.map(l => [l.x, l.y, l.z]).flat();
    };
    const lh = extract(results.leftHandLandmarks, 21);
    const rh = extract(results.rightHandLandmarks, 21);
    const pose = extract(results.poseLandmarks, 33).slice(0, 99);
    return [...lh, ...rh, ...pose]; // รวมได้ 225 ค่า
}

const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});
holistic.setOptions({
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

holistic.onResults((results) => {
    // วาดเส้น MediaPipe
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
    drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#FF0000', lineWidth: 2});
    drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#0000FF', lineWidth: 2});
    canvasCtx.restore();

    // ส่ง keypoints ไป server
    const keypoints = extractKeypoints(results);
    socket.emit('predict_keypoints', keypoints);
});

const camera = new Camera(videoElement, {
    onFrame: async () => { await holistic.send({image: videoElement}); },
    width: 640, height: 480
});
camera.start();

// รับผลทายจาก server
socket.on('response', (data) => {
    document.getElementById('word').innerText = data.detected;
    document.getElementById('conf').innerText = (data.confidence * 100).toFixed(2);
});