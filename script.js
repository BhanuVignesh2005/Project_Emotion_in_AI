document.getElementById('ttsForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const text = document.getElementById('textInput').value;
    const voice = document.getElementById('voiceSelect').value;
    const emotion = document.getElementById('emotionSelect').value;

    start3DAnimation();

    const response = await fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text, emotion, voice_id: voice })
    });

    const data = await response.json();
    if (data.audio_path) {
        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = data.audio_path;
        audioPlayer.style.display = 'block';
        audioPlayer.play();
        visualizeAudio(data.audio_path);
        console.log('Predicted Emotion:', data.predicted_emotion);
        
        audioPlayer.onended = stop3DAnimation;
    } else {
        alert('Error generating speech: ' + data.error);
        stop3DAnimation();
    }
});

function visualizeAudio(url) {
    const canvas = document.getElementById('waveform');
    const canvasCtx = canvas.getContext('2d');
    canvas.style.display = 'block';

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioElement = document.getElementById('audioPlayer');
    const track = audioContext.createMediaElementSource(audioElement);
    const analyser = audioContext.createAnalyser();

    track.connect(analyser);
    analyser.connect(audioContext.destination);

    analyser.fftSize = 2048;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        const WIDTH = canvas.width;
        const HEIGHT = canvas.height;

        requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = 'rgb(200, 200, 200)';
        canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

        canvasCtx.beginPath();

        const sliceWidth = WIDTH * 1.0 / bufferLength;
        let x = 0;

        for(let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * HEIGHT/2;

            if(i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height/2);
        canvasCtx.stroke();
    }

    draw();
}

let scene, camera, renderer, cube, animationId;

function start3DAnimation() {
    const animationDiv = document.getElementById('3dAnimation');
    animationDiv.style.display = 'block';

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, animationDiv.clientWidth / animationDiv.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(animationDiv.clientWidth, animationDiv.clientHeight);
    animationDiv.appendChild(renderer.domElement);

    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    camera.position.z = 5;

    function animate() {
        animationId = requestAnimationFrame(animate);

        cube.rotation.x += 0.01;
        cube.rotation.y += 0.01;

        renderer.render(scene, camera);
    }

    animate();
}

function stop3DAnimation() {
    cancelAnimationFrame(animationId);

    const animationDiv = document.getElementById('3dAnimation');
    animationDiv.innerHTML = '';
    animationDiv.style.display = 'none';
}







