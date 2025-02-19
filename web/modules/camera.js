let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let currentFacingMode = "environment"; // 기본 후면 카메라

async function startCamera(facingMode) {
  let constraints = {
    video: {
        facingMode: facingMode,
        width: { ideal: 9999 },
        height: { ideal: 9999 },
    }, // 최고 해상도로 조절
  };

  try {
    let stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
  } catch (err) {
    console.error("카메라 접근 오류:", err);
  }
}

function updateCanvas() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  requestAnimationFrame(updateCanvas);
}

document.getElementById("switchCam").addEventListener("click", () => {
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  startCamera(currentFacingMode);
});

startCamera(currentFacingMode);
video.addEventListener("loadedmetadata", updateCanvas);
