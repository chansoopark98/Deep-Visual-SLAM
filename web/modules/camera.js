const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let currentFacingMode = "environment";
let currentStream = null;

// 새로운 스트림 시작 전 기존 스트림을 중지하여 리소스를 해제합니다.
async function startCamera(facingMode) {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }
  const constraints = {
    video: {
      facingMode,
      width: { ideal: 9999 },
      height: { ideal: 9999 },
    },
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;
    await video.play();
  } catch (err) {
    console.error("카메라 접근 오류:", err);
  }
}

// video의 데이터가 충분할 때만 캔버스 크기를 업데이트하여 불필요한 리사이징을 방지합니다.
function updateCanvas() {
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth; // 1280
      canvas.height = video.videoHeight; // 960
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  requestAnimationFrame(updateCanvas);
}

document.getElementById("switchCam").addEventListener("click", () => {
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  startCamera(currentFacingMode);
});

// 카메라 시작 및 캔버스 업데이트 루프 시작
startCamera(currentFacingMode);
video.addEventListener("loadedmetadata", updateCanvas);
